"""Use the original RAQ-VAE evaluator with native-size checks and accounting.

Only fixed 256x256 pixel denominators and the input-shape guard are generalized.
Model classes, forward/decode paths, codebooks, serialization, channel, LDPC and
quality functions are the original implementations. Exact source diffs are saved.
"""
import difflib
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'RAQ-VAE-main'


def main():
    argv = sys.argv[1:]
    assert '--dataset' in argv
    dataset = Path(argv[argv.index('--dataset') + 1]).resolve()
    shapes = {}
    for path in dataset.glob('*.png'):
        with Image.open(path) as image:
            width, height = image.size
        assert width % 16 == height % 16 == 0
        shapes[path.name] = (1, 3, height, width)
    assert len(shapes) == 24
    pixel_counts = {shape[2] * shape[3] for shape in shapes.values()}
    assert len(pixel_counts) == 1
    pixels = pixel_counts.pop()
    source_path = PROJECT / 'eval_seq2seq_channel.py'
    source = source_path.read_text()
    replacements = [
        ('3 * 256 * 256', '3 * NATIVE_PIXELS'),
        ('256 * 256', 'NATIVE_PIXELS'),
        ('(1, 3, 256, 256)', 'NATIVE_SHAPES[filenames[image_index]]'),
    ]
    adapted = source
    changes = []
    for old, new in replacements:
        count = adapted.count(old)
        assert count > 0, old
        adapted = adapted.replace(old, new)
        changes.append({'old': old, 'new': new, 'occurrences': count})
    artifacts = ROOT / 'adapter_audit'
    artifacts.mkdir(exist_ok=True)
    snapshot = artifacts / f'vae_evaluator_{pixels}_pixels.py'
    snapshot.write_text(adapted)
    (artifacts / f'vae_evaluator_{pixels}_pixels.diff').write_text(''.join(difflib.unified_diff(
        source.splitlines(keepends=True), adapted.splitlines(keepends=True),
        fromfile=str(source_path), tofile=str(snapshot))))
    (artifacts / f'vae_evaluator_{pixels}_pixels.json').write_text(json.dumps({
        'source_file': str(source_path),
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'adapted_source_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
        'pixel_count': pixels, 'bottom_index_count': pixels // 64,
        'top_index_count': pixels // 256,
        'changes': changes,
        'original_source_modified': False,
        'model_and_communication_functions_modified': False,
    }, indent=2) + '\n')
    sys.path.insert(0, str(PROJECT))
    module = types.ModuleType('native_size_vae_evaluator')
    module.__file__ = str(source_path)
    module.NATIVE_PIXELS = pixels
    module.NATIVE_SHAPES = shapes
    sys.modules[module.__name__] = module
    exec(compile(adapted, str(snapshot), 'exec'), module.__dict__)
    from nets.model_ds8_dualk import RAQVAE_TWO_DS8_DUALK
    module.MODEL_CLASS = RAQVAE_TWO_DS8_DUALK
    module.BOTTOM_INDEX_COUNT = pixels // 64
    module.TOP_INDEX_COUNT = pixels // 256
    module.SOURCE_INDEX_COUNT = module.BOTTOM_INDEX_COUNT + module.TOP_INDEX_COUNT
    module.OUTPUT_TAG = 'seq2seq_ds8_dualk_native'
    module.SPATIAL_SCALES = {'bottom': '1/8', 'top': '1/16'}
    module.DUAL_K = True
    # Native console mode reuses the checkpoint in place and avoids large
    # redundant snapshots/clean controls. Observe only final channel outputs.
    output = Path(argv[argv.index('--output') + 1])
    output.mkdir(parents=True, exist_ok=False)
    names = [name for name in os.listdir(dataset) if name in shapes]
    state = {'calls': 0, 'images': []}
    original_load = module.load_file_module

    def observed_load(name, path):
        loaded = original_load(name, path)
        if name == 'raq_reference_quality':
            original_quality = loaded._image_quality

            def observed_quality(real, reconstructed):
                result = original_quality(real, reconstructed)
                index = state['calls']
                state['calls'] += 1
                if index % 2 == 1:
                    filename = names[index // 2]
                    assert tuple(reconstructed.shape) == shapes[filename]
                    module.save_image(((reconstructed.detach().cpu() + 1) / 2).clamp(0, 1), output / filename)
                    state['images'].append(filename)
                return result

            loaded._image_quality = observed_quality
        return loaded

    module.load_file_module = observed_load
    report = module.main([*argv, '--console-only'])
    assert state['calls'] == 48 and state['images'] == names
    (output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    # Retain the standard native channel subdirectory for the caller's copying.
    channel = output / 'channel_awgn_10db'
    channel.mkdir()
    for name in names:
        (output / name).replace(channel / name)


if __name__ == '__main__':
    main()
