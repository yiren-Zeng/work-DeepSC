"""Observe original decoder outputs; configure only the native test dataset."""
import os
import sys
from pathlib import Path


def install():
    if os.environ.get('NATIVE_EXPORT_ENABLE') != '1':
        return
    if Path(sys.argv[0]).name != os.environ.get('NATIVE_EXPORT_MAIN'):
        return
    import hashlib
    import json
    import data.datasets as datasets
    import evaluation.quality as quality
    from torchvision.utils import save_image
    project = Path(os.environ['NATIVE_EXPORT_PROJECT']).resolve()
    assert Path(datasets.__file__).resolve() == project / 'data/datasets.py'
    assert Path(quality.__file__).resolve() == project / 'evaluation/quality.py'
    if project.name == 'VQ-DeepSC':
        from config_last2_test import Last2CascadeTestConfig
        Last2CascadeTestConfig.TEST_DATASET_PATH = os.environ['NATIVE_EXPORT_DATASET']
    names = json.loads(Path(os.environ['NATIVE_EXPORT_NAMES']).read_text())
    shapes = json.loads(Path(os.environ['NATIVE_EXPORT_SHAPES']).read_text())
    output = Path(os.environ['NATIVE_EXPORT_DIR'])
    state = {'status': 'installed', 'images': [], 'seed': 42,
             'metric_return_values_unchanged': True,
             'no_resize_crop_tiling_or_enhancement': True,
             'original_quality_module': str(quality.__file__)}

    def flush():
        (output / 'image_export.json').write_text(json.dumps(state, indent=2) + '\n')

    original_loader = datasets.get_dataloader

    def observed_loader(*args, **kwargs):
        loader = original_loader(*args, **kwargs)
        assert Path(loader.dataset.root_dir).resolve() == Path(os.environ['NATIVE_EXPORT_DATASET']).resolve()
        assert list(loader.dataset.image_files) == names
        state['loader_image_order_verified'] = True
        flush()
        return loader

    original_quality = quality._image_quality

    def observed_quality(real, reconstructed):
        result = original_quality(real, reconstructed)
        idx = len(state['images'])
        assert idx < len(names)
        name = names[idx]
        height, width = shapes[name]
        assert tuple(real.shape) == tuple(reconstructed.shape) == (1, 3, height, width)
        target = output / name
        save_image(((reconstructed.detach().cpu() + 1) / 2).clamp(0, 1), target)
        state['images'].append({'image': name, 'shape_hw': [height, width],
                                'psnr': float(result[1]), 'ms_ssim': float(result[0]),
                                'png_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        state['status'] = 'complete' if len(state['images']) == len(names) else 'exporting'
        flush()
        return result

    datasets.get_dataloader = observed_loader
    quality._image_quality = observed_quality
    flush()


install()
