"""Export existing JPEG-2000 reconstructions as RGB PNG, preserving each size."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'output_jpeg')
    parser.add_argument('--output', type=Path, default=ROOT / 'generated_images/cbr1over1.2')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    inputs = [source / f'val_0000{i:02d}.jp2' for i in range(1, 25)]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing existing reconstructions: {missing}')
    evaluator = ROOT / 'test_jpeg.py'
    evaluator_hash = sha256(evaluator)
    source_hashes = {p.name: sha256(p) for p in inputs}
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(inputs, start=1):
        target = output / f'kodim{index:02d}.png'
        with Image.open(path) as decoded:
            # This is exactly the RGB conversion used by the original evaluator.
            rgb = decoded.convert('RGB')
            original_size = rgb.size
            original_pixels = rgb.tobytes()
            rgb.save(target, format='PNG')
        with Image.open(target) as exported:
            exported.load()
            assert exported.mode == 'RGB' and exported.size == original_size
            assert exported.tobytes() == original_pixels, target
        assert sha256(path) == source_hashes[path.name], path
        records.append({'source_file': path.name, 'image': target.name,
                        'size': list(original_size), 'mode': 'RGB',
                        'jp2_sha256': source_hashes[path.name],
                        'png_sha256': sha256(target),
                        'decoded_rgb_pixels_identical': True})
    assert sha256(evaluator) == evaluator_hash
    manifest = {'status': 'complete', 'num_images': 24,
                'source_directory': str(source), 'output_directory': str(output),
                'cbr_supplied_by_user': '1/32',
                'operation': 'Decode existing JP2 as RGB and save lossless PNG; no new compression or channel simulation',
                'evaluator': str(evaluator), 'evaluator_sha256_unchanged': evaluator_hash,
                'existing_jp2_files_unchanged': True, 'images': records}
    (output / 'image_export.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'README.md').write_text(
        '本目录保存现有 JPEG-2000 重建结果的 24 张 RGB PNG，逐张保持输入 JP2 的原始宽高。\n\n'
        '输入为 `JPEG/output_jpeg/val_000001.jp2` 至 `val_000024.jp2`；'
        '按照原 `test_jpeg.py` 的对应关系，输出命名为 `kodim01.png` 至 `kodim24.png`。'
        'CBR=1/32 使用用户给出的现有实验配置，本次只导出这些已有文件。\n\n'
        'PNG 与原评测读取 JP2 后得到的 RGB 像素完全一致，已逐张验证；'
        '没有重新压缩、重新模拟信道、调整尺寸或增强图像，原 JP2 和测试脚本均未修改。\n\n'
        '前面选择的两个同编号样本为 [kodim02.png](kodim02.png) 和 '
        '[kodim03.png](kodim03.png)。完整映射和校验结果见 `image_export.json`。\n\n'
        '重新导出：\n\n'
        '```bash\n'
        '/home/yi/.conda/envs/work/bin/python /workspace/yi/work/JPEG/export_reconstruction_png.py\n'
        '```\n', encoding='utf-8')
    print(json.dumps({'images': len(records), 'output': str(output),
                      'rgb_pixels_identical': True,
                      'test_script_and_jp2_files_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
