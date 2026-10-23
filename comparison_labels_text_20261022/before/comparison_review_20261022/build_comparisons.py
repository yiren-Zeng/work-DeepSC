#!/usr/bin/env python3
"""Select and assemble factual Kodak reconstruction comparisons without enhancement."""

import argparse
import csv
import difflib
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
DIRECTORIES = {
    "Original": ROOT / "Kodak",
    "JPEG": ROOT / "JPEG/generated_images/cbr1over1.2",
    "RAQ-VAE": ROOT / "RAQ-VAE-main/generated_images/20261022_205436_261293_88b35026/k32/awgn_snr10",
    "MDRQ": ROOT / "Ours-RAQ/experiments/eval_images/20261022_140841_k32_wwydw7h7/snr_10",
}
SELECTIONS = {
    "kodim05.png": {
        "roi_xyxy": [320, 224, 464, 368],
        "reason": "Motorcycle front fork, wheel spokes and fender edges: thin structures and strong color boundaries.",
    },
    "kodim21.png": {
        "roi_xyxy": [112, 136, 256, 280],
        "reason": "Lighthouse keeper's house: roof edges, windows and facade; a different scene from motorcycles.",
    },
}
METHODS = tuple(DIRECTORIES)
RECONSTRUCTIONS = METHODS[1:]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_images(name):
    images = {}
    for method, folder in DIRECTORIES.items():
        with Image.open(folder / name) as image:
            if image.mode != "RGB":
                raise ValueError(f"Unexpected mode in {folder / name}: {image.mode}")
            images[method] = image.copy()
    if len({im.size for im in images.values()}) != 1:
        raise ValueError(f"Size mismatch for {name}")
    return images


def psnr(original, reconstruction):
    mse = float(np.mean((original.astype(np.float64) - reconstruction.astype(np.float64)) ** 2))
    return 10 * math.log10(255 ** 2 / mse) if mse else math.inf


def integral(array):
    return np.pad(array.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def window_mean(array, x, y, side):
    return float((array[y + side, x + side] - array[y, x + side]
                  - array[y + side, x] + array[y, x]) / side ** 2)


def candidate_rois(images):
    original = np.asarray(images["Original"], dtype=np.float64)
    errors = {
        method: integral(np.mean((original - np.asarray(images[method], dtype=float)) ** 2, axis=2))
        for method in RECONSTRUCTIONS
    }
    gray = original @ np.array([0.299, 0.587, 0.114])
    dy, dx = np.gradient(gray)
    gradient = integral(dx ** 2 + dy ** 2)
    rows, side = [], 144
    for y in range(16, original.shape[0] - side - 16 + 1, 8):
        for x in range(16, original.shape[1] - side - 16 + 1, 8):
            texture = math.sqrt(max(0, window_mean(gradient, x, y, side)))
            if texture < 12:
                continue
            mse = {key: window_mean(value, x, y, side) for key, value in errors.items()}
            gap = min(10 * math.log10(mse[key] / mse["MDRQ"]) for key in ("JPEG", "RAQ-VAE"))
            rows.append({
                "roi_xyxy": [x, y, x + side, y + side],
                "minimum_psnr_advantage_db": gap,
                "original_gradient_rms": texture,
                "patch_psnr_db": {key: 10 * math.log10(255 ** 2 / value) for key, value in mse.items()},
            })
    picked = []
    for row in sorted(rows, key=lambda r: r["minimum_psnr_advantage_db"], reverse=True):
        x, y = row["roi_xyxy"][:2]
        if all((x - other["roi_xyxy"][0]) ** 2 + (y - other["roi_xyxy"][1]) ** 2 > 72 ** 2
               for other in picked):
            picked.append(row)
        if len(picked) == 6:
            break
    return picked


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def centered(draw, x, y, text, size, fill="#111111"):
    draw.text((x, y), text, font=font(size), fill=fill, anchor="mm")


def analyze():
    inventories = {key: sorted(p.name for p in folder.glob("*.png")) for key, folder in DIRECTORIES.items()}
    names = inventories["Original"]
    if len(names) != 24 or any(inventory != names for inventory in inventories.values()):
        raise ValueError("Expected exactly 24 matching PNG names in every directory")
    records, source_manifest, rois = [], [], {}
    for name in names:
        images = read_images(name)
        arrays = {key: np.asarray(value) for key, value in images.items()}
        scores = {method: psnr(arrays["Original"], arrays[method]) for method in RECONSTRUCTIONS}
        gap = scores["MDRQ"] - max(scores["JPEG"], scores["RAQ-VAE"])
        records.append({"image": name, **{f"{key}_psnr_db": value for key, value in scores.items()},
                        "MDRQ_minimum_advantage_db": gap})
        for method, folder in DIRECTORIES.items():
            source_manifest.append({"method": method, "path": str(folder / name),
                                    "size_wh": list(images[method].size), "sha256": sha256(folder / name)})
        rois[name] = candidate_rois(images)
        print(f"ANALYZE {name}: JPEG={scores['JPEG']:.4f}, RAQ-VAE={scores['RAQ-VAE']:.4f}, "
              f"MDRQ={scores['MDRQ']:.4f}, minimum gap={gap:.4f} dB", flush=True)
    records.sort(key=lambda row: row["MDRQ_minimum_advantage_db"], reverse=True)
    with (AUDIT / "all_image_psnr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(AUDIT / "source_manifest.json", source_manifest)
    write_json(AUDIT / "roi_candidates.json", rois)
    candidates = []
    for name, selection in SELECTIONS.items():
        candidates.append((name, selection["roi_xyxy"], "selected"))
        candidates.extend((name, row["roi_xyxy"], f"alternative {i + 1}") for i, row in enumerate(rois[name][:2]))
    preview = Image.new("RGB", (1248, len(candidates) * 354), "white")
    draw = ImageDraw.Draw(preview)
    for row_index, (name, roi, label) in enumerate(candidates):
        images = read_images(name)
        top = row_index * 354
        for column, method in enumerate(METHODS):
            centered(draw, 156 + column * 312, top + 24, f"{name[:7]} | {method}", 19)
            preview.paste(images[method].crop(roi).resize((288, 288), Image.Resampling.NEAREST),
                          (12 + column * 312, top + 46))
        centered(draw, 624, top + 343, f"{label}: ROI={roi}", 16)
    preview.save(AUDIT / "detail_candidates.png")
    print("Analysis and diagnostic crop preview saved.", flush=True)


def metric_function():
    source = ROOT / "Ours-RAQ/utils/metrics.py"
    spec = importlib.util.spec_from_file_location("comparison_existing_metrics", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ms_ssim, source


def render():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    ms_ssim, metric_source = metric_function()
    source_manifest = json.loads((AUDIT / "source_manifest.json").read_text(encoding="utf-8"))
    if any(sha256(Path(row["path"])) != row["sha256"] for row in source_manifest):
        raise ValueError("An input image changed since analysis")
    selected = []
    output_paths, verified_panels = [], 0
    for name, selection in SELECTIONS.items():
        images = read_images(name)
        arrays = {key: np.asarray(value) for key, value in images.items()}
        if images["Original"].size != (768, 512):
            raise ValueError("This layout expects landscape Kodak images at native 768 x 512")
        roi = selection["roi_xyxy"]
        x1, y1, x2, y2 = roi
        if x2 - x1 != y2 - y1 or not (0 <= x1 < x2 <= 768 and 0 <= y1 < y2 <= 512):
            raise ValueError(f"Invalid ROI: {roi}")
        metrics = {}
        for method in RECONSTRUCTIONS:
            score = float(ms_ssim(arrays["Original"] / 255.0, arrays[method] / 255.0))
            if not 0 < score < 1:
                raise ValueError(f"Unexpected MS-SSIM {score}")
            metrics[method] = {
                "psnr_db": psnr(arrays["Original"], arrays[method]),
                "ms_ssim": score,
                "ms_ssim_db": -10 * math.log10(1 - score),
                "roi_psnr_db": psnr(arrays["Original"][y1:y2, x1:x2], arrays[method][y1:y2, x1:x2]),
            }
            print(f"METRIC {name} {method}: {json.dumps(metrics[method])}", flush=True)
        for key in ("psnr_db", "ms_ssim", "roi_psnr_db"):
            if not all(metrics["MDRQ"][key] > metrics[method][key] for method in ("JPEG", "RAQ-VAE")):
                raise ValueError(f"MDRQ does not lead on {name}: {key}")
        width, height = 3272, 1580
        panel_width, gap, margin, full_top, crop_top, crop_side = 768, 40, 40, 150, 770, 672
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        expected_crops = []
        for column, method in enumerate(METHODS):
            left = margin + column * (panel_width + gap)
            center = left + panel_width / 2
            centered(draw, center, 103, method, 44)
            canvas.paste(images[method], (left, full_top))
            draw.rectangle((left, full_top, left + 767, full_top + 511), outline="#666666", width=2)
            draw.rectangle((left + x1, full_top + y1, left + x2 - 1, full_top + y2 - 1),
                           outline="#ff2020", width=4)
            crop = images[method].crop(roi).resize((crop_side, crop_side), Image.Resampling.NEAREST)
            crop_left = left + (panel_width - crop_side) // 2
            canvas.paste(crop, (crop_left, crop_top))
            expected_crops.append((crop_left, crop_top, np.asarray(crop)))
            if method != "Original":
                values = metrics[method]
                text = f"PSNR={values['psnr_db']:.2f} dB | MS-SSIM={values['ms_ssim_db']:.2f} dB"
                centered(draw, center, 1521, text, 31)
        png = ROOT / f"{Path(name).stem}_comparison_20261022.png"
        pdf = png.with_suffix(".pdf")
        canvas.save(png, dpi=(300, 300))
        figure = Figure(figsize=(width / 300, height / 300), dpi=300)
        FigureCanvasAgg(figure)
        axes = figure.add_axes((0, 0, 1, 1))
        axes.imshow(np.asarray(canvas), interpolation="none")
        axes.set_axis_off()
        figure.savefig(pdf, dpi=300, metadata={"Title": f"{name}: Original, JPEG, RAQ-VAE, MDRQ",
                                              "Creator": "build_comparisons.py"})
        with Image.open(png) as saved:
            if saved.size != (width, height):
                raise AssertionError("Wrong figure size")
            for crop_left, crop_y, expected in expected_crops:
                actual = np.asarray(saved.crop((crop_left, crop_y, crop_left + crop_side, crop_y + crop_side)))
                if not np.array_equal(actual, expected):
                    raise AssertionError("Saved zoom panel differs from the source crop")
                verified_panels += 1
            for column, method in enumerate(METHODS):
                left = margin + column * (panel_width + gap)
                actual = np.asarray(saved.crop((left, full_top, left + 768, full_top + 512)))
                mask = np.ones((512, 768), dtype=bool)
                mask[:2] = mask[-2:] = False
                mask[:, :2] = mask[:, -2:] = False
                mask[y1:y2, x1:x2] = False
                if not np.array_equal(actual[mask], arrays[method][mask]):
                    raise AssertionError("Full image pixels changed outside the annotated border and ROI")
                verified_panels += 1
        for path in (png, pdf):
            output_paths.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
        selected.append({"image": name, **selection, "metrics": metrics, "figure_png": str(png), "figure_pdf": str(pdf)})
        print(f"SAVED {png}\nSAVED {pdf}", flush=True)
    unchanged = all(sha256(Path(row["path"])) == row["sha256"] for row in source_manifest)
    if not unchanged:
        raise AssertionError("Input source image hash changed")
    write_json(AUDIT / "selection_and_metrics.json", {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "metric_source": str(metric_source), "metric_source_sha256": sha256(metric_source),
        "psnr_definition": "RGB full-image MSE; 10*log10(255^2/MSE), unrounded source PNG pixels",
        "ms_ssim_definition": "Existing Ours-RAQ/utils/metrics.py::ms_ssim on RGB float64 in [0,1], five scales",
        "ms_ssim_db_definition": "-10*log10(1-MS-SSIM); raw MS-SSIM also retained",
        "image_operations": "Lossless PNG input, native-size full images, identical ROI coordinates, nearest-neighbor zoom; no enhancement",
        "selection_scope": "Two deliberately selected illustrative examples; not an average over all 24 images",
        "selections": selected,
    })
    write_json(AUDIT / "verification.json", {
        "matching_image_counts": {method: 24 for method in METHODS},
        "source_images_verified_unchanged": len(source_manifest) if unchanged else 0,
        "panels_verified_against_source_pixels": verified_panels,
        "MDRQ_leads_selected_full_image_PSNR_MS_SSIM_and_ROI_PSNR": True,
        "zoom_interpolation": "nearest neighbor, identical for all methods",
        "roi_shape": [144, 144], "displayed_zoom_shape": [672, 672],
        "output_png_size_wh": [3272, 1580], "outputs": output_paths,
    })
    print(f"VERIFIED {len(source_manifest)} unchanged source images, {verified_panels} source-derived panels.", flush=True)


def review():
    patch_path = AUDIT / "changes.patch"
    status_path = AUDIT / "git_status_after.txt"
    manifest_path = AUDIT / "artifacts_manifest.json"
    for path in (patch_path, status_path, manifest_path):
        path.touch(exist_ok=True)
    status = subprocess.run(
        ["git", "-c", "core.quotepath=false", "status", "--short", "--untracked-files=all"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    status_path.write_text(status, encoding="utf-8")
    figures = [ROOT / f"{Path(name).stem}_comparison_20261022{extension}"
               for name in SELECTIONS for extension in (".png", ".pdf")]
    paths = sorted([path for path in AUDIT.rglob("*") if path.is_file()] + figures)
    owned = {str(path.relative_to(ROOT)) for path in paths}
    before = (AUDIT / "git_status_before.txt").read_text(encoding="utf-8").splitlines()
    previous_owned = [line for line in before if line[3:] in owned]
    if previous_owned:
        raise ValueError(f"An output was present before this task: {previous_owned}")
    missing = [path for path in owned if f"?? {path}" not in status.splitlines()]
    ignored = set()
    if missing:
        result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
                                input="\n".join(missing) + "\n", capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise ValueError(result.stderr)
        ignored = set(result.stdout.splitlines())
        if set(missing) - ignored:
            raise ValueError(f"Expected new outputs absent from status: {sorted(set(missing) - ignored)}")
    tracked_diff = subprocess.run(
        ["git", "diff", "--", *sorted(owned)], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    if tracked_diff:
        raise ValueError("Unexpected tracked diff in this task's output paths")
    patch = []
    for path in paths:
        if path.suffix not in (".py", ".md", ".json", ".csv") or path == manifest_path:
            continue
        relative = str(path.relative_to(ROOT))
        patch.append(f"diff --git a/{relative} b/{relative}\nnew file mode 100644\n")
        patch.extend(difflib.unified_diff([], path.read_text(encoding="utf-8").splitlines(keepends=True),
                                           fromfile="/dev/null", tofile=f"b/{relative}"))
    patch_path.write_text("".join(patch), encoding="utf-8")
    outside_before = {line for line in before if line[3:] not in owned}
    outside_after = {line for line in status.splitlines() if line[3:] not in owned}
    write_json(manifest_path, {
        "command": f"python -u {__file__} --stage review",
        "scope": "New outputs only; existing workspace modifications are excluded",
        "tracked_changes_in_output_paths": False,
        "new_untracked_outputs": sorted(owned - ignored),
        "new_outputs_ignored_by_git": sorted(ignored),
        "outside_scope_status_added": sorted(outside_after - outside_before),
        "outside_scope_status_removed": sorted(outside_before - outside_after),
        "patch_covers": "Complete new .py/.md/.csv/.json contents except this manifest; logs and binary images are linked separately",
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in paths if path != manifest_path],
        "manifest_self": str(manifest_path),
    })
    print(f"REVIEW {len(paths)} new files: {len(owned - ignored)} untracked, {len(ignored)} ignored logs; "
          "tracked diff empty for all output paths.", flush=True)
    print(f"OUTSIDE SCOPE status additions={len(outside_after - outside_before)}, "
          f"removals={len(outside_before - outside_after)}", flush=True)
    print(f"PATCH {patch_path}\nMANIFEST {manifest_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("analyze", "render", "review", "all"), default="all")
    args = parser.parse_args()
    print(f"RUN python {__file__} --stage {args.stage}", flush=True)
    if args.stage in ("analyze", "all"):
        analyze()
    if args.stage in ("render", "all"):
        render()
    if args.stage in ("review", "all"):
        review()
