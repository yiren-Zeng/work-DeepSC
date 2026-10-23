#!/usr/bin/env python3
"""Review lettering candidates and update comparison labels without source-image edits."""

import argparse
import difflib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
HELPERS = ROOT / "comparison_review_20261022/build_comparisons.py"
spec = importlib.util.spec_from_file_location("comparison_helpers", HELPERS)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
DISPLAY = {"Original": "Original", "JPEG": "JPEG-2000", "RAQ-VAE": "RAQJSCC", "MDRQ": "MDRQ"}
OLD_FIGURES = [ROOT / name for name in (
    "kodim05_comparison_20261022.png", "kodim21_comparison_20261022.png",
    "kodim05_comparison_20261022_v2.png", "kodim14_comparison_20261022_v2.png",
)]
OLD_SCRIPTS = [HELPERS, ROOT / "comparison_revision_20261022/revise_comparisons.py"]
CANDIDATES = [
    ("kodim08.png", "full fine-print sign", [544, 288, 656, 400]),
    ("kodim08.png", "red initial and fine strokes", [536, 296, 632, 392]),
    ("kodim08.png", "Alfred sign", [664, 288, 752, 376]),
    ("kodim08.png", "fabrikation strokes", [584, 304, 664, 384]),
    ("kodim19.png", "warning sign", [396, 440, 460, 504]),
    ("kodim09.png", "sail digits", [376, 376, 472, 472]),
    ("kodim10.png", "sail digits", [152, 28, 312, 188]),
    ("kodim24.png", "painted wall script", [376, 272, 440, 336]),
]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_sources():
    entries = json.loads((AUDIT / "source_manifest.json").read_text(encoding="utf-8"))
    for entry in entries:
        if base.sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Changed source: {entry['path']}")
    return len(entries)


def prepare():
    manifest_path = AUDIT / "before_manifest.json"
    if manifest_path.exists():
        raise ValueError("Before-operation snapshot already exists; refusing to replace it")
    source_records = []
    names = sorted(path.name for path in base.DIRECTORIES["Original"].glob("*.png"))
    if len(names) != 24:
        raise ValueError("Expected 24 Kodak images")
    for method, folder in base.DIRECTORIES.items():
        if sorted(path.name for path in folder.glob("*.png")) != names:
            raise ValueError(f"Mismatched input inventory: {folder}")
        for name in names:
            path = folder / name
            source_records.append({"path": str(path), "sha256": base.sha256(path)})
    write_json(AUDIT / "source_manifest.json", source_records)
    targets = OLD_SCRIPTS + [path for png in OLD_FIGURES for path in (png, png.with_suffix(".pdf"))]
    previous = []
    for path in targets:
        snapshot = AUDIT / "before" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot)
        previous.append({"path": str(path), "snapshot": str(snapshot), "sha256": base.sha256(path)})
    write_json(manifest_path, previous)
    print(f"PREPARED {len(source_records)} source-image hashes, {len(previous)} before-operation snapshots.", flush=True)


def inspect():
    verify_sources()
    records = []
    for page_index in range(math.ceil(len(CANDIDATES) / 4)):
        choices = CANDIDATES[page_index * 4:(page_index + 1) * 4]
        canvas = Image.new("RGB", (1600, len(choices) * 400), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, (name, description, roi) in enumerate(choices):
            images = base.read_images(name)
            scores = {}
            for column, method in enumerate(base.METHODS):
                crop = images[method].crop(roi)
                left, top = 400 * column + 34, 400 * row_index + 40
                canvas.paste(crop.resize((332, 332), Image.Resampling.NEAREST), (left, top))
                base.centered(draw, column * 400 + 200, row_index * 400 + 21,
                              f"{name[:7]} | {DISPLAY[method]}", 21)
                if method != "Original":
                    scores[method] = base.psnr(np.asarray(images["Original"].crop(roi)), np.asarray(crop))
            base.centered(draw, 800, row_index * 400 + 386, f"{description} {roi}", 18)
            records.append({"image": name, "description": description, "roi_xyxy": roi, "roi_psnr_db": scores,
                            "minimum_MDRQ_advantage_db": scores["MDRQ"] - max(scores["JPEG"], scores["RAQ-VAE"])})
        path = AUDIT / f"lettering_candidates_{page_index + 1}.png"
        canvas.save(path)
        print(f"PREVIEW {path}", flush=True)
    write_json(AUDIT / "candidate_regions.json", records)
    for record in records:
        print(json.dumps(record), flush=True)


def save_pdf(canvas, path, title):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    width, height = canvas.size
    figure = Figure(figsize=(width / 300, height / 300), dpi=300)
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.imshow(np.asarray(canvas), interpolation="none")
    axes.set_axis_off()
    figure.savefig(path, dpi=300, metadata={"Title": title, "Creator": "update_labels_and_text.py"})


def labels():
    verify_sources()
    before = {entry["path"]: entry for entry in json.loads((AUDIT / "before_manifest.json").read_text(encoding="utf-8"))}
    outputs = []
    for png in OLD_FIGURES:
        entry = before[str(png)]
        if base.sha256(png) != entry["sha256"]:
            raise ValueError(f"Existing figure changed since the before-operation snapshot: {png}")
        with Image.open(png) as original:
            canvas = original.copy()
        original_pixels = np.asarray(canvas).copy()
        if canvas.size != (3272, 1580) or canvas.mode != "RGB":
            raise ValueError(f"Unexpected figure format: {png}")
        draw = ImageDraw.Draw(canvas)
        allowed = np.zeros((1580, 3272), dtype=bool)
        for column, method in ((1, "JPEG"), (2, "RAQ-VAE")):
            left = 40 + column * 808
            draw.rectangle((left, 60, left + 767, 140), fill="white")
            base.centered(draw, left + 384, 103, DISPLAY[method], 44)
            allowed[60:141, left:left + 768] = True
        canvas.save(png, dpi=(300, 300))
        with Image.open(png) as result:
            after = np.asarray(result)
        if not np.array_equal(original_pixels[~allowed], after[~allowed]):
            raise AssertionError("An existing figure changed outside its two method titles")
        changed = int(np.count_nonzero(np.any(original_pixels != after, axis=2)))
        if changed == 0:
            raise AssertionError("Expected updated title pixels")
        pdf = png.with_suffix(".pdf")
        save_pdf(canvas, pdf, "Original | JPEG-2000 | RAQJSCC | MDRQ")
        for path in (png, pdf):
            outputs.append({"path": str(path), "before_sha256": before[str(path)]["sha256"],
                            "after_sha256": base.sha256(path), "bytes": path.stat().st_size})
        print(f"RELABELLED {png.name}: {changed} changed pixels, all within method-title regions.", flush=True)
    verify_sources()
    for path in OLD_SCRIPTS:
        if base.sha256(path) != before[str(path)]["sha256"]:
            raise AssertionError("Historical generator script unexpectedly changed")
    write_json(AUDIT / "renamed_existing_figures.json", {
        "display_labels": DISPLAY, "existing_png_figures_updated": 4,
        "verification": "Every pixel outside the two method-title rectangles is identical to the before-operation snapshot",
        "allowed_changed_rectangles_xyxy": [[848, 60, 1616, 141], [1656, 60, 2424, 141]],
        "source_images_verified_unchanged": 96, "historical_generator_scripts_unchanged": True,
        "outputs": outputs,
    })


def new_figure(name, roi):
    verify_sources()
    images = base.read_images(name)
    arrays = {method: np.asarray(image) for method, image in images.items()}
    image_width, image_height = images["Original"].size
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height and x2 - x1 == y2 - y1):
        raise ValueError(f"Invalid square crop: {roi}")
    metrics = {}
    ms_ssim, metric_source = base.metric_function()
    for method in base.METHODS[1:]:
        score = float(ms_ssim(arrays["Original"] / 255.0, arrays[method] / 255.0))
        if not 0 < score < 1:
            raise ValueError(f"Unexpected MS-SSIM: {score}")
        metrics[DISPLAY[method]] = {
            "source_method_key": method,
            "psnr_db": base.psnr(arrays["Original"], arrays[method]),
            "ms_ssim": score, "ms_ssim_db": -10 * math.log10(1 - score),
            "roi_psnr_db": base.psnr(arrays["Original"][y1:y2, x1:x2], arrays[method][y1:y2, x1:x2]),
        }
        print(f"METRIC {name} {DISPLAY[method]}: {json.dumps(metrics[DISPLAY[method]])}", flush=True)
    for key in ("psnr_db", "ms_ssim", "roi_psnr_db"):
        if not all(metrics["MDRQ"][key] > metrics[method][key] for method in ("JPEG-2000", "RAQJSCC")):
            raise ValueError(f"MDRQ does not lead the selected example on {key}")
    crop_side = round(image_width * 0.875)
    crop_top = 150 + image_height + 108
    metric_y = crop_top + crop_side + 79
    width, height = 4 * image_width + 200, metric_y + 59
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    crop_checks = []
    for column, method in enumerate(base.METHODS):
        left = 40 + column * (image_width + 40)
        center = left + image_width / 2
        base.centered(draw, center, 103, DISPLAY[method], 44)
        canvas.paste(images[method], (left, 150))
        draw.rectangle((left, 150, left + image_width - 1, 150 + image_height - 1), outline="#666666", width=2)
        draw.rectangle((left + x1, 150 + y1, left + x2 - 1, 150 + y2 - 1), outline="#ff2020", width=4)
        crop = images[method].crop(roi).resize((crop_side, crop_side), Image.Resampling.NEAREST)
        crop_left = left + (image_width - crop_side) // 2
        canvas.paste(crop, (crop_left, crop_top))
        crop_checks.append((crop_left, np.asarray(crop)))
        if method != "Original":
            values = metrics[DISPLAY[method]]
            text = f"PSNR={values['psnr_db']:.2f} dB | MS-SSIM={values['ms_ssim_db']:.2f} dB"
            base.centered(draw, center, metric_y, text, round(31 * image_width / 768))
    png = ROOT / f"{Path(name).stem}_comparison_20261022.png"
    pdf = png.with_suffix(".pdf")
    if png.exists() or pdf.exists():
        raise ValueError(f"New figure path already exists: {png}")
    canvas.save(png, dpi=(300, 300))
    save_pdf(canvas, pdf, f"{name}: Original | JPEG-2000 | RAQJSCC | MDRQ")
    with Image.open(png) as saved:
        for left, expected in crop_checks:
            actual = np.asarray(saved.crop((left, crop_top, left + crop_side, crop_top + crop_side)))
            if not np.array_equal(expected, actual):
                raise AssertionError("New zoom panel differs from source crop")
        for column, method in enumerate(base.METHODS):
            left = 40 + column * (image_width + 40)
            actual = np.asarray(saved.crop((left, 150, left + image_width, 150 + image_height)))
            mask = np.ones((image_height, image_width), dtype=bool)
            mask[:2] = mask[-2:] = False
            mask[:, :2] = mask[:, -2:] = False
            mask[y1:y2, x1:x2] = False
            if not np.array_equal(actual[mask], arrays[method][mask]):
                raise AssertionError("New full-image panel differs from source outside annotations")
    verify_sources()
    write_json(AUDIT / "new_figure_metrics.json", {
        "image": name, "roi_xyxy": roi, "display_labels": DISPLAY,
        "metric_source": str(metric_source), "metric_source_sha256": base.sha256(metric_source),
        "psnr_definition": "RGB full-image MSE; 10*log10(255^2/MSE)",
        "ms_ssim_definition": "Existing five-scale Ours-RAQ/utils/metrics.py::ms_ssim on RGB in [0,1]",
        "ms_ssim_db_definition": "-10*log10(1-MS-SSIM), raw score retained",
        "interpretation": "Selected lettering example; visual clarity and source fidelity, not an OCR accuracy measurement",
        "metrics": metrics, "source_images_verified_unchanged": 96,
        "new_figure_panels_verified": 8, "figure_size_wh": [width, height],
        "roi_size_wh": [x2 - x1, y2 - y1], "zoom_size_wh": [crop_side, crop_side],
        "zoom_interpolation": "nearest neighbor, identical across methods",
        "outputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": base.sha256(path)} for path in (png, pdf)],
    })
    print(f"SAVED {png}\nSAVED {pdf}\nVERIFIED 8 new figure panels, 96 unchanged source images.", flush=True)


def review():
    verify_sources()
    before_entries = json.loads((AUDIT / "before_manifest.json").read_text(encoding="utf-8"))
    for entry in before_entries:
        if base.sha256(Path(entry["snapshot"])) != entry["sha256"]:
            raise ValueError(f"Before-operation snapshot changed: {entry['snapshot']}")
    relabelled = json.loads((AUDIT / "renamed_existing_figures.json").read_text(encoding="utf-8"))
    new = json.loads((AUDIT / "new_figure_metrics.json").read_text(encoding="utf-8"))
    existing_paths = [Path(entry["path"]) for entry in relabelled["outputs"]]
    new_output_paths = [Path(entry["path"]) for entry in new["outputs"]]
    for entry in relabelled["outputs"]:
        if base.sha256(Path(entry["path"])) != entry["after_sha256"]:
            raise ValueError(f"Updated figure changed: {entry['path']}")
    for entry in new["outputs"]:
        if base.sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"New figure changed: {entry['path']}")
    patch_path, status_path, manifest_path = (AUDIT / name for name in ("changes.patch", "git_status_after.txt", "artifacts_manifest.json"))
    for path in (patch_path, status_path, manifest_path):
        path.touch(exist_ok=True)
    status = subprocess.run(["git", "-c", "core.quotepath=false", "status", "--short", "--untracked-files=all"],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout
    status_path.write_text(status, encoding="utf-8")
    new_paths = sorted([path for path in AUDIT.rglob("*") if path.is_file()] + new_output_paths)
    all_paths = sorted(new_paths + existing_paths)
    owned = {str(path.relative_to(ROOT)) for path in all_paths}
    new_relative = {str(path.relative_to(ROOT)) for path in new_paths}
    before_status = (AUDIT / "git_status_before.txt").read_text(encoding="utf-8").splitlines()
    if any(line[3:] in new_relative for line in before_status):
        raise ValueError("A supposed new file existed before this task")
    untracked = {line[3:] for line in status.splitlines() if line.startswith("?? ")}
    missing = new_relative - untracked
    ignored = set()
    if missing:
        result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
                                input="\n".join(sorted(missing)) + "\n", capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise ValueError(result.stderr)
        ignored = set(result.stdout.splitlines())
        if missing - ignored:
            raise ValueError(f"New outputs missing from Git status: {missing - ignored}")
    tracked_diff = subprocess.run(["git", "diff", "--", *sorted(owned)], cwd=ROOT,
                                  check=True, capture_output=True, text=True).stdout
    if tracked_diff:
        raise ValueError("Unexpected tracked changes; before-operation review is required")
    patch = []
    for path in new_paths:
        if path.suffix not in (".py", ".md", ".json") or path == manifest_path:
            continue
        relative = str(path.relative_to(ROOT))
        patch.append(f"diff --git a/{relative} b/{relative}\nnew file mode 100644\n")
        patch.extend(difflib.unified_diff([], path.read_text(encoding="utf-8").splitlines(keepends=True),
                                           fromfile="/dev/null", tofile=f"b/{relative}"))
    patch_path.write_text("".join(patch), encoding="utf-8")
    outside_before = {line for line in before_status if line[3:] not in owned}
    outside_after = {line for line in status.splitlines() if line[3:] not in owned}
    write_json(manifest_path, {
        "command": f"python -u {__file__} --stage review",
        "existing_files_modified": relabelled["outputs"],
        "new_untracked_outputs": sorted(new_relative - ignored), "new_outputs_ignored_by_git": sorted(ignored),
        "outside_scope_status_added": sorted(outside_after - outside_before),
        "outside_scope_status_removed": sorted(outside_before - outside_after),
        "patch_covers": "Complete new .py/.md/.json contents except this manifest; binary before/after files and hashes are retained",
        "files": [{"path": str(path), "action": "modified" if path in existing_paths else "new",
                   "bytes": path.stat().st_size, "sha256": base.sha256(path)}
                  for path in all_paths if path != manifest_path],
        "manifest_self": str(manifest_path),
    })
    print(f"REVIEW {len(existing_paths)} existing files modified; {len(new_paths)} new files "
          f"({len(new_relative - ignored)} untracked, {len(ignored)} ignored logs).", flush=True)
    print(f"OUTSIDE SCOPE status additions={len(outside_after - outside_before)}, removals={len(outside_before - outside_after)}", flush=True)
    print(f"PATCH {patch_path}\nMANIFEST {manifest_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "inspect", "labels", "new", "review"))
    parser.add_argument("--image")
    parser.add_argument("--roi", nargs=4, type=int)
    arguments = parser.parse_args()
    print(f"RUN python {__file__} --stage {arguments.stage}", flush=True)
    if arguments.stage == "new":
        if not arguments.image or not arguments.roi:
            parser.error("--stage new requires --image and --roi")
        new_figure(arguments.image, arguments.roi)
    else:
        {"prepare": prepare, "inspect": inspect, "labels": labels, "review": review}[arguments.stage]()
