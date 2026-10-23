#!/usr/bin/env python3
"""Move kodim10's comparison crop from sail numbers to user-marked hull lettering."""

import argparse
import difflib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
spec = importlib.util.spec_from_file_location("existing_comparison_helpers", ROOT / "comparison_review_20261022/build_comparisons.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
DISPLAY = {"Original": "Original", "JPEG": "JPEG-2000", "RAQ-VAE": "RAQJSCC", "MDRQ": "MDRQ"}
PNG = ROOT / "kodim10_comparison_20261022.png"
PDF = PNG.with_suffix(".pdf")
OLD_ROI = [152, 28, 312, 188]
NEW_ROI = [224, 624, 304, 704]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_sources():
    entries = json.loads((AUDIT / "source_manifest.json").read_text(encoding="utf-8"))
    if any(base.sha256(Path(entry["path"])) != entry["sha256"] for entry in entries):
        raise ValueError("Source images changed")
    return len(entries)


def prepare():
    manifest = AUDIT / "before_manifest.json"
    if manifest.exists():
        raise ValueError("Refusing to replace the before-operation snapshot")
    old_metrics = json.loads((ROOT / "comparison_labels_text_20261022/new_figure_metrics.json").read_text(encoding="utf-8"))
    for entry in old_metrics["outputs"]:
        if base.sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Existing comparison differs from its recorded version")
    source_entries = json.loads((ROOT / "comparison_labels_text_20261022/source_manifest.json").read_text(encoding="utf-8"))
    write_json(AUDIT / "source_manifest.json", source_entries)
    verify_sources()
    snapshots = []
    for path in (PNG, PDF):
        snapshot = AUDIT / "before" / path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot)
        snapshots.append({"path": str(path), "snapshot": str(snapshot), "sha256": base.sha256(path)})
    write_json(manifest, snapshots)
    write_json(AUDIT / "other_figures_before.json", [
        {"path": str(path), "sha256": base.sha256(path)}
        for path in sorted(ROOT.glob("kodim*_comparison_20261022*")) if path.is_file() and path not in (PNG, PDF)
    ])
    images = base.read_images("kodim10.png")
    candidates = [[216, 612, 328, 724], [216, 616, 312, 712], NEW_ROI]
    preview = Image.new("RGB", (1440, len(candidates) * 390), "white")
    draw = ImageDraw.Draw(preview)
    for row, roi in enumerate(candidates):
        for column, method in enumerate(base.METHODS):
            base.centered(draw, column * 360 + 180, row * 390 + 21, DISPLAY[method], 22)
            preview.paste(images[method].crop(roi).resize((320, 320), Image.Resampling.NEAREST),
                          (column * 360 + 20, row * 390 + 44))
        base.centered(draw, 720, row * 390 + 378, f"ROI={roi}", 18)
    preview.save(AUDIT / "hull_crop_candidates.png")
    print("PREPARED two complete before-operation snapshots, 96 source hashes and hull-crop preview.", flush=True)


def render():
    verify_sources()
    snapshots = json.loads((AUDIT / "before_manifest.json").read_text(encoding="utf-8"))
    for entry in snapshots:
        if base.sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Comparison changed since prepare")
    cached = json.loads((ROOT / "comparison_labels_text_20261022/new_figure_metrics.json").read_text(encoding="utf-8"))
    if cached["roi_xyxy"] != OLD_ROI:
        raise ValueError("Unexpected original crop")
    images = base.read_images("kodim10.png")
    with Image.open(PNG) as old:
        canvas = old.copy()
    if canvas.size != (2248, 1612):
        raise ValueError("Unexpected figure dimensions")
    before_pixels = np.asarray(canvas).copy()
    draw = ImageDraw.Draw(canvas)
    allowed = np.zeros((1612, 2248), dtype=bool)
    crops, metrics = [], {}
    x1, y1, x2, y2 = NEW_ROI
    for column, method in enumerate(base.METHODS):
        left = 40 + column * 552
        canvas.paste(images[method], (left, 150))
        draw.rectangle((left, 150, left + 511, 917), outline="#666666", width=2)
        draw.rectangle((left + x1, 150 + y1, left + x2 - 1, 150 + y2 - 1), outline="#ff2020", width=4)
        crop = images[method].crop(NEW_ROI).resize((448, 448), Image.Resampling.NEAREST)
        canvas.paste(crop, (left + 32, 1026))
        crops.append((left + 32, np.asarray(crop)))
        for roi in (OLD_ROI, NEW_ROI):
            a, b, c, d = roi
            allowed[150 + b:150 + d, left + a:left + c] = True
        allowed[1026:1474, left + 32:left + 480] = True
        if method != "Original":
            values = dict(cached["metrics"][DISPLAY[method]])
            actual_psnr = base.psnr(np.asarray(images["Original"]), np.asarray(images[method]))
            if abs(actual_psnr - values["psnr_db"]) > 1e-10:
                raise ValueError("Cached whole-image metric does not match input files")
            values["roi_psnr_db"] = base.psnr(np.asarray(images["Original"].crop(NEW_ROI)),
                                              np.asarray(images[method].crop(NEW_ROI)))
            metrics[DISPLAY[method]] = values
            print(f"HULL METRIC {DISPLAY[method]}: {json.dumps(values)}", flush=True)
    canvas.save(PNG, dpi=(300, 300))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(2248 / 300, 1612 / 300), dpi=300)
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.imshow(np.asarray(canvas), interpolation="none")
    axes.set_axis_off()
    figure.savefig(PDF, dpi=300, metadata={"Title": "Kodak kodim10 hull lettering: Original | JPEG-2000 | RAQJSCC | MDRQ",
                                          "Creator": "replace_crop.py"})
    with Image.open(PNG) as saved:
        after_pixels = np.asarray(saved)
        if not np.array_equal(before_pixels[~allowed], after_pixels[~allowed]):
            raise AssertionError("Pixels changed outside the old/new red boxes and zoom panels")
        for left, expected in crops:
            if not np.array_equal(np.asarray(saved.crop((left, 1026, left + 448, 1474))), expected):
                raise AssertionError("Zoom panel differs from source crop")
    for entry in json.loads((AUDIT / "other_figures_before.json").read_text(encoding="utf-8")):
        if base.sha256(Path(entry["path"])) != entry["sha256"]:
            raise AssertionError("Another figure changed")
    verify_sources()
    write_json(AUDIT / "verification_and_metrics.json", {
        "image": "kodim10.png", "old_roi_xyxy": OLD_ROI, "new_roi_xyxy": NEW_ROI,
        "request": "User-marked hull lettering, replacing the sail-number detail",
        "display_labels": DISPLAY, "roi_size_wh": [80, 80], "zoom_size_wh": [448, 448],
        "zoom_interpolation": "nearest neighbor, identical across all methods", "figure_size_wh": [2248, 1612],
        "unchanged_pixels": "Every pixel outside the old/new red-box rectangles and four zoom panels",
        "unchanged_whole_image_metrics": True, "metrics": metrics, "verified_zoom_panels": 4,
        "source_images_verified_unchanged": 96, "other_comparison_files_verified_unchanged": 8,
        "outputs": [{"path": str(path), "before_sha256": entry["sha256"], "after_sha256": base.sha256(path),
                     "bytes": path.stat().st_size} for path, entry in zip((PNG, PDF), snapshots)],
    })
    print(f"SAVED {PNG}\nSAVED {PDF}\nVERIFIED unchanged titles and whole-image metrics, four faithful zoom panels.", flush=True)


def review():
    patch_path, status_path, manifest_path = (AUDIT / name for name in ("changes.patch", "git_status_after.txt", "artifacts_manifest.json"))
    for path in (patch_path, status_path, manifest_path):
        path.touch(exist_ok=True)
    status = subprocess.run(["git", "-c", "core.quotepath=false", "status", "--short", "--untracked-files=all"],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout
    status_path.write_text(status, encoding="utf-8")
    new_paths = sorted(path for path in AUDIT.rglob("*") if path.is_file())
    paths = sorted(new_paths + [PNG, PDF])
    owned = {str(path.relative_to(ROOT)) for path in paths}
    new_relative = {str(path.relative_to(ROOT)) for path in new_paths}
    before = (AUDIT / "git_status_before.txt").read_text(encoding="utf-8").splitlines()
    if any(line[3:] in new_relative for line in before):
        raise ValueError("An audit file already existed before this task")
    missing = new_relative - {line[3:] for line in status.splitlines() if line.startswith("?? ")}
    ignored = set()
    if missing:
        result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT, input="\n".join(sorted(missing)) + "\n",
                                capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise ValueError(result.stderr)
        ignored = set(result.stdout.splitlines())
        if missing - ignored:
            raise ValueError(f"Unaccounted files: {missing - ignored}")
    tracked_diff = subprocess.run(["git", "diff", "--", *sorted(owned)], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout
    if tracked_diff:
        raise ValueError("Unexpected tracked changes")
    patch = []
    for path in new_paths:
        if path.suffix not in (".py", ".md", ".json") or path == manifest_path:
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
        "modified_existing_files": [str(PNG), str(PDF)], "new_untracked_files": sorted(new_relative - ignored),
        "new_files_ignored_by_git": sorted(ignored), "outside_scope_status_added": sorted(outside_after - outside_before),
        "outside_scope_status_removed": sorted(outside_before - outside_after),
        "files": [{"path": str(path), "action": "modified" if path in (PNG, PDF) else "new",
                   "bytes": path.stat().st_size, "sha256": base.sha256(path)} for path in paths if path != manifest_path],
        "manifest_self": str(manifest_path),
    })
    print(f"REVIEW two modified existing files, {len(new_paths)} new audit files; "
          f"{len(new_relative - ignored)} untracked, {len(ignored)} ignored logs.", flush=True)
    print(f"OUTSIDE SCOPE status additions={len(outside_after - outside_before)}, removals={len(outside_before - outside_after)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "render", "review"), required=True)
    args = parser.parse_args()
    print(f"RUN python {__file__} --stage {args.stage}", flush=True)
    {"prepare": prepare, "render": render, "review": review}[args.stage]()
