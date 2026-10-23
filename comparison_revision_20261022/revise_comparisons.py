#!/usr/bin/env python3
"""Inspect arm and lettering crops, then assemble source-faithful revised comparisons."""

import argparse
import difflib
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
PREVIOUS = ROOT / "comparison_review_20261022/build_comparisons.py"
spec = importlib.util.spec_from_file_location("previous_comparison_helpers", PREVIOUS)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
METHODS = previous.METHODS
CANDIDATES = [
    ("kodim05.png", "arm", [352, 64, 480, 192]),
    ("kodim05.png", "sleeve lettering", [360, 88, 456, 184]),
    ("kodim08.png", "left sign", [544, 296, 672, 424]),
    ("kodim08.png", "right sign", [632, 280, 760, 408]),
    ("kodim14.png", "raft lettering", [504, 368, 648, 448]),
    ("kodim14.png", "raft square", [504, 320, 648, 464]),
    ("kodim20.png", "aircraft lettering", [224, 232, 368, 376]),
    ("kodim03.png", "hat lettering", [104, 112, 248, 256]),
    ("kodim06.png", "boat name square", [264, 304, 408, 448]),
    ("kodim06.png", "boat name tight", [280, 332, 408, 396]),
    ("kodim14.png", "selected raft text square", [508, 352, 636, 480]),
]
SELECTIONS = {
    "kodim05.png": {
        "roi_xyxy": [352, 64, 480, 192],
        "reason": "User-marked rider's arm, blue-white sleeve lettering, shoulder and glove.",
    },
    "kodim14.png": {
        "roi_xyxy": [508, 352, 636, 480],
        "reason": "User-selected raft lettering: SNOWMASS / WHITEWATER, emphasizing the fidelity of character edges.",
    },
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_snapshot(path):
    records = json.loads(path.read_text(encoding="utf-8"))
    changed = [record["path"] for record in records if previous.sha256(Path(record["path"])) != record["sha256"]]
    if changed:
        raise ValueError(f"Changed input or prior artifact: {changed}")
    return len(records)


def inspect():
    for name in ("source_manifest.json", "previous_artifacts_manifest.json"):
        if (AUDIT / name).exists():
            verify_snapshot(AUDIT / name)
    sources = []
    inventories = {method: sorted(p.name for p in folder.glob("*.png"))
                   for method, folder in previous.DIRECTORIES.items()}
    if len(inventories["Original"]) != 24 or any(names != inventories["Original"] for names in inventories.values()):
        raise ValueError("Expected 24 matching source images in all four directories")
    for method, directory in previous.DIRECTORIES.items():
        for name in inventories[method]:
            path = directory / name
            sources.append({"path": str(path), "sha256": previous.sha256(path)})
    write_json(AUDIT / "source_manifest.json", sources)
    write_json(AUDIT / "previous_artifacts_manifest.json", [
        {"path": str(path), "sha256": previous.sha256(path)}
        for path in sorted((ROOT / "comparison_review_20261022").rglob("*")) if path.is_file()
    ] + [{"path": str(path), "sha256": previous.sha256(path)}
         for path in sorted(ROOT.glob("kodim*_comparison_20261022.*"))])
    records = []
    for page_index in range(math.ceil(len(CANDIDATES) / 4)):
        page_candidates = CANDIDATES[page_index * 4:(page_index + 1) * 4]
        canvas = Image.new("RGB", (1600, 400 * len(page_candidates)), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, (name, label, roi) in enumerate(page_candidates):
            images = previous.read_images(name)
            metrics = {}
            for column, method in enumerate(METHODS):
                crop = images[method].crop(roi)
                scale = min(384 / crop.width, 332 / crop.height)
                scaled = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.NEAREST)
                left = column * 400 + (400 - scaled.width) // 2
                top = row_index * 400 + 40 + (332 - scaled.height) // 2
                previous.centered(draw, column * 400 + 200, row_index * 400 + 22, f"{name[:7]} | {method}", 22)
                canvas.paste(scaled, (left, top))
                if method != "Original":
                    metrics[method] = previous.psnr(np.asarray(images["Original"].crop(roi)), np.asarray(crop))
            previous.centered(draw, 800, row_index * 400 + 385, f"{label} {roi}", 19)
            records.append({"image": name, "label": label, "roi_xyxy": roi, "roi_psnr_db": metrics,
                            "minimum_MDRQ_advantage_db": metrics["MDRQ"] - max(metrics["JPEG"], metrics["RAQ-VAE"])})
        path = AUDIT / f"candidate_details_{page_index + 1}.png"
        canvas.save(path)
        print(f"PREVIEW {path}", flush=True)
    write_json(AUDIT / "candidate_regions.json", records)
    for row in records:
        print(json.dumps(row), flush=True)


def render():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    source_count = verify_snapshot(AUDIT / "source_manifest.json")
    prior_count = verify_snapshot(AUDIT / "previous_artifacts_manifest.json")
    ms_ssim, metric_source = previous.metric_function()
    selections, outputs = [], []
    verified_panels = 0
    for name, selection in SELECTIONS.items():
        images = previous.read_images(name)
        arrays = {method: np.asarray(image) for method, image in images.items()}
        if any(image.size != (768, 512) for image in images.values()):
            raise ValueError("Expected native 768 x 512 RGB images")
        roi = selection["roi_xyxy"]
        x1, y1, x2, y2 = roi
        if x2 - x1 != 128 or y2 - y1 != 128:
            raise ValueError("Both revisions use identical 128 x 128 crop dimensions")
        metrics = {}
        for method in METHODS[1:]:
            score = float(ms_ssim(arrays["Original"] / 255.0, arrays[method] / 255.0))
            if not 0 < score < 1:
                raise ValueError(f"Invalid MS-SSIM: {score}")
            metrics[method] = {
                "psnr_db": previous.psnr(arrays["Original"], arrays[method]),
                "ms_ssim": score,
                "ms_ssim_db": -10 * math.log10(1 - score),
                "roi_psnr_db": previous.psnr(arrays["Original"][y1:y2, x1:x2], arrays[method][y1:y2, x1:x2]),
            }
            print(f"METRIC {name} {method}: {json.dumps(metrics[method])}", flush=True)
        for key in ("psnr_db", "ms_ssim", "roi_psnr_db"):
            if any(metrics["MDRQ"][key] <= metrics[method][key] for method in ("JPEG", "RAQ-VAE")):
                raise ValueError(f"MDRQ does not lead the selected example on {key}")
        canvas = Image.new("RGB", (3272, 1580), "white")
        draw = ImageDraw.Draw(canvas)
        crop_checks = []
        for column, method in enumerate(METHODS):
            left, top = 40 + column * 808, 150
            previous.centered(draw, left + 384, 103, method, 44)
            canvas.paste(images[method], (left, top))
            draw.rectangle((left, top, left + 767, top + 511), outline="#666666", width=2)
            draw.rectangle((left + x1, top + y1, left + x2 - 1, top + y2 - 1), outline="#ff2020", width=4)
            crop = images[method].crop(roi).resize((672, 672), Image.Resampling.NEAREST)
            canvas.paste(crop, (left + 48, 770))
            crop_checks.append((left + 48, np.asarray(crop)))
            if method != "Original":
                values = metrics[method]
                text = f"PSNR={values['psnr_db']:.2f} dB | MS-SSIM={values['ms_ssim_db']:.2f} dB"
                previous.centered(draw, left + 384, 1521, text, 31)
        png = ROOT / f"{Path(name).stem}_comparison_20261022_v2.png"
        pdf = png.with_suffix(".pdf")
        canvas.save(png, dpi=(300, 300))
        figure = Figure(figsize=(3272 / 300, 1580 / 300), dpi=300)
        FigureCanvasAgg(figure)
        axes = figure.add_axes((0, 0, 1, 1))
        axes.imshow(np.asarray(canvas), interpolation="none")
        axes.set_axis_off()
        figure.savefig(pdf, dpi=300, metadata={"Title": f"Revised {name}: arm or lettering detail",
                                              "Creator": "revise_comparisons.py"})
        with Image.open(png) as saved:
            assert saved.size == (3272, 1580)
            for left, expected in crop_checks:
                actual = np.asarray(saved.crop((left, 770, left + 672, 1442)))
                if not np.array_equal(expected, actual):
                    raise AssertionError("Zoom panel changed source-derived pixels")
                verified_panels += 1
            for column, method in enumerate(METHODS):
                left = 40 + column * 808
                actual = np.asarray(saved.crop((left, 150, left + 768, 662)))
                mask = np.ones((512, 768), dtype=bool)
                mask[:2] = mask[-2:] = False
                mask[:, :2] = mask[:, -2:] = False
                mask[y1:y2, x1:x2] = False
                if not np.array_equal(actual[mask], arrays[method][mask]):
                    raise AssertionError("Full-image pixels changed outside annotations")
                verified_panels += 1
        for path in (png, pdf):
            outputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": previous.sha256(path)})
        selections.append({"image": name, **selection, "metrics": metrics,
                           "figure_png": str(png), "figure_pdf": str(pdf)})
        print(f"SAVED {png}\nSAVED {pdf}", flush=True)
    verify_snapshot(AUDIT / "source_manifest.json")
    verify_snapshot(AUDIT / "previous_artifacts_manifest.json")
    write_json(AUDIT / "selection_and_metrics.json", {
        "revision": "v2, 2026-10-22; arm and user-selected raft lettering",
        "metric_source": str(metric_source), "metric_source_sha256": previous.sha256(metric_source),
        "helper_source": str(PREVIOUS), "helper_source_sha256": previous.sha256(PREVIOUS),
        "psnr_definition": "RGB full-image MSE; 10*log10(255^2/MSE)",
        "ms_ssim_definition": "Existing Ours-RAQ/utils/metrics.py::ms_ssim on RGB in [0,1], five scales",
        "ms_ssim_db_definition": "-10*log10(1-MS-SSIM); untransformed score also retained",
        "interpretation": "Selected illustrative examples; lettering comparison concerns edge and shape fidelity, not an OCR accuracy claim",
        "selections": selections,
    })
    write_json(AUDIT / "verification.json", {
        "source_images_verified_unchanged": source_count,
        "previous_artifacts_verified_unchanged": prior_count,
        "panels_verified_against_source_pixels": verified_panels,
        "MDRQ_leads_selected_full_image_PSNR_MS_SSIM_and_ROI_PSNR": True,
        "crop_size_wh": [128, 128], "zoom_size_wh": [672, 672],
        "zoom_interpolation": "nearest neighbor, identical across all four methods",
        "output_png_size_wh": [3272, 1580], "outputs": outputs,
    })
    print(f"VERIFIED {source_count} source images, {prior_count} prior artifacts unchanged; {verified_panels} panels.", flush=True)


def review():
    patch_path, status_path, manifest_path = (AUDIT / name for name in ("changes.patch", "git_status_after.txt", "artifacts_manifest.json"))
    for path in (patch_path, status_path, manifest_path):
        path.touch(exist_ok=True)
    status = subprocess.run(["git", "-c", "core.quotepath=false", "status", "--short", "--untracked-files=all"],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout
    status_path.write_text(status, encoding="utf-8")
    outputs = [ROOT / f"{Path(name).stem}_comparison_20261022_v2{extension}"
               for name in SELECTIONS for extension in (".png", ".pdf")]
    paths = sorted([path for path in AUDIT.rglob("*") if path.is_file()] + outputs)
    owned = {str(path.relative_to(ROOT)) for path in paths}
    before = (AUDIT / "git_status_before.txt").read_text(encoding="utf-8").splitlines()
    if any(line[3:] in owned for line in before):
        raise ValueError("A revision output was present before this task")
    missing = owned - {line[3:] for line in status.splitlines() if line.startswith("?? ")}
    ignored = set()
    if missing:
        result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
                                input="\n".join(sorted(missing)) + "\n", capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise ValueError(result.stderr)
        ignored = set(result.stdout.splitlines())
        if missing - ignored:
            raise ValueError(f"Missing new files from Git status: {missing - ignored}")
    tracked_diff = subprocess.run(["git", "diff", "--", *sorted(owned)], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout
    if tracked_diff:
        raise ValueError("Unexpected tracked changes in revision outputs")
    patch = []
    for path in paths:
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
        "scope": "New v2 outputs only; all previous figures and review records remain unchanged",
        "new_untracked_outputs": sorted(owned - ignored), "new_outputs_ignored_by_git": sorted(ignored),
        "outside_scope_status_added": sorted(outside_after - outside_before),
        "outside_scope_status_removed": sorted(outside_before - outside_after),
        "patch_covers": "Full new .py/.md/.json contents except this manifest; logs and images are separately linked",
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": previous.sha256(path)}
                  for path in paths if path != manifest_path], "manifest_self": str(manifest_path),
    })
    print(f"REVIEW {len(paths)} new files: {len(owned - ignored)} untracked, {len(ignored)} ignored logs.", flush=True)
    print(f"OUTSIDE SCOPE status additions={len(outside_after - outside_before)}, removals={len(outside_before - outside_after)}", flush=True)
    print(f"PATCH {patch_path}\nMANIFEST {manifest_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "render", "review"), default="inspect")
    args = parser.parse_args()
    print(f"RUN python {__file__} --stage {args.stage}", flush=True)
    {"inspect": inspect, "render": render, "review": review}[args.stage]()
