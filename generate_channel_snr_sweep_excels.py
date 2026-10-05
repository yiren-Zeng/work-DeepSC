#!/usr/bin/python3
"""Build the two requested Monte Carlo workbooks from saved evaluation JSON."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.append("/usr/lib/python3/dist-packages")
import uno  # type: ignore  # noqa: E402


WORKSPACE = Path("/workspace/yi/work")
SNRS = [-10, -8, -6, -4, -2, 0]
SEEDS = list(range(42, 52))

RAQ_RVQ_DIR = WORKSPACE / "RAQ-RVQ/experiments/eval/snr_sweep_awgn_rician_k10_seeds42_51_20261004"
VQ_DIR = WORKSPACE / "VQ-DeepSC/experiments/eval/snr_sweep_awgn_rician_k10_seeds42_51_20261004"
RAQ_VAE_DIR = WORKSPACE / "RAQ-VAE-main/experiments/eval/snr_sweep_awgn_rician_k10_seeds42_51_20261004"

SYSTEM_CONFIGS = {
    "RAQ-RVQ": [
        {
            "rate": "1/2", "modulation": "QPSK", "codebook": "[2,4], [8,2]",
            "cbr": "1/48", "json": RAQ_RVQ_DIR / "raq_rvq_1over48_qpsk_k2-4_8-2.json",
        },
        {
            "rate": "7/16", "modulation": "16QAM", "codebook": "[2,32], [8,2]",
            "cbr": "1/48", "json": RAQ_RVQ_DIR / "raq_rvq_1over48_16qam_k2-32_8-2.json",
        },
        {
            "rate": "1/2", "modulation": "QPSK", "codebook": "[2,2], [8,2]",
            "cbr": "1/64", "json": RAQ_RVQ_DIR / "raq_rvq_1over64_qpsk_k2-2_8-2.json",
        },
        {
            "rate": "7/16", "modulation": "16QAM", "codebook": "[8,2], [2,16]",
            "cbr": "1/64", "json": RAQ_RVQ_DIR / "raq_rvq_1over64_16qam_k8-2_2-16.json",
        },
    ],
    "VQ-DeepSC": [
        {
            "rate": "1/2", "modulation": "QPSK", "codebook": "[8,16]",
            "cbr": "1/48", "json": VQ_DIR / "vq_1over48_qpsk_k8_16.json",
        },
        {
            "rate": "7/16", "modulation": "16QAM", "codebook": "[64,16]",
            "cbr": "1/48", "json": VQ_DIR / "vq_1over48_16qam_k64_16.json",
        },
        {
            "rate": "1/2", "modulation": "QPSK", "codebook": "[4,16]",
            "cbr": "1/64", "json": VQ_DIR / "vq_1over64_qpsk_k4_16.json",
        },
        {
            "rate": "7/16", "modulation": "16QAM", "codebook": "[16,32]",
            "cbr": "1/64", "json": VQ_DIR / "vq_1over64_16qam_k16_32.json",
        },
    ],
    "RAQ-VAE": [
        {
            "rate": "1/2", "modulation": "QPSK", "codebook": "[8,16]",
            "cbr": "1/48", "label": "raq_vae_1over48_qpsk_k8_16",
        },
        {
            "rate": "7/16", "modulation": "16QAM", "codebook": "[64,16]",
            "cbr": "1/48", "label": "raq_vae_1over48_16qam_k64_16",
        },
        {
            "rate": "1/2", "modulation": "QPSK", "codebook": "[4,16]",
            "cbr": "1/64", "label": "raq_vae_1over64_qpsk_k4_16",
        },
        {
            "rate": "7/16", "modulation": "16QAM", "codebook": "[16,32]",
            "cbr": "1/64", "label": "raq_vae_1over64_16qam_k16_32",
        },
    ],
}


def _snr_key(snr: int) -> str:
    return f"{float(snr):.1f}"


def _expected_ratio(cbr: str) -> float:
    return 1.0 / int(cbr.split("/")[1])


def _validate_summary(summary: dict, cbr: str) -> None:
    if summary["num_trials"] != 10 or summary["channel_seeds"] != SEEDS:
        raise ValueError(f"Monte Carlo seed validation failed: {summary}")
    if not math.isclose(
        float(summary["transmission_ratio"]), _expected_ratio(cbr), rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(
            f"CBR mismatch: got {summary['transmission_ratio']}, expected {cbr}"
        )
    expected_db = -10.0 * math.log10(1.0 - float(summary["ms_ssim_mean"]))
    if not math.isclose(
        float(summary["ms_ssim_db_from_mean"]), expected_db, rel_tol=0, abs_tol=1e-10
    ):
        raise ValueError("MS-SSIM dB was not calculated from the mean raw MS-SSIM")


def load_rows(channel: str) -> dict[str, list[list[object]]]:
    rows_by_system: dict[str, list[list[object]]] = {}
    for system, configs in SYSTEM_CONFIGS.items():
        rows: list[list[object]] = []
        for config_index, config in enumerate(configs, start=1):
            if system != "RAQ-VAE":
                report = json.loads(config["json"].read_text(encoding="utf-8"))
                if report["channel_seeds"] != SEEDS:
                    raise ValueError(f"Unexpected seeds in {config['json']}")
                if report["channel_types"] != ["awgn", "rician"]:
                    raise ValueError(f"Unexpected channel list in {config['json']}")
                if float(report["rician_k_factor"]) != 10.0:
                    raise ValueError(f"Unexpected Rician K in {config['json']}")
                per_channel = report["results"][channel]
            for snr in SNRS:
                if system == "RAQ-VAE":
                    snr_label = str(snr).replace("-", "m")
                    channel_dir = "awgn" if channel == "awgn" else "rician_k10"
                    path = (
                        RAQ_VAE_DIR
                        / f"{config['label']}_snr_{snr_label}"
                        / channel_dir
                        / "monte_carlo_results.json"
                    )
                    aggregate = json.loads(path.read_text(encoding="utf-8"))
                    signature = aggregate["configuration"]
                    if signature["channel"] != channel or float(signature["snr_db"]) != snr:
                        raise ValueError(f"Unexpected RAQ-VAE configuration in {path}")
                    if float(signature["rician_k_factor"]) != 10.0:
                        raise ValueError(f"Unexpected Rician K in {path}")
                    summary = aggregate["monte_carlo"]
                else:
                    metric = per_channel[_snr_key(snr)]
                    summary = metric["monte_carlo"]
                _validate_summary(summary, config["cbr"])
                rows.append(
                    [
                        config_index,
                        config["rate"],
                        "LDPC",
                        config["modulation"],
                        config["codebook"],
                        config["cbr"],
                        snr,
                        float(summary["psnr_mean"]),
                        float(summary["ms_ssim_mean"]),
                        float(summary["ms_ssim_db_from_mean"]),
                        float(summary.get("ber", summary.get("source_ber", 0.0))),
                        summary["num_trials"],
                    ]
                )
        rows_by_system[system] = rows
    return rows_by_system


def _property(name: str, value: object):
    prop = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
    prop.Name = name
    prop.Value = value
    return prop


def _connect_office(port: int, profile: Path):
    cmd = [
        shutil.which("libreoffice") or "libreoffice",
        "--headless",
        "--nologo",
        "--nodefault",
        "--nofirststartwizard",
        f"-env:UserInstallation={uno.systemPathToFileUrl(str(profile))}",
        f"--accept=socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )
    for _ in range(100):
        try:
            ctx = resolver.resolve(
                f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext"
            )
            desktop = ctx.ServiceManager.createInstanceWithContext(
                "com.sun.star.frame.Desktop", ctx
            )
            return process, desktop
        except Exception:
            if process.poll() is not None:
                raise RuntimeError("LibreOffice exited before accepting a connection")
            time.sleep(0.1)
    process.terminate()
    raise TimeoutError("Timed out connecting to LibreOffice")


def _set_cell(sheet, col: int, row: int, value: object) -> None:
    cell = sheet.getCellByPosition(col, row)
    if isinstance(value, bool):
        cell.setValue(float(value))
    elif isinstance(value, (int, float)):
        cell.setValue(float(value))
    else:
        cell.setString(str(value))


def _populate_info(sheet, channel: str) -> None:
    channel_label = "AWGN" if channel == "awgn" else "Rician (K=10)"
    info = [
        [f"{channel_label}：三模型低信噪比蒙特卡洛测试", ""],
        ["信道", channel_label],
        ["SNR (dB)", "-10, -8, -6, -4, -2, 0"],
        ["信道随机种子", "42–51（共10次，逐次重新采样）"],
        ["PSNR汇总", "10次试验的算术平均"],
        ["MS-SSIM汇总", "先平均10次原始MS-SSIM，再计算 -10log10(1-平均值)"],
        ["Rician接收端", "逐符号完美CSI" if channel == "rician" else "不适用"],
        ["数据集", "Kodak-256-transform-resize，24张256×256 RGB图像"],
        ["工作表", "RAQ-RVQ、VQ-DeepSC、RAQ-VAE"],
        ["结果追溯", "各项目 experiments/eval/snr_sweep_awgn_rician_k10_seeds42_51_20261004"],
    ]
    for r, values in enumerate(info):
        for c, value in enumerate(values):
            _set_cell(sheet, c, r, value)
    sheet.getCellRangeByPosition(0, 0, 1, 0).merge(True)
    title = sheet.getCellByPosition(0, 0)
    title.CharWeight = 150.0
    title.CharHeight = 16.0
    title.CellBackColor = 0x1F4E78
    title.CharColor = 0xFFFFFF
    sheet.getCellRangeByPosition(0, 1, 0, len(info) - 1).CharWeight = 150.0
    sheet.getColumns().getByIndex(0).Width = 4200
    sheet.getColumns().getByIndex(1).Width = 16000


def _number_format_key(document, pattern: str) -> int:
    locale = uno.createUnoStruct("com.sun.star.lang.Locale")
    locale.Language = "en"
    locale.Country = "US"
    formats = document.getNumberFormats()
    key = formats.queryKey(pattern, locale, True)
    return key if key != -1 else formats.addNew(pattern, locale)


def _populate_results(document, sheet, system: str, rows: list[list[object]], channel: str) -> None:
    channel_label = "AWGN" if channel == "awgn" else "Rician (K=10)"
    headers = [
        "配置序号", "编码率", "信道编码", "调制方式", "码本", "CBR", "SNR(dB)",
        "PSNR均值(dB)", "原始MS-SSIM均值", "MS-SSIM(dB)", "BER", "试验次数",
    ]
    sheet.getCellRangeByPosition(0, 0, len(headers) - 1, 0).merge(True)
    title = sheet.getCellByPosition(0, 0)
    title.setString(f"{system} — {channel_label} — 10个随机种子均值")
    title.CharWeight = 150.0
    title.CharHeight = 15.0
    title.CellBackColor = 0x1F4E78
    title.CharColor = 0xFFFFFF
    for c, header in enumerate(headers):
        _set_cell(sheet, c, 1, header)
    header_range = sheet.getCellRangeByPosition(0, 1, len(headers) - 1, 1)
    header_range.CharWeight = 150.0
    header_range.CellBackColor = 0xD9EAF7
    palette = [0xF7FBFF, 0xFFF8E7, 0xF2F9F1, 0xFCEFF3]
    for r, values in enumerate(rows, start=2):
        for c, value in enumerate(values):
            _set_cell(sheet, c, r, value)
        sheet.getCellRangeByPosition(0, r, len(headers) - 1, r).CellBackColor = palette[int(values[0]) - 1]
    last_row = len(rows) + 1
    sheet.getCellRangeByPosition(7, 2, 7, last_row).NumberFormat = _number_format_key(document, "0.0000")
    sheet.getCellRangeByPosition(8, 2, 8, last_row).NumberFormat = _number_format_key(document, "0.000000")
    sheet.getCellRangeByPosition(9, 2, 9, last_row).NumberFormat = _number_format_key(document, "0.0000")
    sheet.getCellRangeByPosition(10, 2, 10, last_row).NumberFormat = _number_format_key(document, "0.00000000")
    widths = [2200, 2200, 2800, 2800, 5200, 2200, 2600, 3900, 4400, 3900, 3600, 2600]
    for index, width in enumerate(widths):
        sheet.getColumns().getByIndex(index).Width = width
    sheet.getCellRangeByPosition(0, 0, len(headers) - 1, last_row).VertJustify = 2


def create_workbook(desktop, channel: str, rows_by_system: dict[str, list[list[object]]], output: Path) -> None:
    document = desktop.loadComponentFromURL("private:factory/scalc", "_blank", 0, ())
    sheets = document.getSheets()
    default_name = sheets.getElementNames()[0]
    sheets.getByName(default_name).setName("说明")
    _populate_info(sheets.getByName("说明"), channel)
    for index, system in enumerate(("RAQ-RVQ", "VQ-DeepSC", "RAQ-VAE"), start=1):
        sheets.insertNewByName(system, index)
        _populate_results(document, sheets.getByName(system), system, rows_by_system[system], channel)
    output.unlink(missing_ok=True)
    document.storeAsURL(
        uno.systemPathToFileUrl(str(output)),
        (
            _property("FilterName", "Calc MS Excel 2007 XML"),
            _property("Overwrite", True),
        ),
    )
    document.close(True)


def main() -> None:
    all_rows = {channel: load_rows(channel) for channel in ("awgn", "rician")}
    profile = Path(tempfile.mkdtemp(prefix="channel_sweep_libreoffice_"))
    process = None
    try:
        process, desktop = _connect_office(20847, profile)
        create_workbook(
            desktop,
            "awgn",
            all_rows["awgn"],
            WORKSPACE / "AWGN_三模型_SNR_-10到0_10种子均值_20261004.xlsx",
        )
        create_workbook(
            desktop,
            "rician",
            all_rows["rician"],
            WORKSPACE / "Rician_K10_三模型_SNR_-10到0_10种子均值_20261004.xlsx",
        )
        desktop.terminate()
    finally:
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
