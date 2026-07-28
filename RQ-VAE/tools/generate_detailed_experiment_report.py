#!/usr/bin/env python3
"""Generate a maintainable Excel workbook for the SimVQ experiment ledger."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "experiments" / "reports"
SOURCE_CSV = REPORT_DIR / "SimVQ_experiment_results_SNR0_20260531.csv"
OUTPUT_XLSX = REPORT_DIR / "SimVQ_experiment_results_SNR0_detailed_20260601.xlsx"
OUTPUT_CSV = REPORT_DIR / "SimVQ_experiment_results_SNR0_detailed_20260601.csv"
AUTO_RESULT_DIR = ROOT / "experiments" / "auto_results"


def read_source_rows() -> list[dict[str, str]]:
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest_path = ROOT / "experiments" / "auto_eval_manifest.json"
    if not manifest_path.exists():
        return rows
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {row["实验名称"]: row for row in rows}
    for name, settings in manifest.items():
        if name in by_name or "report_clone" not in settings:
            continue
        clone_name = settings["report_clone"]
        if clone_name not in by_name:
            raise KeyError(f"Unknown report_clone={clone_name!r} for {name!r}")
        row = dict(by_name[clone_name])
        env = settings["env"]
        codebooks = [int(item) for item in env["SIMVQ_NUM_EMBEDDINGS_LIST"].split(",")]
        strides = [int(item) for item in env["SIMVQ_DOWNSAMPLE_STRIDES"].split(",")]
        cumulative = 1
        source_bpp = 0.0
        for stride, codebook in zip(strides, codebooks):
            cumulative *= stride
            source_bpp += math.log2(codebook) / cumulative ** 2
        row.update({
            "实验名称": name,
            "状态": "训练中",
            "Checkpoint": f"checkpoints/{name}/best_vq_deepsc.pth",
            "已记录Epoch": "0",
            "计划Epoch": "200",
            "下采样步幅": str(strides),
            "码本大小": str(codebooks),
            "索引比特数除以HW": str(source_bpp),
            "发送比特数除以HW": str(source_bpp / float(row["LDPC码率"])),
            "压缩率_发送比特数除以3HW": str(source_bpp / float(row["LDPC码率"]) / 3),
            "No_Channel_PSNR_dB": "",
            "No_Channel_MS_SSIM": "",
            "SNR0真实链路_PSNR_dB": "",
            "SNR0真实链路_MS_SSIM": "",
            "最佳验证损失": "",
        })
        row.update(settings.get("report_overrides", {}))
        rows.append(row)
        by_name[name] = row
    return rows


def parse_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.strip("[]").split(",") if item.strip()]


def float_or_blank(value: str):
    return float(value) if value not in ("", None) else ""


def int_or_blank(value: str):
    return int(value) if value not in ("", None) else ""


def latest_epoch_metrics(name: str) -> dict[str, object]:
    path = ROOT / "experiments" / f"{name}_epoch_metrics.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "epoch" not in rows[0]:
        return {}
    latest = max(rows, key=lambda row: int(row["epoch"]))
    return {
        "epoch": int(latest["epoch"]),
        "best_val_recon": float(latest["best_val_recon"]),
    }


def load_json_metrics(path: Path, condition: str) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("results", {}).get(condition, {})


def automatic_results(name: str) -> dict[str, object]:
    no_channel = load_json_metrics(AUTO_RESULT_DIR / f"{name}_no_channel.json", "no_channel")
    snr0 = load_json_metrics(AUTO_RESULT_DIR / f"{name}_snr0.json", "0")
    return {
        "no_channel_psnr": no_channel.get("psnr", ""),
        "no_channel_ms_ssim": no_channel.get("ms_ssim", ""),
        "snr0_psnr": snr0.get("psnr", ""),
        "snr0_ms_ssim": snr0.get("ms_ssim", ""),
        "complete": bool(no_channel and snr0),
    }


def experiment_extras(name: str) -> dict[str, object]:
    extras: dict[str, object] = {
        "实验类别": "正式对比",
        "训练数据集": "Cars196/train_data",
        "验证数据集": "Cars196/val_data",
        "测试数据集": "Kodak（24 张图像）",
        "输入通道数": 3,
        "输出通道数": 3,
        "Commitment Cost": 0.25,
        "生成器初始学习率": "5e-5",
        "Codebook Projection 学习率": "2e-4",
        "Adam Betas": "(0.5, 0.999)",
        "计划总 Batch Size": 24,
        "Micro Batch Size": 24,
        "训练信道类型": "AWGN",
        "训练随机 SNR 范围 dB": "[0, 15]",
        "训练 LDPC 码率": 0.5,
        "验证 LDPC 码率": 0.5,
        "LDPC Block Length": 256,
        "Rician K Factor": 10,
        "Channel Prob 起始 Epoch": 80,
        "Channel Prob 结束 Epoch": 120,
        "预训练初始化": "从零训练",
        "可继续优化的主变量": "",
        "配置来源": "config.py + 原始汇总表",
    }
    if name == "observed_001_baseline":
        extras.update(
            {
                "实验类别": "历史参考",
                "Channel Prob 起始 Epoch": "不适用",
                "Channel Prob 结束 Epoch": "不适用",
                "预训练初始化": "历史运行中发现的 checkpoint",
                "可继续优化的主变量": "仅作为历史基线；不建议继续投入算力",
                "配置来源": "历史 checkpoint + experiments/observed_001_epoch_metrics.csv",
            }
        )
    elif name == "quality_v1_unet2_ds4x2_k64":
        extras.update(
            {
                "实验类别": "高码率质量参考",
                "Channel Prob 起始 Epoch": "不适用",
                "Channel Prob 结束 Epoch": "不适用",
                "可继续优化的主变量": "用于判断低码率模型的质量损失；码率较高，不作为最终方案",
            }
        )
    elif name.startswith("archive_"):
        extras.update(
            {
                "实验类别": "归档消融",
                "预训练初始化": "从零训练",
                "配置来源": "experiments/archive 下对应 CSV + 原始汇总表",
            }
        )
    elif "_LPIPS_" in name:
        extras.update(
            {
                "实验类别": "感知损失消融",
                "预训练初始化": "Stage B k64-256 最佳权重",
                "可继续优化的主变量": "VGG 感知损失权重；损失归一化；是否延后加入感知项",
                "配置来源": "run_exp1_lpips.sh + config.py + 结果 JSON",
            }
        )
    elif "_larger_" in name:
        extras.update(
            {
                "实验类别": "容量扩展消融",
                "计划总 Batch Size": 24,
                "Micro Batch Size": 24,
                "预训练初始化": "Stage B k64-256 最佳权重；仅形状匹配参数可加载",
                "可继续优化的主变量": "Base Channels；残差块数；显存占用；训练是否已收敛",
                "配置来源": "run_exp2_larger.sh + config.py + epoch CSV",
            }
        )
    elif "_SwinEnhance_" in name:
        extras.update(
            {
                "实验类别": "SwinIR 后处理消融",
                "计划总 Batch Size": 8,
                "Micro Batch Size": 8,
                "预训练初始化": "Stage B k64-256 最佳权重；主干参数匹配",
                "可继续优化的主变量": "SwinIR block 数；窗口设置；Batch Size；训练时长",
                "配置来源": "run_exp3_swin.sh + config.py + epoch CSV",
            }
        )
    elif "_DynSwinEnhance_" in name:
        extras.update(
            {
                "实验类别": "加重 SwinIR 与容量联合消融",
                "计划总 Batch Size": 8,
                "Micro Batch Size": 8,
                "预训练初始化": "Stage B k64-256 最佳权重；形状匹配参数可加载",
                "可继续优化的主变量": "SwinIR block 数；Base Channels；单变量拆分实验",
                "配置来源": "run_exp4_dynamic_swin.sh + config.py + epoch CSV",
            }
        )
    elif "_A_curriculum_" in name:
        extras["可继续优化的主变量"] = "码本大小；下采样；课程学习区间"
    elif "_B_backbone_" in name:
        extras["可继续优化的主变量"] = "码本大小；主干容量；残差块数；归一化"
    elif "_C_full_" in name:
        extras["可继续优化的主变量"] = "Attention block 数；对比 B 判断 attention 是否值得保留"
    return extras


def detail_rows(source_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in source_rows:
        name = source["实验名称"]
        latest_metrics = latest_epoch_metrics(name)
        recorded_epoch = latest_metrics.get("epoch", int_or_blank(source["已记录Epoch"]))
        auto = automatic_results(name)
        status = "已完成（自动评测）" if auto["complete"] else source["状态"]
        codebooks = parse_list(source["码本大小"])
        strides = parse_list(source["下采样步幅"])
        dims = parse_list(source["特征维度"])
        cumulative = []
        scale = 1
        for stride in strides:
            scale *= stride
            cumulative.append(scale)
        row: dict[str, object] = {
            "实验名称": name,
            "状态": status,
            "实验阶段": source["实验阶段"],
            "实验类别": "",
            "Checkpoint": source["Checkpoint"],
            "已记录 Epoch": recorded_epoch,
            "计划 Epoch": int_or_blank(source["计划Epoch"]),
            "完成度": (
                int(recorded_epoch) / int(source["计划Epoch"])
                if recorded_epoch != "" and source["计划Epoch"]
                else ""
            ),
            "U-Net 深度": int_or_blank(source["U_Net深度"]),
            "输入通道数": "",
            "输出通道数": "",
            "Base Channels": int_or_blank(source["Base_Channels"]),
            "第 1 层下采样步幅": strides[0] if len(strides) > 0 else "",
            "第 2 层下采样步幅": strides[1] if len(strides) > 1 else "",
            "下采样步幅列表": source["下采样步幅"],
            "下采样实现": source["下采样实现"],
            "总下采样倍率": source["总下采样倍率"],
            "第 1 层累计下采样": cumulative[0] if len(cumulative) > 0 else "",
            "第 2 层累计下采样": cumulative[1] if len(cumulative) > 1 else "",
            "第 1 层特征维度": dims[0] if len(dims) > 0 else "",
            "第 2 层特征维度": dims[1] if len(dims) > 1 else "",
            "特征维度列表": source["特征维度"],
            "第 1 层码本大小 K1": codebooks[0] if len(codebooks) > 0 else "",
            "第 2 层码本大小 K2": codebooks[1] if len(codebooks) > 1 else "",
            "码本大小列表": source["码本大小"],
            "第 1 层索引位数 log2(K1)": math.log2(codebooks[0]) if len(codebooks) > 0 else "",
            "第 2 层索引位数 log2(K2)": math.log2(codebooks[1]) if len(codebooks) > 1 else "",
            "第 1 层 Source BPP": math.log2(codebooks[0]) / cumulative[0] ** 2 if len(codebooks) > 0 else "",
            "第 2 层 Source BPP": math.log2(codebooks[1]) / cumulative[1] ** 2 if len(codebooks) > 1 else "",
            "总 Source BPP": float_or_blank(source["索引比特数除以HW"]),
            "LDPC 码率": float_or_blank(source["LDPC码率"]),
            "调制方式": source["调制方式"],
            "发送 BPP": float_or_blank(source["发送比特数除以HW"]),
            "发送 bits/value": float_or_blank(source["压缩率_发送比特数除以3HW"]),
            "归一化": source["归一化"],
            "GroupNorm Groups": 32 if source["归一化"] == "GroupNorm" else "不适用",
            "激活函数": source["激活函数"],
            "编码器残差块数/采样块": int_or_blank(source["编码器残差块数"]),
            "解码器残差块数/采样块": int_or_blank(source["解码器残差块数"]),
            "上采样方式": source["上采样方式"],
            "Bottleneck Attention": source["Bottleneck_Attention"],
            "SwinIR 后处理": source["SwinIR后处理"],
            "Commitment Cost": "",
            "MSE 损失权重": float_or_blank(source["MSE损失权重"]),
            "MS-SSIM 损失权重": float_or_blank(source["MS_SSIM损失权重"]),
            "VGG 感知损失权重": float_or_blank(source["VGG感知损失权重"]),
            "VQ 损失": source["VQ损失"],
            "VQ 层权重退火": source["VQ层权重退火"],
            "Skip Dropout 退火": source["Skip_Dropout退火"],
            "PHASE1_END / PHASE2_END": source["PHASE1_END与PHASE2_END"],
            "信道课程": source["信道课程"],
            "Channel Prob 起始 Epoch": "",
            "Channel Prob 结束 Epoch": "",
            "训练信道类型": "",
            "训练随机 SNR 范围 dB": "",
            "训练 LDPC 码率": "",
            "验证 LDPC 码率": "",
            "LDPC Block Length": "",
            "Rician K Factor": "",
            "生成器初始学习率": "",
            "Codebook Projection 学习率": "",
            "Adam Betas": "",
            "计划总 Batch Size": "",
            "Micro Batch Size": "",
            "训练数据集": "",
            "验证数据集": "",
            "测试数据集": "",
            "预训练初始化": "",
            "No-channel PSNR dB": auto["no_channel_psnr"] or float_or_blank(source["No_Channel_PSNR_dB"]),
            "No-channel MS-SSIM": auto["no_channel_ms_ssim"] or float_or_blank(source["No_Channel_MS_SSIM"]),
            "SNR=0 dB 真实链路 PSNR dB": auto["snr0_psnr"] or float_or_blank(source["SNR0真实链路_PSNR_dB"]),
            "SNR=0 dB 真实链路 MS-SSIM": auto["snr0_ms_ssim"] or float_or_blank(source["SNR0真实链路_MS_SSIM"]),
            "SNR=0 dB 测试链路": source["SNR0测试链路"],
            "最佳验证损失": latest_metrics.get("best_val_recon", float_or_blank(source["最佳验证损失"])),
            "可继续优化的主变量": "",
            "配置来源": "",
            "备注": source["备注"],
        }
        row.update(experiment_extras(name))
        rows.append(row)
    return rows


def rate_rows(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in details:
        rows.append(
            {
                "实验名称": row["实验名称"],
                "K1": row["第 1 层码本大小 K1"],
                "K2": row["第 2 层码本大小 K2"],
                "第 1 层累计下采样": row["第 1 层累计下采样"],
                "第 2 层累计下采样": row["第 2 层累计下采样"],
                "第 1 层公式": f"log2({row['第 1 层码本大小 K1']}) / {row['第 1 层累计下采样']}^2",
                "第 1 层 Source BPP": row["第 1 层 Source BPP"],
                "第 2 层公式": f"log2({row['第 2 层码本大小 K2']}) / {row['第 2 层累计下采样']}^2",
                "第 2 层 Source BPP": row["第 2 层 Source BPP"],
                "总 Source BPP": row["总 Source BPP"],
                "LDPC 码率": row["LDPC 码率"],
                "发送 BPP = Source BPP / LDPC R": row["发送 BPP"],
                "发送 bits/value = 发送 BPP / 3": row["发送 bits/value"],
            }
        )
    return rows


def completed_ranking(details: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in details:
        psnr = row["SNR=0 dB 真实链路 PSNR dB"]
        if psnr == "":
            continue
        rows.append(
            {
                "排名": 0,
                "实验名称": row["实验名称"],
                "状态": row["状态"],
                "实验阶段": row["实验阶段"],
                "实验类别": row["实验类别"],
                "SNR=0 dB PSNR dB": psnr,
                "SNR=0 dB MS-SSIM": row["SNR=0 dB 真实链路 MS-SSIM"],
                "No-channel PSNR dB": row["No-channel PSNR dB"],
                "No-channel MS-SSIM": row["No-channel MS-SSIM"],
                "总 Source BPP": row["总 Source BPP"],
                "发送 BPP": row["发送 BPP"],
                "发送 bits/value": row["发送 bits/value"],
                "码本": row["码本大小列表"],
                "下采样": row["下采样步幅列表"],
                "主干摘要": (
                    f"{row['归一化']} + {row['激活函数']}；"
                    f"Enc/Dec ResBlocks={row['编码器残差块数/采样块']}/"
                    f"{row['解码器残差块数/采样块']}；"
                    f"Attention={row['Bottleneck Attention']}；"
                    f"SwinIR={row['SwinIR 后处理']}"
                ),
                "Checkpoint": row["Checkpoint"],
            }
        )
    rows.sort(key=lambda item: float(item["SNR=0 dB PSNR dB"]), reverse=True)
    for index, row in enumerate(rows, 1):
        row["排名"] = index
    return rows


def optimization_rows() -> list[dict[str, object]]:
    return [
        {
            "优先级": "P0",
            "对比组": "B k64-256 vs C k64-256",
            "已观察结果": "B: 26.2314 dB / 0.9332；C: 26.1259 dB / 0.9298",
            "结论": "当前 1-block bottleneck attention 没有带来收益。",
            "下一步建议": "优先以 B k64-256 为正式低码率基线；attention 如继续研究，应单独扫 block 数、位置和训练初始化。",
        },
        {
            "优先级": "P0",
            "对比组": "B k16-32 vs B k64-256",
            "已观察结果": "25.1973 -> 26.2314 dB；发送 bits/value 0.0547 -> 0.0833",
            "结论": "扩大码本明显提升质量，但也提高发送码率。",
            "下一步建议": "围绕目标码率建立码本网格，例如 [32,128]、[64,128]、[64,256]，画 PSNR-rate 曲线。",
        },
        {
            "优先级": "P1",
            "对比组": "一步 stride=8 vs 旧级联 stride=2",
            "已观察结果": "A: 24.5661 -> 24.2158 dB；B: 25.2151 -> 25.1973 dB",
            "结论": "A 中级联更好；B 中差距很小，不能简单认定一步卷积更优。",
            "下一步建议": "在 B k64-256 上补做级联下采样单变量实验，避免结构选择被低码本配置误导。",
        },
        {
            "优先级": "P1",
            "对比组": "B k64-256 vs B + VGG 感知损失",
            "已观察结果": "26.2314 -> 24.2727 dB；MS-SSIM 0.9332 -> 0.8934",
            "结论": "当前权重 0.1 的 VGG 感知损失显著伤害指标。",
            "下一步建议": "若关注主观质量，降低权重并延后启用；若目标是 PSNR/MS-SSIM，当前方案应停止。",
        },
        {
            "优先级": "P1",
            "对比组": "B k64-256 vs 扩容 / SwinIR 系列",
            "已观察结果": "扩容、轻量 SwinIR、加重 SwinIR 尚未跑完。",
            "结论": "不要提前根据中途验证损失宣称收益。",
            "下一步建议": "训练完成后统一使用同一 Kodak、同一 LDPC R=1/2、BPSK、AWGN、SNR=0 dB 流程补测。",
        },
    ]


def guide_rows() -> list[dict[str, object]]:
    return [
        {"字段": "总 Source BPP", "含义": "离散索引在信道编码前占用的 bits/pixel。", "用途": "衡量语义编码器本身的源端码率。"},
        {"字段": "发送 BPP", "含义": "总 Source BPP / LDPC 码率。当前 R=0.5，因此发送量翻倍。", "用途": "与真实无线链路占用直接相关。"},
        {"字段": "发送 bits/value", "含义": "发送 BPP / RGB 通道数 3。", "用途": "旧讨论里常写成压缩率，建议以后明确标注口径。"},
        {"字段": "No-channel", "含义": "不经过 LDPC、BPSK、AWGN 的重建指标。", "用途": "判断模型重建上限，不能替代真实链路结果。"},
        {"字段": "SNR=0 dB 真实链路", "含义": "LDPC R=1/2 + BPSK + AWGN，在 0 dB 下的 Kodak 测试结果。", "用途": "本表排名依据。"},
        {"字段": "归档消融", "含义": "训练完成或被中止后保留的旧配置。", "用途": "用于回溯结构决策，不应和当前正式方案混淆。"},
        {"字段": "最佳验证损失", "含义": "各实验内部用于 checkpoint 选择的损失。", "用途": "不同损失定义之间不可直接横向比较，例如 VGG 感知损失实验。"},
    ]


def write_detailed_csv(details: list[dict[str, object]]) -> None:
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)


def column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell_xml(row: int, col: int, value: object, style: int = 0) -> str:
    ref = f"{column_name(col)}{row}"
    if isinstance(value, bool):
        return f'<c r="{ref}" s="{style}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
    text = "" if value is None else str(value)
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t{preserve}>{escape(text)}</t></is></c>'


def sheet_xml(
    headers: list[str],
    rows: list[dict[str, object]],
    *,
    widths: dict[str, float] | None = None,
    title_rows: list[list[object]] | None = None,
) -> str:
    title_rows = title_rows or []
    widths = widths or {}
    total_rows = len(title_rows) + 1 + len(rows)
    total_cols = len(headers)
    cols = []
    for index, header in enumerate(headers, 1):
        width = widths.get(header, min(max(len(header) * 1.7 + 2, 10), 26))
        cols.append(f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>')
    xml_rows = []
    for row_index, title in enumerate(title_rows, 1):
        cells = "".join(cell_xml(row_index, col, value, 3) for col, value in enumerate(title, 1))
        xml_rows.append(f'<row r="{row_index}" ht="28" customHeight="1">{cells}</row>')
    header_index = len(title_rows) + 1
    header_cells = "".join(cell_xml(header_index, col, header, 1) for col, header in enumerate(headers, 1))
    xml_rows.append(f'<row r="{header_index}" ht="32" customHeight="1">{header_cells}</row>')
    for row_index, item in enumerate(rows, header_index + 1):
        cells = "".join(cell_xml(row_index, col, item.get(header, ""), 2) for col, header in enumerate(headers, 1))
        xml_rows.append(f'<row r="{row_index}">{cells}</row>')
    end_ref = f"{column_name(total_cols)}{max(total_rows, 1)}"
    filter_ref = f"A{header_index}:{column_name(total_cols)}{max(total_rows, header_index)}"
    pane = f'<pane ySplit="{header_index}" topLeftCell="A{header_index + 1}" activePane="bottomLeft" state="frozen"/>'
    merges = ""
    if title_rows:
        merge_cells = "".join(
            f'<mergeCell ref="A{index}:{column_name(total_cols)}{index}"/>'
            for index in range(1, len(title_rows) + 1)
        )
        merges = f'<mergeCells count="{len(title_rows)}">{merge_cells}</mergeCells>'
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{end_ref}"/>'
        f'<sheetViews><sheetView workbookViewId="0">{pane}</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="18"/>'
        f'<cols>{"".join(cols)}</cols>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        f'{merges}<autoFilter ref="{filter_ref}"/>'
        '</worksheet>'
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name={quoteattr(name)} sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{sheets}</sheets></workbook>'
    )


def workbook_rels_xml(sheet_count: int) -> str:
    relationships = "".join(
        f'<Relationship Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, sheet_count + 1)
    )
    relationships += (
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{relationships}</Relationships>'
    )


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="4">
    <font><sz val="10"/><name val="Arial"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Arial"/></font>
    <font><sz val="10"/><name val="Arial"/></font>
    <font><b/><color rgb="FF1F4E78"/><sz val="14"/><name val="Arial"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF3F7FB"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"><color rgb="FFD9E2F3"/></left><right style="thin"><color rgb="FFD9E2F3"/></right><top style="thin"><color rgb="FFD9E2F3"/></top><bottom style="thin"><color rgb="FFD9E2F3"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="4">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" xfId="0"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="3" fillId="0" borderId="0" xfId="0"><alignment vertical="center"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def content_types_xml(sheet_count: int) -> str:
    worksheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f'{worksheets}</Types>'
    )


def package_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""


def write_workbook(sheets: list[tuple[str, list[str], list[dict[str, object]], dict[str, float], list[list[object]]]]) -> None:
    sheet_names = [sheet[0] for sheet in sheets]
    with zipfile.ZipFile(OUTPUT_XLSX, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        archive.writestr("_rels/.rels", package_rels_xml())
        archive.writestr("xl/workbook.xml", workbook_xml(sheet_names))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheets)))
        archive.writestr("xl/styles.xml", styles_xml())
        for index, (_, headers, rows, widths, titles) in enumerate(sheets, 1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                sheet_xml(headers, rows, widths=widths, title_rows=titles),
            )


def selected_rows(rows: list[dict[str, object]], headers: list[str]) -> list[list[object]]:
    return [[row.get(header, "") for header in headers] for row in rows]


def pretty_sheets(
    details: list[dict[str, object]],
    ranking: list[dict[str, object]],
    rates: list[dict[str, object]],
    optimization: list[dict[str, object]],
    guide: list[dict[str, object]],
) -> list[dict[str, object]]:
    ranking_headers = [
        "排名", "实验名称", "状态", "实验阶段", "SNR=0 dB PSNR dB", "SNR=0 dB MS-SSIM",
        "No-channel PSNR dB", "No-channel MS-SSIM", "总 Source BPP", "发送 BPP",
        "发送 bits/value", "码本", "下采样", "主干摘要",
    ]
    core_headers = [
        "实验名称", "状态", "实验阶段", "实验类别", "已记录 Epoch", "计划 Epoch", "完成度",
        "总 Source BPP", "发送 BPP", "发送 bits/value", "No-channel PSNR dB", "No-channel MS-SSIM",
        "SNR=0 dB 真实链路 PSNR dB", "SNR=0 dB 真实链路 MS-SSIM", "可继续优化的主变量", "备注",
    ]
    architecture_headers = [
        "实验名称", "实验阶段", "U-Net 深度", "输入通道数", "输出通道数", "Base Channels",
        "第 1 层下采样步幅", "第 2 层下采样步幅", "下采样实现", "总下采样倍率",
        "第 1 层特征维度", "第 2 层特征维度", "第 1 层码本大小 K1", "第 2 层码本大小 K2",
        "归一化", "GroupNorm Groups", "激活函数", "编码器残差块数/采样块", "解码器残差块数/采样块",
        "上采样方式", "Bottleneck Attention", "SwinIR 后处理",
    ]
    training_headers = [
        "实验名称", "状态", "Commitment Cost", "MSE 损失权重", "MS-SSIM 损失权重", "VGG 感知损失权重",
        "VQ 损失", "VQ 层权重退火", "Skip Dropout 退火", "PHASE1_END / PHASE2_END", "信道课程",
        "Channel Prob 起始 Epoch", "Channel Prob 结束 Epoch", "训练信道类型", "训练随机 SNR 范围 dB",
        "训练 LDPC 码率", "验证 LDPC 码率", "LDPC Block Length", "生成器初始学习率",
        "Codebook Projection 学习率", "Adam Betas", "计划总 Batch Size", "Micro Batch Size", "预训练初始化",
    ]
    rate_headers = list(rates[0])
    optimization_headers = list(optimization[0])
    guide_headers = list(guide[0])
    appendix_headers = list(details[0])
    return [
        {
            "name": "01_SNR0排名",
            "title": "SimVQ 实验结果排名",
            "subtitle": "Kodak 24 张图像 | LDPC R=1/2 + BPSK + AWGN | SNR=0 dB | 仅列出已有真实链路结果的方案",
            "headers": ranking_headers,
            "rows": selected_rows(ranking, ranking_headers),
            "widths": {"实验名称": 55, "状态": 24, "主干摘要": 75, "码本": 16, "下采样": 14},
            "tab_color": 0x1F4E78,
        },
        {
            "name": "02_核心总览",
            "title": "全部实验核心总览",
            "subtitle": "默认从本页开始浏览：历史、归档、已完成和训练中实验均保留；黄色状态表示仍在训练。",
            "headers": core_headers,
            "rows": selected_rows(details, core_headers),
            "widths": {"实验名称": 55, "状态": 25, "实验类别": 23, "可继续优化的主变量": 62, "备注": 75},
            "tab_color": 0x5B9BD5,
        },
        {
            "name": "03_网络结构",
            "title": "网络结构配置",
            "subtitle": "将下采样、特征维度、码本、归一化、残差块和增强模块拆开，便于设计单变量消融。",
            "headers": architecture_headers,
            "rows": selected_rows(details, architecture_headers),
            "widths": {"实验名称": 55, "下采样实现": 22, "Bottleneck Attention": 22, "SwinIR 后处理": 22},
            "tab_color": 0x70AD47,
        },
        {
            "name": "04_训练与信道",
            "title": "训练、损失与信道配置",
            "subtitle": "训练调度和信道配置单独成表，避免和网络结构混在一起。",
            "headers": training_headers,
            "rows": selected_rows(details, training_headers),
            "widths": {"实验名称": 55, "信道课程": 72, "预训练初始化": 62, "VQ 层权重退火": 22, "Skip Dropout 退火": 22},
            "tab_color": 0xED7D31,
        },
        {
            "name": "05_逐层码率",
            "title": "逐层码率计算",
            "subtitle": "Source BPP 是信道编码前的索引码率；发送 BPP 已计入 LDPC R=1/2 的冗余。",
            "headers": rate_headers,
            "rows": selected_rows(rates, rate_headers),
            "widths": {"实验名称": 55, "第 1 层公式": 24, "第 2 层公式": 24},
            "tab_color": 0xA5A5A5,
        },
        {
            "name": "06_优化建议",
            "title": "下一轮优化建议",
            "subtitle": "按当前完成实验得出的优先级。建议继续保持单变量对照，避免把多个改动混为一个结论。",
            "headers": optimization_headers,
            "rows": selected_rows(optimization, optimization_headers),
            "widths": {"对比组": 34, "已观察结果": 58, "结论": 66, "下一步建议": 80},
            "tab_color": 0xFFC000,
        },
        {
            "name": "07_字段说明",
            "title": "字段说明与阅读口径",
            "subtitle": "重点区分源端码率、真实发送码率、no-channel 上限和真实链路指标。",
            "headers": guide_headers,
            "rows": selected_rows(guide, guide_headers),
            "widths": {"字段": 28, "含义": 78, "用途": 68},
            "tab_color": 0x8064A2,
        },
        {
            "name": "08_完整字段附录",
            "title": "完整字段附录",
            "subtitle": "用于追溯和程序化筛选。日常浏览建议使用前面的主题工作表。",
            "headers": appendix_headers,
            "rows": selected_rows(details, appendix_headers),
            "widths": {"实验名称": 55, "状态": 25, "Checkpoint": 75, "备注": 70},
            "tab_color": 0x7F7F7F,
        },
    ]


def write_pretty_workbook(sheets: list[dict[str, object]]) -> None:
    helper = ROOT / "tools" / "write_pretty_experiment_report_uno.py"
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump({"sheets": sheets}, handle, ensure_ascii=False)
        payload_path = Path(handle.name)
    try:
        subprocess.run(
            ["/usr/bin/python3", str(helper), str(payload_path), str(OUTPUT_XLSX)],
            check=True,
        )
    finally:
        payload_path.unlink(missing_ok=True)


def main() -> None:
    source = read_source_rows()
    details = detail_rows(source)
    ranking = completed_ranking(details)
    rates = rate_rows(details)
    optimization = optimization_rows()
    guide = guide_rows()
    write_detailed_csv(details)
    write_pretty_workbook(pretty_sheets(details, ranking, rates, optimization, guide))
    print(f"Wrote {OUTPUT_XLSX.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"Experiments: {len(details)}; ranked SNR=0 dB results: {len(ranking)}")


if __name__ == "__main__":
    main()
