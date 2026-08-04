import numpy as np
import torch
from communications.channel import awgn_channel
from communications.modulation import (
    bpsk_demodulate,
    bpsk_llr,
    bpsk_modulate,
    qpsk_llr,
    qpsk_modulate,
    qam16_llr,
    qam16_modulate,
)
from utils.bit_utils import (
    bits_to_index_tensor,
    bits_to_indices,
    index_tensor_to_bits,
    indices_to_bits,
)
from utils.metrics import calculate_ms_ssim
from utils.reproducibility import _reset_eval_seed


_MODULATION_BITS = {"bpsk": 1, "qpsk": 2, "16qam": 4}


def _image_quality(real_image, reconstructed_images):
    if real_image.device != reconstructed_images.device:
        real_image = real_image.to(reconstructed_images.device, non_blocking=True)
    img1 = (real_image + 1) / 2
    img2 = (reconstructed_images + 1) / 2
    ms_ssim = calculate_ms_ssim(img1, img2)
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 100.0 if mse == 0 else 10 * torch.log10(1.0 / mse).item()
    return ms_ssim, psnr


def _rvq_layout(out):
    """Validate and return the explicit scale -> stage RVQ transmission layout."""
    enabled = bool(out.get("test_raq_rvq_enabled", False))
    if not enabled:
        return False, None

    indices = out.get("indices")
    k_lists = out.get("rvq_k_lists")
    if not isinstance(indices, (list, tuple)) or not isinstance(k_lists, (list, tuple)):
        raise ValueError(
            "test_raq_rvq_enabled=True requires nested 'indices' and 'rvq_k_lists'."
        )
    if len(indices) != len(k_lists):
        raise ValueError(
            "RAQ-RVQ scale count mismatch between 'indices' and 'rvq_k_lists'."
        )
    for scale_idx, (stage_indices, stage_ks) in enumerate(zip(indices, k_lists)):
        if not isinstance(stage_indices, (list, tuple)) or not isinstance(
            stage_ks, (list, tuple)
        ):
            raise ValueError(
                f"RAQ-RVQ scale {scale_idx} must contain stage lists."
            )
        if not stage_indices or len(stage_indices) != len(stage_ks):
            raise ValueError(
                f"RAQ-RVQ scale {scale_idx} has inconsistent indices/stage-K counts."
            )
    return True, [[int(k) for k in ks] for ks in k_lists]


def _source_sizes(real_image):
    batch = int(real_image.shape[0]) if real_image.ndim else 1
    if real_image.ndim >= 3:
        pixels = batch * int(real_image.shape[-2]) * int(real_image.shape[-1])
    else:
        pixels = int(real_image.numel())
    return batch, pixels, int(real_image.numel())


def _new_stage_record(scale, stage, num_embeddings):
    return {
        "scale": scale,
        "stage": stage,
        "num_embeddings": int(num_embeddings),
        "bits_per_index": int(np.log2(num_embeddings)),
        "num_images": 0,
        "source_pixels": 0,
        "source_values": 0,
        "num_indices": 0,
        "payload_bits": 0,
        "ldpc_input_bits": 0,
        "ldpc_padding_bits": 0,
        "coded_bits": 0,
        "modulation_padding_bits": 0,
        "transmitted_bits": 0,
        "channel_symbols": 0,
        "bit_errors": 0,
        "index_errors": 0,
        "sent_index_min": None,
        "sent_index_max": None,
        "recovered_index_min": None,
        "recovered_index_max": None,
    }


def _update_range(record, prefix, tensor):
    if tensor.numel() == 0:
        return
    current_min = int(tensor.min().item())
    current_max = int(tensor.max().item())
    min_key, max_key = f"{prefix}_index_min", f"{prefix}_index_max"
    old_min, old_max = record[min_key], record[max_key]
    record[min_key] = current_min if old_min is None else min(old_min, current_min)
    record[max_key] = current_max if old_max is None else max(old_max, current_max)


def _update_stage_record(
    record, real_image, sent_indices, recovered_indices=None, channel_stats=None
):
    num_images, pixels, values = _source_sizes(real_image)
    record["num_images"] += num_images
    record["source_pixels"] += pixels
    record["source_values"] += values
    record["num_indices"] += int(sent_indices.numel())
    record["payload_bits"] += int(
        sent_indices.numel() * record["bits_per_index"]
    )
    _update_range(record, "sent", sent_indices.detach().cpu())

    if channel_stats is not None:
        for key in (
            "ldpc_input_bits",
            "ldpc_padding_bits",
            "coded_bits",
            "modulation_padding_bits",
            "transmitted_bits",
            "channel_symbols",
            "bit_errors",
        ):
            record[key] += int(channel_stats[key])

    if recovered_indices is not None:
        sent_cpu = sent_indices.detach().cpu()
        recovered_cpu = recovered_indices.detach().cpu()
        if sent_cpu.shape != recovered_cpu.shape:
            raise ValueError(
                "Recovered RAQ-RVQ index tensor shape does not match transmitted shape: "
                f"{tuple(recovered_cpu.shape)} != {tuple(sent_cpu.shape)}."
            )
        record["index_errors"] += int(torch.count_nonzero(sent_cpu != recovered_cpu))
        _update_range(record, "recovered", recovered_cpu)


def _rate(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _stream_length_reference(payload_bits, ldpc_code, modulation_bits):
    """Physical lengths if all payload bits used one legacy LDPC stream."""
    payload_bits = int(payload_bits)
    if ldpc_code is None:
        ldpc_input_bits = payload_bits
        coded_bits = payload_bits
    else:
        k, n = int(ldpc_code["k"]), int(ldpc_code["n"])
        blocks = (payload_bits + k - 1) // k
        ldpc_input_bits = blocks * k
        coded_bits = blocks * n
    modulation_padding_bits = (-coded_bits) % modulation_bits
    transmitted_bits = coded_bits + modulation_padding_bits
    return {
        "payload_bits": payload_bits,
        "ldpc_input_bits": ldpc_input_bits,
        "ldpc_padding_bits": ldpc_input_bits - payload_bits,
        "coded_bits": coded_bits,
        "modulation_padding_bits": modulation_padding_bits,
        "transmitted_bits": transmitted_bits,
        "channel_symbols": transmitted_bits // modulation_bits,
    }


def _add_stream_lengths(total, update):
    for key, value in update.items():
        total[key] = int(total.get(key, 0) + value)


def _finalize_stage_record(record, channel_enabled):
    result = dict(record)
    result["payload_bpp"] = _rate(result["payload_bits"], result["source_pixels"])
    result["coded_bpp"] = _rate(result["coded_bits"], result["source_pixels"])
    result["transmitted_bpp"] = _rate(
        result["transmitted_bits"], result["source_pixels"]
    )
    result["channel_uses_per_pixel"] = _rate(
        result["channel_symbols"], result["source_pixels"]
    )
    # This project historically reports channel uses / RGB source values as
    # the "transmission ratio" (for RGB it is uses-per-pixel divided by 3).
    result["transmission_ratio"] = _rate(
        result["channel_symbols"], result["source_values"]
    )
    if channel_enabled:
        result["ber"] = _rate(result["bit_errors"], result["payload_bits"])
        result["index_error_rate"] = _rate(
            result["index_errors"], result["num_indices"]
        )
    else:
        result["ber"] = None
        result["index_error_rate"] = None
    return result


def _total_record(stage_records, source_sizes, channel_enabled):
    num_images, source_pixels, source_values = source_sizes
    sum_keys = (
        "num_indices",
        "payload_bits",
        "ldpc_input_bits",
        "ldpc_padding_bits",
        "coded_bits",
        "modulation_padding_bits",
        "transmitted_bits",
        "channel_symbols",
        "bit_errors",
        "index_errors",
    )
    total = {
        "num_images": int(num_images),
        "source_pixels": int(source_pixels),
        "source_values": int(source_values),
    }
    for key in sum_keys:
        total[key] = int(sum(record[key] for record in stage_records))
    total["payload_bpp"] = _rate(total["payload_bits"], source_pixels)
    total["coded_bpp"] = _rate(total["coded_bits"], source_pixels)
    total["transmitted_bpp"] = _rate(total["transmitted_bits"], source_pixels)
    total["channel_uses_per_pixel"] = _rate(total["channel_symbols"], source_pixels)
    total["transmission_ratio"] = _rate(total["channel_symbols"], source_values)
    if channel_enabled:
        total["ber"] = _rate(total["bit_errors"], total["payload_bits"])
        total["index_error_rate"] = _rate(
            total["index_errors"], total["num_indices"]
        )
    else:
        total["ber"] = None
        total["index_error_rate"] = None
    return total


def _accumulate_rvq_quantization(records, out, real_image):
    """Aggregate model-side residual/quantizer diagnostics across the loader."""
    batch_size = int(real_image.shape[0])
    diagnostics = out.get("rvq_diagnostics")
    if not isinstance(diagnostics, (list, tuple)):
        raise ValueError(
            "test_raq_rvq_enabled=True requires model-side 'rvq_diagnostics'."
        )

    for scale_diag in diagnostics:
        scale_idx = int(scale_diag["scale_index"])
        stage_diags = scale_diag["stage_diagnostics"]
        if scale_idx not in records:
            records[scale_idx] = {
                "scale": scale_idx,
                "source_route": scale_diag["source_route"],
                "k_total": int(scale_diag["k_total"]),
                "stage_k_list": [int(k) for k in scale_diag["stage_k_list"]],
                "num_images": 0,
                "input_energy_sum": 0.0,
                "quantized_sum_energy_sum": 0.0,
                "final_residual_energy_sum": 0.0,
                "residual_energy_sums": [0.0] * len(stage_diags),
                "payload_bits": 0,
                "baseline_payload_bits": 0,
                "bit_budget_matches": True,
                "stage_diagnostics": [
                    {
                        "stage": int(stage["stage_index"]),
                        "num_embeddings": int(stage["k"]),
                        "bits_per_index": int(stage["bits_per_index"]),
                        "codebook_size": int(stage["codebook_size"]),
                        "sent_index_min": None,
                        "sent_index_max": None,
                        "payload_bits": 0,
                        "residual_energy_sum": 0.0,
                    }
                    for stage in stage_diags
                ],
            }

        record = records[scale_idx]
        if record["stage_k_list"] != [int(k) for k in scale_diag["stage_k_list"]]:
            raise ValueError(f"RAQ-RVQ stage-K layout changed for scale {scale_idx}.")
        record["num_images"] += batch_size
        record["input_energy_sum"] += float(scale_diag["input_mse_energy"]) * batch_size
        record["quantized_sum_energy_sum"] += (
            float(scale_diag["quantized_sum_mse_energy"]) * batch_size
        )
        record["final_residual_energy_sum"] += (
            float(scale_diag["final_residual_mse_energy"]) * batch_size
        )
        for stage_idx, energy in enumerate(scale_diag["residual_mse_energies"]):
            record["residual_energy_sums"][stage_idx] += float(energy) * batch_size
        record["payload_bits"] += int(scale_diag["payload_bits"])
        record["baseline_payload_bits"] += int(scale_diag["baseline_payload_bits"])
        record["bit_budget_matches"] = bool(
            record["bit_budget_matches"] and scale_diag["bit_budget_matches"]
        )

        for aggregate, stage in zip(record["stage_diagnostics"], stage_diags):
            stage_min, stage_max = int(stage["index_min"]), int(stage["index_max"])
            old_min, old_max = aggregate["sent_index_min"], aggregate["sent_index_max"]
            aggregate["sent_index_min"] = (
                stage_min if old_min is None else min(old_min, stage_min)
            )
            aggregate["sent_index_max"] = (
                stage_max if old_max is None else max(old_max, stage_max)
            )
            aggregate["payload_bits"] += int(stage["payload_bits"])
            aggregate["residual_energy_sum"] += (
                float(stage["residual_mse_energy"]) * batch_size
            )


def _finalize_rvq_quantization(records):
    per_scale = []
    for scale_idx in sorted(records):
        record = records[scale_idx]
        count = record["num_images"]
        stage_diagnostics = []
        for stage in record["stage_diagnostics"]:
            finalized = {key: value for key, value in stage.items() if key != "residual_energy_sum"}
            finalized["mean_residual_mse_energy"] = _rate(
                stage["residual_energy_sum"], count
            )
            stage_diagnostics.append(finalized)
        per_scale.append({
            "scale": record["scale"],
            "source_route": record["source_route"],
            "k_total": record["k_total"],
            "stage_k_list": record["stage_k_list"],
            "num_images": count,
            "mean_input_mse_energy": _rate(record["input_energy_sum"], count),
            "mean_residual_mse_energies": [
                _rate(energy, count) for energy in record["residual_energy_sums"]
            ],
            "mean_final_residual_mse_energy": _rate(
                record["final_residual_energy_sum"], count
            ),
            "mean_quantized_sum_mse_energy": _rate(
                record["quantized_sum_energy_sum"], count
            ),
            "stage_diagnostics": stage_diagnostics,
            "payload_bits": record["payload_bits"],
            "baseline_payload_bits": record["baseline_payload_bits"],
            "bit_budget_matches": bool(
                record["bit_budget_matches"]
                and record["payload_bits"] == record["baseline_payload_bits"]
            ),
        })
    return {
        "per_scale": per_scale,
        "all_bit_budgets_match": all(
            record["bit_budget_matches"] for record in per_scale
        ),
    }


def _build_diagnostics(
    mode,
    rvq_enabled,
    records,
    source_sizes,
    modulation=None,
    ldpc_code=None,
    rvq_quantization_records=None,
    single_stream_reference=None,
    stream_packing=None,
    packed_stream_totals=None,
):
    channel_enabled = mode in {"ldpc", "uncoded"}
    per_stage = [
        _finalize_stage_record(records[key], channel_enabled)
        for key in sorted(records, key=lambda item: (-1 if item[0] is None else item[0],
                                                     -1 if item[1] is None else item[1]))
    ]
    diagnostics = {
        "mode": mode,
        "rvq_enabled": bool(rvq_enabled),
        "per_stage": per_stage,
        "total": _total_record(per_stage, source_sizes, channel_enabled),
    }
    if stream_packing is not None:
        diagnostics["stream_packing"] = stream_packing
    if packed_stream_totals is not None:
        packed = {key: int(value) for key, value in packed_stream_totals.items()}
        total = diagnostics["total"]
        if packed["payload_bits"] != total["payload_bits"]:
            raise RuntimeError(
                "Combined-stream payload accounting does not match the nested "
                f"stage payloads: {packed['payload_bits']} != {total['payload_bits']}."
            )
        if packed["bit_errors"] != total["bit_errors"]:
            raise RuntimeError(
                "Combined-stream bit-error accounting does not match the decoded "
                f"stage segments: {packed['bit_errors']} != {total['bit_errors']}."
            )
        for key in (
            "ldpc_input_bits",
            "ldpc_padding_bits",
            "coded_bits",
            "modulation_padding_bits",
            "transmitted_bits",
            "channel_symbols",
            "bit_errors",
        ):
            total[key] = packed[key]
        total["payload_bpp"] = _rate(total["payload_bits"], source_sizes[1])
        total["coded_bpp"] = _rate(total["coded_bits"], source_sizes[1])
        total["transmitted_bpp"] = _rate(
            total["transmitted_bits"], source_sizes[1]
        )
        total["channel_uses_per_pixel"] = _rate(
            total["channel_symbols"], source_sizes[1]
        )
        total["transmission_ratio"] = _rate(
            total["channel_symbols"], source_sizes[2]
        )
        total["ber"] = _rate(total["bit_errors"], total["payload_bits"])

        combined_stream = dict(packed)
        combined_stream["source_pixels"] = int(source_sizes[1])
        combined_stream["source_values"] = int(source_sizes[2])
        combined_stream["payload_bpp"] = total["payload_bpp"]
        combined_stream["coded_bpp"] = total["coded_bpp"]
        combined_stream["transmitted_bpp"] = total["transmitted_bpp"]
        combined_stream["channel_uses_per_pixel"] = total[
            "channel_uses_per_pixel"
        ]
        combined_stream["transmission_ratio"] = total["transmission_ratio"]
        combined_stream["ber"] = total["ber"]
        diagnostics["combined_stream"] = combined_stream
    if rvq_enabled:
        diagnostics["rvq_quantization"] = _finalize_rvq_quantization(
            rvq_quantization_records or {}
        )
    if modulation is not None:
        diagnostics["modulation"] = modulation
        diagnostics["modulation_bits_per_symbol"] = _MODULATION_BITS[modulation]
    if ldpc_code is not None:
        diagnostics["ldpc"] = {
            "k": int(ldpc_code["k"]),
            "n": int(ldpc_code["n"]),
            "rate": float(ldpc_code.get("rate", ldpc_code["k"] / ldpc_code["n"])),
        }
    if mode == "ldpc":
        diagnostics["padding_notes"] = {
            "ldpc_padding_bits": (
                "zero information bits appended before LDPC encoding; excluded from payload_bits"
            ),
            "modulation_padding_bits": (
                "zero coded bits appended only to fill one modulation symbol; excluded from coded_bits"
            ),
            "transmitted_bits": "coded_bits + modulation_padding_bits",
        }
        reference = dict(single_stream_reference or {})
        if not reference and not rvq_enabled:
            # The legacy branch already is one stream, so its measured lengths
            # are the corresponding reference values.
            reference = {
                key: diagnostics["total"][key]
                for key in (
                    "payload_bits",
                    "ldpc_input_bits",
                    "ldpc_padding_bits",
                    "coded_bits",
                    "modulation_padding_bits",
                    "transmitted_bits",
                    "channel_symbols",
                )
            }
        reference["payload_bpp"] = _rate(reference.get("payload_bits", 0), source_sizes[1])
        reference["coded_bpp"] = _rate(reference.get("coded_bits", 0), source_sizes[1])
        reference["transmitted_bpp"] = _rate(
            reference.get("transmitted_bits", 0), source_sizes[1]
        )
        reference["channel_uses_per_pixel"] = _rate(
            reference.get("channel_symbols", 0), source_sizes[1]
        )
        reference["transmission_ratio"] = _rate(
            reference.get("channel_symbols", 0), source_sizes[2]
        )
        diagnostics["single_stream_reference"] = reference
        total = diagnostics["total"]
        total["single_stream_coded_bits"] = int(reference.get("coded_bits", 0))
        total["single_stream_transmitted_bits"] = int(
            reference.get("transmitted_bits", 0)
        )
        total["single_stream_channel_symbols"] = int(
            reference.get("channel_symbols", 0)
        )
        total["coded_bits_delta_vs_single_stream"] = (
            total["coded_bits"] - total["single_stream_coded_bits"]
        )
        total["transmitted_bits_delta_vs_single_stream"] = (
            total["transmitted_bits"] - total["single_stream_transmitted_bits"]
        )
        total["coded_bits_match_single_stream"] = (
            total["coded_bits"] == total["single_stream_coded_bits"]
        )
        total["transmitted_bits_match_single_stream"] = (
            total["transmitted_bits"] == total["single_stream_transmitted_bits"]
        )
        if rvq_enabled:
            baseline_payload = sum(
                scale["baseline_payload_bits"]
                for scale in diagnostics["rvq_quantization"]["per_scale"]
            )
            total["baseline_single_stage_payload_bits"] = int(baseline_payload)
            total["payload_bits_match_single_stage_budget"] = (
                total["payload_bits"] == baseline_payload
            )
    return diagnostics


def _transmit_ldpc_stream(
    flat_bits,
    target_snr,
    ldpc_code,
    device,
    modulate,
    calculate_llr,
    modulation_bits,
    ldpc_encode,
    ldpc_decode,
):
    """Transmit exactly one payload stream and expose both padding layers."""
    flat_bits = np.asarray(flat_bits, dtype=np.uint8).reshape(-1)
    payload_bits = len(flat_bits)

    if ldpc_code is None:
        ldpc_input_bits = payload_bits
    else:
        k = int(ldpc_code["k"])
        ldpc_input_bits = ((payload_bits + k - 1) // k) * k
    ldpc_padding_bits = ldpc_input_bits - payload_bits

    coded = np.asarray(ldpc_encode(flat_bits, code=ldpc_code)).reshape(-1)
    coded_bits = len(coded)
    modulation_padding_bits = (-coded_bits) % modulation_bits
    if modulation_padding_bits:
        transmitted = np.pad(coded, (0, modulation_padding_bits), "constant")
    else:
        transmitted = coded

    transmitted_tensor = torch.from_numpy(transmitted).float().to(device)
    symbols = modulate(transmitted_tensor)
    noisy_symbols = awgn_channel(symbols, target_snr)
    llrs = calculate_llr(noisy_symbols, target_snr, device).reshape(-1)
    # Modulation-only padding is never presented to the LDPC decoder.
    decoded = np.asarray(
        ldpc_decode(llrs[:coded_bits].detach().cpu().numpy(), ldpc_code)
    ).reshape(-1)
    decoded = decoded[:payload_bits]
    if len(decoded) < payload_bits:
        decoded = np.pad(decoded, (0, payload_bits - len(decoded)), "constant")
    decoded = decoded.astype(np.uint8, copy=False)

    stats = {
        "ldpc_input_bits": ldpc_input_bits,
        "ldpc_padding_bits": ldpc_padding_bits,
        "coded_bits": coded_bits,
        "modulation_padding_bits": modulation_padding_bits,
        "transmitted_bits": len(transmitted),
        "channel_symbols": int(symbols.numel()),
        "bit_errors": int(np.count_nonzero(decoded != flat_bits)),
    }
    return decoded, stats


@torch.no_grad()
def evaluate_no_channel(model, loader, device, return_diagnostics=False):
    model.eval()
    ms_ssim_scores = []
    psnr_scores = []
    records = {}
    rvq_quantization_records = {}
    total_images = total_pixels = total_values = 0
    saw_rvq = None

    for real_image in loader:
        real_image = real_image.to(device)
        out = model.forward_test(real_image)
        rvq_enabled, rvq_k_lists = _rvq_layout(out)
        if saw_rvq is None:
            saw_rvq = rvq_enabled
        elif saw_rvq != rvq_enabled:
            raise ValueError("forward_test changed RAQ-RVQ layout during one evaluation.")
        if rvq_enabled:
            _accumulate_rvq_quantization(rvq_quantization_records, out, real_image)

        reconstructed_images = model.reconstruct_from_indices(
            out["indices"],
            feature_shapes=out.get("feature_shapes"),
            codebooks=out.get("codebooks"),
        )
        ms_ssim, psnr = _image_quality(real_image, reconstructed_images)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)

        num_images, pixels, values = _source_sizes(real_image)
        total_images += num_images
        total_pixels += pixels
        total_values += values
        if rvq_enabled:
            for scale_idx, (stage_indices, stage_ks) in enumerate(
                zip(out["indices"], rvq_k_lists)
            ):
                for stage_idx, (indices, num_embeddings) in enumerate(
                    zip(stage_indices, stage_ks)
                ):
                    key = (scale_idx, stage_idx)
                    records.setdefault(
                        key, _new_stage_record(scale_idx, stage_idx, num_embeddings)
                    )
                    _update_stage_record(records[key], real_image, indices)

    diagnostics = _build_diagnostics(
        "no_channel",
        bool(saw_rvq),
        records,
        (total_images, total_pixels, total_values),
        rvq_quantization_records=rvq_quantization_records,
    )
    result = (np.mean(ms_ssim_scores), np.mean(psnr_scores))
    return (*result, diagnostics) if return_diagnostics else result


@torch.no_grad()
def evaluate_ldpc_channel(
    model,
    loader,
    num_embeddings_list,
    target_snr,
    ldpc_code,
    device,
    modulation="bpsk",
    return_diagnostics=False,
    stream_packing="per_stage",
):
    from communications.ldpc_coding import ldpc_decode, ldpc_encode

    modulators = {
        "bpsk": (bpsk_modulate, bpsk_llr),
        "qpsk": (qpsk_modulate, qpsk_llr),
        "16qam": (qam16_modulate, qam16_llr),
    }
    if modulation not in modulators:
        raise ValueError(f"Unsupported modulation: {modulation}")
    stream_packing = str(stream_packing).strip().lower()
    if stream_packing not in {"per_stage", "combined"}:
        raise ValueError(
            "stream_packing must be either 'per_stage' or 'combined', got "
            f"{stream_packing!r}."
        )
    modulate, calculate_llr = modulators[modulation]
    modulation_bits = _MODULATION_BITS[modulation]

    _reset_eval_seed()
    model.eval()
    ms_ssim_scores = []
    psnr_scores = []
    records = {}
    rvq_quantization_records = {}
    single_stream_reference = {}
    packed_stream_totals = {
        "payload_bits": 0,
        "ldpc_input_bits": 0,
        "ldpc_padding_bits": 0,
        "coded_bits": 0,
        "modulation_padding_bits": 0,
        "transmitted_bits": 0,
        "channel_symbols": 0,
        "bit_errors": 0,
    }
    total_images = total_pixels = total_values = 0
    saw_rvq = None

    for real_image in loader:
        real_image = real_image.to(device)
        out = model.forward_test(real_image)
        rvq_enabled, rvq_k_lists = _rvq_layout(out)
        if saw_rvq is None:
            saw_rvq = rvq_enabled
        elif saw_rvq != rvq_enabled:
            raise ValueError("forward_test changed RAQ-RVQ layout during one evaluation.")
        if rvq_enabled:
            _accumulate_rvq_quantization(rvq_quantization_records, out, real_image)
            payload_bits = sum(
                int(indices.numel()) * int(np.log2(num_embeddings))
                for stage_indices, stage_ks in zip(out["indices"], rvq_k_lists)
                for indices, num_embeddings in zip(stage_indices, stage_ks)
            )
            _add_stream_lengths(
                single_stream_reference,
                _stream_length_reference(payload_bits, ldpc_code, modulation_bits),
            )

        if rvq_enabled and stream_packing == "combined":
            segments = []
            recovered_indices_list = [[] for _ in out["indices"]]
            for scale_idx, (stage_indices, stage_ks) in enumerate(
                zip(out["indices"], rvq_k_lists)
            ):
                for stage_idx, (indices, num_embeddings) in enumerate(
                    zip(stage_indices, stage_ks)
                ):
                    flat_bits, original_shape, _ = index_tensor_to_bits(
                        indices, num_embeddings
                    )
                    flat_bits = np.asarray(flat_bits, dtype=np.uint8).reshape(-1)
                    segments.append(
                        (
                            scale_idx,
                            stage_idx,
                            indices,
                            num_embeddings,
                            original_shape,
                            flat_bits,
                        )
                    )

            combined_bits = np.concatenate(
                [segment[-1] for segment in segments]
            ).astype(np.uint8, copy=False)
            decoded_bits, channel_stats = _transmit_ldpc_stream(
                combined_bits,
                target_snr,
                ldpc_code,
                device,
                modulate,
                calculate_llr,
                modulation_bits,
                ldpc_encode,
                ldpc_decode,
            )
            _add_stream_lengths(
                packed_stream_totals,
                {"payload_bits": len(combined_bits), **channel_stats},
            )

            offset = 0
            for (
                scale_idx,
                stage_idx,
                indices,
                num_embeddings,
                original_shape,
                sent_bits,
            ) in segments:
                end = offset + len(sent_bits)
                recovered_bits = decoded_bits[offset:end]
                offset = end
                recovered = bits_to_index_tensor(
                    recovered_bits, original_shape, num_embeddings
                ).to(device)
                recovered_indices_list[scale_idx].append(recovered)

                key = (scale_idx, stage_idx)
                records.setdefault(
                    key, _new_stage_record(scale_idx, stage_idx, num_embeddings)
                )
                _update_stage_record(
                    records[key], real_image, indices, recovered
                )
                records[key]["bit_errors"] += int(
                    np.count_nonzero(recovered_bits != sent_bits)
                )
            if offset != len(decoded_bits):
                raise RuntimeError(
                    "Combined-stream decoded payload was not consumed exactly: "
                    f"{offset} != {len(decoded_bits)}."
                )
        elif rvq_enabled:
            recovered_indices_list = []
            for scale_idx, (stage_indices, stage_ks) in enumerate(
                zip(out["indices"], rvq_k_lists)
            ):
                recovered_stages = []
                for stage_idx, (indices, num_embeddings) in enumerate(
                    zip(stage_indices, stage_ks)
                ):
                    flat_bits, original_shape, _ = index_tensor_to_bits(
                        indices, num_embeddings
                    )
                    decoded_bits, channel_stats = _transmit_ldpc_stream(
                        flat_bits,
                        target_snr,
                        ldpc_code,
                        device,
                        modulate,
                        calculate_llr,
                        modulation_bits,
                        ldpc_encode,
                        ldpc_decode,
                    )
                    recovered = bits_to_index_tensor(
                        decoded_bits, original_shape, num_embeddings
                    ).to(device)
                    recovered_stages.append(recovered)

                    key = (scale_idx, stage_idx)
                    records.setdefault(
                        key, _new_stage_record(scale_idx, stage_idx, num_embeddings)
                    )
                    _update_stage_record(
                        records[key], real_image, indices, recovered, channel_stats
                    )
                recovered_indices_list.append(recovered_stages)
        else:
            # Keep the historical flat path byte-for-byte in layout: all scales
            # share one LDPC stream.  This preserves old checkpoint evaluation
            # and, importantly, its single LDPC padding boundary.
            bit_num_embeddings = out.get("num_embeddings_list", num_embeddings_list)
            flat_bits, original_spatial_dims, original_num_embeddings = indices_to_bits(
                out["indices"], bit_num_embeddings
            )
            decoded_bits, channel_stats = _transmit_ldpc_stream(
                flat_bits,
                target_snr,
                ldpc_code,
                device,
                modulate,
                calculate_llr,
                modulation_bits,
                ldpc_encode,
                ldpc_decode,
            )
            recovered_indices_list = bits_to_indices(
                decoded_bits, original_spatial_dims, original_num_embeddings
            )
            recovered_indices_list = [idx.to(device) for idx in recovered_indices_list]

            # A flat diagnostic is intentionally a single stream; allocating
            # LDPC padding to individual scales would be misleading.
            key = (None, None)
            records.setdefault(key, _new_stage_record(None, None, 1))
            record = records[key]
            num_images, pixels, values = _source_sizes(real_image)
            record["num_images"] += num_images
            record["source_pixels"] += pixels
            record["source_values"] += values
            record["num_indices"] += sum(int(idx.numel()) for idx in out["indices"])
            record["payload_bits"] += len(flat_bits)
            for stat_key in (
                "ldpc_input_bits",
                "ldpc_padding_bits",
                "coded_bits",
                "modulation_padding_bits",
                "transmitted_bits",
                "channel_symbols",
                "bit_errors",
            ):
                record[stat_key] += int(channel_stats[stat_key])
            record["index_errors"] += sum(
                int(torch.count_nonzero(sent.detach().cpu().squeeze(0) != recovered.detach().cpu()))
                for sent, recovered in zip(out["indices"], recovered_indices_list)
            )

        reconstructed_images = model.reconstruct_from_indices(
            recovered_indices_list,
            feature_shapes=out.get("feature_shapes"),
            codebooks=out.get("codebooks"),
        )
        ms_ssim, psnr = _image_quality(real_image, reconstructed_images)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)
        num_images, pixels, values = _source_sizes(real_image)
        total_images += num_images
        total_pixels += pixels
        total_values += values

    diagnostics = _build_diagnostics(
        "ldpc",
        bool(saw_rvq),
        records,
        (total_images, total_pixels, total_values),
        modulation=modulation,
        ldpc_code=ldpc_code,
        rvq_quantization_records=rvq_quantization_records,
        single_stream_reference=single_stream_reference,
        stream_packing=stream_packing,
        packed_stream_totals=(
            packed_stream_totals
            if bool(saw_rvq) and stream_packing == "combined"
            else None
        ),
    )
    result = (np.mean(ms_ssim_scores), np.mean(psnr_scores))
    return (*result, diagnostics) if return_diagnostics else result


@torch.no_grad()
def evaluate_uncoded_channel(
    model, loader, num_embeddings_list, target_snr, device
):
    _reset_eval_seed()
    model.eval()
    ms_ssim_scores = []
    psnr_scores = []

    for real_image in loader:
        real_image = real_image.to(device)
        out = model.forward_test(real_image)
        rvq_enabled, _ = _rvq_layout(out)
        if rvq_enabled:
            raise ValueError(
                "evaluate_uncoded_channel does not support nested test-time RAQ-RVQ; "
                "use evaluate_ldpc_channel or evaluate_no_channel."
            )
        bit_num_embeddings = out.get("num_embeddings_list", num_embeddings_list)
        flat_bits, original_spatial_dims, original_num_embeddings = indices_to_bits(
            out["indices"], bit_num_embeddings
        )

        bits_tensor = torch.from_numpy(flat_bits).float().to(device)
        symbols = bpsk_modulate(bits_tensor)
        noisy_symbols = awgn_channel(symbols, target_snr)
        decoded_bits = bpsk_demodulate(noisy_symbols).cpu().numpy()

        recovered_indices_list = bits_to_indices(
            decoded_bits, original_spatial_dims, original_num_embeddings
        )
        recovered_indices_list = [idx.to(device) for idx in recovered_indices_list]
        reconstructed_images = model.reconstruct_from_indices(
            recovered_indices_list,
            feature_shapes=out.get("feature_shapes"),
            codebooks=out.get("codebooks"),
        )

        ms_ssim, psnr = _image_quality(real_image, reconstructed_images)
        ms_ssim_scores.append(ms_ssim)
        psnr_scores.append(psnr)

    return np.mean(ms_ssim_scores), np.mean(psnr_scores)
