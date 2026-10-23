"""Evaluate joint CSCR using the unchanged RVQ reconstruction.

Transport records are per scale (stage=None): BER counts wire labels and index
errors count restored joint tuples. Original per-stage quantizer diagnostics
remain separate.
"""

import math

import numpy as np
import torch

from evaluation import quality
from utils.bit_utils import bits_to_index_tensor, index_tensor_to_bits
from utils.joint_cscr import (
    build_joint_cscr_mappings, pack_joint_indices, unpack_joint_indices,
)


@torch.no_grad()
def evaluate_joint(
    model, loader, device, *, return_diagnostics=False,
    target_snr=None, ldpc_code=None, modulation=None, channel_type="awgn",
    rician_k_factor=10.0, channel_seed=42, transport=None,
):
    channel_enabled = transport is not None
    if channel_enabled:
        quality._reset_eval_seed(channel_seed)
    model.eval()
    scores = []
    records, quantization, reference, packed = {}, {}, {}, {}
    source_sizes = [0, 0, 0]
    layout = None
    for image in loader:
        image = image.to(device)
        out = model.forward_test(image)
        sizes = quality._rvq_layout(out)
        if layout is not None and sizes != layout:
            raise ValueError("Joint evaluation requires a fixed RVQ stage layout")
        layout = sizes
        quality._accumulate_rvq_quantization(quantization, out, image)
        mappings = build_joint_cscr_mappings(out.get("codebooks"), sizes)
        segments = []
        for indices, stage_sizes, mapping in zip(out["indices"], sizes, mappings):
            joint = pack_joint_indices(indices, stage_sizes)
            joint_k = math.prod(stage_sizes)
            wire = mapping["index_to_label"][joint]
            bits, shape, _ = index_tensor_to_bits(wire, joint_k)
            segments.append((joint, joint_k, bits, shape, mapping, stage_sizes))
        payload = np.concatenate([segment[2] for segment in segments])
        if channel_enabled:
            decoded, stats = transport(payload)
            quality._add_stream_lengths(packed, {"payload_bits": len(payload), **stats})
            quality._add_stream_lengths(reference, quality._stream_length_reference(
                len(payload), ldpc_code, quality._MODULATION_BITS[modulation],
            ))
        else:
            decoded = payload
        recovered_stages = []
        offset = 0
        for scale, (joint, joint_k, bits, shape, mapping, stage_sizes) in enumerate(segments):
            received_bits = decoded[offset:offset + len(bits)]
            offset += len(bits)
            recovered = bits_to_index_tensor(received_bits, shape, joint_k).to(device)
            recovered = mapping["label_to_index"][recovered]
            recovered_stages.append(unpack_joint_indices(recovered, stage_sizes))
            key = (scale, None)
            record = records.setdefault(key, quality._new_stage_record(scale, None, joint_k))
            quality._update_stage_record(record, image, joint, recovered)
            record["bit_errors"] += int(np.count_nonzero(received_bits != bits))
        if offset != len(decoded):
            raise RuntimeError("Joint decoded payload was not consumed exactly")
        reconstruction = model.reconstruct_from_indices(
            recovered_stages, feature_shapes=out.get("feature_shapes"),
            codebooks=out.get("codebooks"),
        )
        scores.append(quality._image_quality(image, reconstruction))
        source_sizes = [a + b for a, b in zip(source_sizes, quality._source_sizes(image))]
    diagnostics = quality._build_diagnostics(
        "ldpc" if channel_enabled else "no_channel", records, source_sizes,
        modulation=modulation if channel_enabled else None,
        ldpc_code=ldpc_code if channel_enabled else None,
        rvq_quantization_records=quantization,
        single_stream_reference=reference if channel_enabled else None,
        stream_packing="combined" if channel_enabled else None,
        packed_stream_totals=packed if channel_enabled and packed else None,
    )
    diagnostics["cscr"] = {
        "enabled": True, "mode": "joint-cscr", "scope": "joint_per_scale",
        "method": "cosine_greedy_gray_v1",
    }
    diagnostics["joint_packing"] = {
        "enabled": True, "stage_k_lists": layout,
        "joint_k_per_scale": [math.prod(sizes) for sizes in layout or []],
        "index_order": "mixed_radix_stage_order",
        "transport_records": "per_scale_joint_index; stage=None",
        "ber_definition": "transmitted label bits after LDPC decoding",
        "index_error_definition": "restored joint tuples; any stage differs",
    }
    if channel_enabled:
        diagnostics.update(channel_type=channel_type, channel_seed=int(channel_seed))
        if channel_type == "rician":
            diagnostics.update(rician_k_factor=float(rician_k_factor),
                               receiver_csi="perfect_per_symbol")
    result = tuple(np.mean([score[i] for score in scores]) for i in (0, 1))
    return (*result, diagnostics) if return_diagnostics else result
