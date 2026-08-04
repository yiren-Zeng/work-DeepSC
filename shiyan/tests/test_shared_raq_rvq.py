"""Contracts for the trained strict shared-codebook RAQ-RVQ mode."""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from losses.deepsc_loss import DeepSCLoss
from models.deepsc import DeepSC
from models.shared_raq_rvq import quantize_shared_raq_rvq
from models.vector_quantizer import VectorQuantizer
from monitoring.codebook import compute_codebook_utilization


torch.set_num_threads(1)


def _build_model(shared):
    return DeepSC(
        in_channels=3,
        out_channels=3,
        num_downsample_blocks=1,
        base_channels=4,
        num_embeddings_list=[4],
        embedding_dim_list=[8],
        commitment_cost=0.25,
        device="cpu",
        strides=[2],
        norm_groups=1,
        encoder_res_blocks=1,
        decoder_res_blocks=1,
        use_raq=True,
        raq_target_list=[4],
        raq_min_trg=2,
        raq_max_trg=4,
        raq_recon_grad_mode="ste",
        use_shared_raq_rvq=shared,
        shared_raq_rvq_depth=2,
    )


def _has_finite_nonzero_grad(parameters):
    grads = [
        parameter.grad
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(grads) and all(torch.isfinite(grad).all() for grad in grads) and any(
        int(torch.count_nonzero(grad)) > 0 for grad in grads
    )


def test_exact_depth_average_indices_sum_and_single_ste():
    quantizer = VectorQuantizer(2, 1, commitment_cost=0.25)
    inputs = torch.tensor(
        [[[[0.0, 0.4, 1.0, 1.8]]]], requires_grad=True
    )
    codebook = torch.tensor([[0.0], [1.0]], requires_grad=True)

    result = quantize_shared_raq_rvq(
        quantizer, inputs, codebook, depth=2
    )

    expected_indices = torch.tensor(
        [[[[0, 0], [0, 0], [1, 0], [1, 1]]]]
    )
    expected_quantized = torch.tensor([[[[0.0, 0.0, 1.0, 2.0]]]])
    expected_per_depth = torch.tensor([0.20, 0.05])
    expected_component = expected_per_depth.mean()

    assert torch.equal(result["indices"], expected_indices)
    assert torch.allclose(result["quantized"], expected_quantized)
    assert torch.allclose(result["quantized_raw"], expected_quantized)
    assert torch.allclose(
        result["codebook_loss_per_depth"], expected_per_depth
    )
    assert torch.allclose(
        result["commitment_loss_per_depth"], expected_per_depth
    )
    assert torch.allclose(
        result["loss"], expected_component + 0.25 * expected_component
    )

    # Reconstruction sees one identity bridge, not one bridge per depth.
    result["quantized"].sum().backward()
    assert torch.allclose(inputs.grad, torch.ones_like(inputs))
    assert codebook.grad is None

    inputs.grad = None
    codebook.grad = None
    result = quantize_shared_raq_rvq(
        quantizer, inputs, codebook, depth=2
    )
    result["loss"].backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert codebook.grad is not None and torch.isfinite(codebook.grad).all()


def test_shared_mode_adds_no_state_and_generates_one_codebook_per_scale():
    torch.manual_seed(7)
    ordinary = _build_model(False)
    torch.manual_seed(7)
    shared = _build_model(True).train()
    shared.set_channel_prob(0.0)

    assert list(ordinary.state_dict()) == list(shared.state_dict())
    assert len(shared.raqs_rvq_stage2) == 0

    image = torch.randn(1, 3, 8, 8)
    with mock.patch.object(
        shared,
        "_generate_raq_codebook",
        wraps=shared._generate_raq_codebook,
    ) as generator_spy:
        out = shared.forward_train(image, raq_trg_list=[4])

    assert generator_spy.call_count == 1
    assert out["shared_raq_rvq_enabled"]
    assert out["rvq_k_lists"] == [[4, 4]]
    assert len(out["rvq_indices_list"][0]) == 2
    assert (
        out["rvq_codebooks_list"][0][0]
        is out["rvq_codebooks_list"][0][1]
    )
    for indices in out["rvq_indices_list"][0]:
        assert tuple(indices.shape) == (1, 4, 4)
        assert int(indices.min()) >= 0
        assert int(indices.max()) < 4


def test_joint_loss_updates_encoder_decoder_source_projection_and_raq():
    torch.manual_seed(8)
    model = _build_model(True).train()
    model.set_channel_prob(0.0)
    image = torch.randn(1, 3, 8, 8)
    out = model.forward_train(image, raq_trg_list=[4])

    criterion = DeepSCLoss(layer_weights=[0.25])
    recon_loss, auxiliary_loss = criterion(
        image,
        out["reconstructed_images_src"],
        out["vq_losses_src"],
        out["reconstructed_images_raq"],
        out["vq_losses_raq"],
        out["W_trg_list"],
        out["z_q_src_list"],
        out["z_q_raq_list"],
        out["source_codebooks_list"],
    )
    total_loss = recon_loss + auxiliary_loss
    assert torch.isfinite(total_loss)
    total_loss.backward()

    assert _has_finite_nonzero_grad(model.semantic_encoder.parameters())
    assert _has_finite_nonzero_grad(model.semantic_decoder.parameters())
    assert _has_finite_nonzero_grad(
        model.vector_quantizers[0].codebook.proj.parameters()
    )
    assert _has_finite_nonzero_grad(model.raqs[0].parameters())
    assert model.vector_quantizers[0].codebook.embed.weight.grad is None
    assert model.raqs[0].trg_embed.embed.weight.grad is None

    details = out["rvq_loss_details"][0]
    expected_vq = (
        details["codebook_loss_per_depth"].mean()
        + 0.25 * details["commitment_loss_per_depth"].mean()
    )
    torch.testing.assert_close(out["vq_losses_raq"][0], expected_vq)


def test_shared_forward_test_round_trip_reuses_one_codebook():
    torch.manual_seed(9)
    model = _build_model(True).eval()
    image = torch.randn(1, 3, 8, 8)

    with mock.patch.object(
        model,
        "_generate_raq_codebook",
        wraps=model._generate_raq_codebook,
    ) as generator_spy, torch.no_grad():
        out = model.forward_test(image)
        reconstructed = model.reconstruct_from_indices(
            out["indices"],
            feature_shapes=out["feature_shapes"],
            codebooks=out["codebooks"],
        )

    assert generator_spy.call_count == 1
    assert out["branch"] == "shared_raq_rvq"
    assert out["rvq_k_lists"] == [[4, 4]]
    assert out["codebooks"][0][0] is out["codebooks"][0][1]
    assert out["rvq_diagnostics"][0]["shared_codebook_identity_verified"]
    assert out["rvq_diagnostics"][0]["bit_budget_matches"]
    assert reconstructed.shape == image.shape
    assert torch.isfinite(F.mse_loss(reconstructed, image))


def test_codebook_monitor_aggregates_both_residual_depths():
    torch.manual_seed(10)
    model = _build_model(True).eval()
    stats = compute_codebook_utilization(
        model,
        [torch.randn(1, 3, 8, 8)],
        device="cpu",
    )

    assert stats["raq_rvq_depth"] == 2
    assert int(stats["src"][0]["usage_counts"].sum().item()) == 16
    assert int(stats["raq"][0]["usage_counts"].sum().item()) == 32


class SharedRaqRvqContractTest(unittest.TestCase):
    def test_exact_depth_average_indices_sum_and_single_ste(self):
        test_exact_depth_average_indices_sum_and_single_ste()

    def test_shared_mode_adds_no_state_and_generates_once(self):
        test_shared_mode_adds_no_state_and_generates_one_codebook_per_scale()

    def test_joint_loss_updates_all_expected_modules(self):
        test_joint_loss_updates_encoder_decoder_source_projection_and_raq()

    def test_shared_forward_test_round_trip(self):
        test_shared_forward_test_round_trip_reuses_one_codebook()

    def test_codebook_monitor_aggregates_both_depths(self):
        test_codebook_monitor_aggregates_both_residual_depths()


if __name__ == "__main__":
    unittest.main()
