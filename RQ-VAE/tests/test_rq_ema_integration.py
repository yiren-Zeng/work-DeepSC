import torch

from losses.deepsc_loss import DeepSCLoss
from models.channel import FiniteBlocklengthChannel
from models.deepsc import DeepSC
from models.rq_ema_quantizer import RQEMAQuantizer
from models.vector_quantizer import VectorQuantizer
from training.schedules import compute_schedule
from utils.bit_utils import bits_to_indices, indices_to_bits


torch.set_num_threads(1)


def _make_model(quantizer_type="rq_ema"):
    return DeepSC(
        in_channels=3,
        out_channels=3,
        num_downsample_blocks=2,
        base_channels=4,
        num_embeddings_list=[4, 2],
        embedding_dim_list=[8, 16],
        commitment_cost=0.25,
        device=torch.device("cpu"),
        strides=[2, 2],
        skip_dropout_p=[0.0],
        norm_type="group",
        norm_groups=4,
        activation="silu",
        encoder_res_blocks=0,
        decoder_res_blocks=0,
        upsample_mode="nearest",
        use_cascade_downsample=False,
        use_bottleneck_attention=False,
        quantizer_type=quantizer_type,
        quantizer_axis_list=["patch", "patch"],
        cvq_codeword_shapes=[None, None],
        rq_depth_list=[2, 2],
        rq_ema_decay=0.99,
        rq_restart_unused_codes=False,
        rq_shared_codebook=True,
    )


def test_finite_blocklength_channel_preserves_bhwd_indices():
    torch.manual_seed(5)
    channel = FiniteBlocklengthChannel(
        channel_coding_rate=0.5,
        coded_block_length_bits=256,
        device=torch.device("cpu"),
    )
    indices = torch.randint(0, 4, (1, 5, 7, 2), dtype=torch.long)
    corrupted, ber = channel.apply_channel_noise(
        indices,
        num_embeddings=4,
        snr_db=torch.tensor(0.0),
        rc=0.5,
        mod_bits=1,
    )
    assert corrupted.shape == indices.shape
    assert corrupted.dtype == indices.dtype
    assert int(corrupted.min()) >= 0 and int(corrupted.max()) < 4
    assert torch.is_tensor(ber) and ber.ndim == 0


def test_eval_no_channel_indices_reconstruct_the_clean_quantized_decoder_result():
    torch.manual_seed(6)
    model = _make_model("rq_ema").eval()
    image = torch.randn(1, 3, 16, 16)

    with torch.no_grad():
        encoded = model.forward_test(image)
        reconstructed_from_indices = model.reconstruct_from_indices(
            encoded["indices"], feature_shapes=encoded["feature_shapes"]
        )

        features = model.semantic_encoder(image)
        features[-1] = model.bottleneck_attention(features[-1])
        clean_quantized = [
            quantizer(feature)[1]
            for quantizer, feature in zip(model.vector_quantizers, features)
        ]
        expected = model.swinir_enhance(model.semantic_decoder(clean_quantized))

    assert [tuple(indices.shape) for indices in encoded["indices"]] == [
        (1, 8, 8, 2),
        (1, 4, 4, 2),
    ]
    assert reconstructed_from_indices.shape == image.shape
    assert torch.allclose(reconstructed_from_indices, expected, atol=1e-6, rtol=1e-5)


def test_4608_bit_rate_and_hwd_bitstream_roundtrip():
    torch.manual_seed(7)
    indices = [
        torch.randint(0, 4, (1, 32, 32, 2), dtype=torch.long),
        torch.randint(0, 2, (1, 16, 16, 2), dtype=torch.long),
    ]
    bitstream, hwd_shapes, codebook_sizes, stats = indices_to_bits(
        indices, [4, 2], return_stats=True
    )
    recovered = bits_to_indices(bitstream, hwd_shapes, codebook_sizes)

    assert stats["per_scale_bits"] == [4096, 512]
    assert stats["total_bits"] == 4608
    assert len(bitstream) == 4608
    assert hwd_shapes == [(32, 32, 2), (16, 16, 2)]
    assert all(torch.equal(actual, original.squeeze(0)) for actual, original in zip(recovered, indices))
    source_bpp = stats["total_bits"] / (256 * 256)
    transmission_ratio = source_bpp / (0.5 * 1 * 3)
    assert source_bpp == 0.0703125
    assert transmission_ratio == 0.046875


def test_legacy_simvq_still_instantiates_and_encodes():
    torch.manual_seed(8)
    model = _make_model("simvq").eval()
    assert len(model.vector_quantizers) == 2
    assert all(isinstance(quantizer, VectorQuantizer) for quantizer in model.vector_quantizers)
    assert not any(isinstance(quantizer, RQEMAQuantizer) for quantizer in model.vector_quantizers)
    with torch.no_grad():
        encoded = model.forward_test(torch.randn(1, 3, 16, 16))
    assert [indices.ndim for indices in encoded["indices"]] == [3, 3]


def test_deepsc_loss_applies_the_global_quarter_weight_exactly_once():
    criterion = DeepSCLoss(
        layer_weights=[1.0, 1.0],
        mse_weight=1.0,
        ms_ssim_weight=0.0,
        lpips_weight=0.0,
        quantization_weight=0.25,
    )
    image = torch.zeros(1, 3, 4, 4)
    reconstruction_loss, quantization_loss = criterion(
        image,
        image.clone(),
        [torch.tensor(2.0), torch.tensor(6.0)],
    )
    assert reconstruction_loss.item() == 0.0
    assert quantization_loss.item() == 2.0


def test_rq_schedule_keeps_both_scale_weights_at_one_in_every_phase():
    class ScheduleConfig:
        QUANTIZER_TYPE = "rq_ema"
        UNET_DEPTH = 2
        PHASE1_END = 0.1
        PHASE2_END = 0.4
        SKIP_DROPOUT_P_INIT = [0.1]
        SKIP_DROPOUT_P_FINAL = [0.0]
        LAYER_LOSS_WEIGHTS_INIT = [3.0, 4.0]
        LAYER_LOSS_WEIGHTS_FINAL = [0.1, 0.2]
        CHANNEL_PROB_START_EPOCH = 20
        CHANNEL_PROB_END_EPOCH = 80

    for epoch in (0, 20, 99):
        _, weights, _, _ = compute_schedule(epoch, 100, ScheduleConfig)
        assert weights == [1.0, 1.0]
