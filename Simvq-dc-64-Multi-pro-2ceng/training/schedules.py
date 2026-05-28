def compute_schedule(epoch, num_epochs, cfg):
    """
    Return skip-dropout probabilities, VQ-loss weights, channel probability,
    and phase label.
    """
    phase1_end = int(cfg.PHASE1_END * num_epochs)
    phase2_end = int(cfg.PHASE2_END * num_epochs)
    channel_start = cfg.CHANNEL_PROB_START_EPOCH
    channel_end = cfg.CHANNEL_PROB_END_EPOCH

    if epoch < phase1_end:
        dropout_p = list(cfg.SKIP_DROPOUT_P_INIT)
        loss_weights = list(cfg.LAYER_LOSS_WEIGHTS_INIT)
        phase_desc = "Phase1-拓荒"
    elif epoch < phase2_end:
        progress = (epoch - phase1_end) / (phase2_end - phase1_end)
        dropout_p = [
            init * (1 - progress) + final * progress
            for init, final in zip(cfg.SKIP_DROPOUT_P_INIT, cfg.SKIP_DROPOUT_P_FINAL)
        ]
        loss_weights = [
            init + (final - init) * progress
            for init, final in zip(cfg.LAYER_LOSS_WEIGHTS_INIT, cfg.LAYER_LOSS_WEIGHTS_FINAL)
        ]
        phase_desc = f"Phase2-退火({progress:.0%})"
    else:
        dropout_p = list(cfg.SKIP_DROPOUT_P_FINAL)
        loss_weights = list(cfg.LAYER_LOSS_WEIGHTS_FINAL)
        phase_desc = "Phase3-微调"

    if epoch < channel_start:
        channel_prob = 0.0
    elif epoch < channel_end:
        progress = (epoch - channel_start) / max(channel_end - channel_start, 1)
        channel_prob = max(0.0, min(1.0, progress))
    else:
        channel_prob = 1.0

    return dropout_p, loss_weights, channel_prob, phase_desc
