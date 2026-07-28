"""Backward-compatible communication evaluation aliases."""

from evaluation.quality import evaluate_ldpc_channel, evaluate_uncoded_channel


def evaluate_metrics_with_channel(model, loader, num_embeddings_list, target_snr, ldpc_code, device):
    return evaluate_ldpc_channel(model, loader, num_embeddings_list, target_snr, ldpc_code, device)


def evaluate_metrics_with_channel_withoutLDPC(model, loader, num_embeddings_list, target_snr, device):
    return evaluate_uncoded_channel(model, loader, num_embeddings_list, target_snr, device)
