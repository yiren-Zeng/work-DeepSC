import math

import torch


@torch.no_grad()
def calculate_bpp(model, loader, num_embeddings_list, device):
    model.eval()
    bits_per_token = [math.log2(k) for k in num_embeddings_list]
    total_bits = 0.0
    total_pixels = 0.0
    layer_bits_accumulated = [0.0] * len(num_embeddings_list)

    for images in loader:
        images = images.to(device)
        batch_size, _, height, width = images.shape
        batch_pixels = batch_size * height * width
        indices_list = model.forward_test(images)["indices"]

        batch_bits = 0.0
        for layer_idx, indices in enumerate(indices_list):
            _, token_h, token_w = indices.shape
            layer_bits = batch_size * token_h * token_w * bits_per_token[layer_idx]
            layer_bits_accumulated[layer_idx] += layer_bits
            batch_bits += layer_bits

        total_bits += batch_bits
        total_pixels += batch_pixels

    layer_bpp = [layer_bits / total_pixels for layer_bits in layer_bits_accumulated]
    average_bpp = total_bits / total_pixels
    return {
        "bits_per_token": bits_per_token,
        "layer_bpp": layer_bpp,
        "average_bpp": average_bpp,
    }
