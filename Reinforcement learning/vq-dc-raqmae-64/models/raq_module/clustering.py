from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class RAQClusterArgs:
    max_iter: int = 100
    epsilon: float = 1e-4
    temp: float = 1.0
    device: str = "cpu"

def _compute_kernel(x, y, device, kernel="multiscale"):
    def gaussian_kernel(a, b, sigma):
        a = a.to(device); b = b.to(device)
        a_norm = (a ** 2).sum(1).view(-1, 1)
        b_norm = (b ** 2).sum(1).view(1, -1)
        dist = a_norm + b_norm - 2.0 * (a @ b.t())
        return torch.exp(-dist / (2 * sigma ** 2))

    if kernel == "multiscale":
        sigmas = [0.2, 0.5, 1.0, 2.0, 5.0]
        k = sum(gaussian_kernel(x, y, s) for s in sigmas) / len(sigmas)
    else:
        k = gaussian_kernel(x, y, 1.0)
    return k

def MMD(x, y, device, kernel="multiscale"):
    x_kernel = _compute_kernel(x, x, device, kernel=kernel)
    y_kernel = _compute_kernel(y, y, device, kernel=kernel)
    xy_kernel = _compute_kernel(x, y, device, kernel=kernel)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

def dkm(weights, k, args: RAQClusterArgs):
    """
    Soft K-means over 'weights' to obtain k centroids.
    Args:
        weights: [N, D]
        k: number of clusters
        args: RAQClusterArgs
    Returns:
        centroids: [k, D]
        closest_indices: [N] assigned cluster id for each input vector
    """
    device = args.device
    max_iterations = args.max_iter
    epsilon = args.epsilon
    temp = args.temp

    N, D = weights.size()
    indices = torch.randperm(N, device=device)[:k]
    centroids = weights[indices].clone()

    for _ in range(max_iterations):
        # responsibilities
        dist = torch.cdist(weights, centroids, p=2)  # [N,k]
        logits = -dist / (temp + 1e-8)
        R = torch.softmax(logits, dim=1)             # [N,k]

        # M-step
        numer = R.t() @ weights                       # [k,D]
        denom = R.sum(0).unsqueeze(1) + 1e-8
        new_centroids = numer / denom

        # convergence
        if torch.norm(new_centroids - centroids) < epsilon:
            centroids = new_centroids
            break
        centroids = new_centroids

    closest = torch.argmin(torch.cdist(weights, centroids, p=2), dim=1)
    return centroids, closest

def inverse_dkm(src_weight, k_target, embedding_dim=None, args: RAQClusterArgs = RAQClusterArgs()):
    """
    Inverse-KMeans (IKM): learn a larger synthetic set W_tilde (k_target > |src|) such that
    dkm(W_tilde, k=|src|) produces centroids matching src_weight (via MMD minimization).
    This is compute-heavy; keep k_target modest or reduce max_iter.
    """
    device = args.device
    src = src_weight.to(device).detach()
    if embedding_dim is None:
        embedding_dim = src.shape[1]

    W = (torch.randn(k_target, embedding_dim, device=device) * 0.125).requires_grad_(True)
    opt = torch.optim.AdamW([W], lr=5e-3, weight_decay=0.01)

    best_W = None
    best_loss = float('inf')

    for i in range(max(2000, args.max_iter * 20)):  # a practical default loop
        centroids, _ = dkm(W, k=src.shape[0], args=args)
        loss = MMD(src, centroids, device, kernel="multiscale")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_W = W.detach().clone()

        loss.backward()
        opt.step()
        opt.zero_grad()

    return best_W, best_loss
