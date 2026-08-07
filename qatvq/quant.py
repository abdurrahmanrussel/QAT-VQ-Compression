"""Quantization + Vector Quantization building blocks.

This is where the thesis's original method is FIXED.

Original thesis QAT+VQ failed because:
  * VQ codebook was tiny (k=16/32 per FFN layer) -> destroyed weight info
  * VQ applied to activations -> error compounds every forward pass
  * no proper recovery: assignments frozen, codebook never adapted to task loss

Fix here:
  * Product Quantization on WEIGHTS (not activations)
  * K=256 centroids (8-bit indices), subvector dim configurable
  * codebook is a trainable Parameter -> fine-tuned on SST-2 so the model
    recovers accuracy (gradient flows through the gather, no info lost)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# INT8 weight quantization (used by PTQ + QAT)
# ----------------------------------------------------------------------------
def quantize_int8_per_channel(W):
    """Symmetric per-output-channel int8. Returns (q int8, scale fp32)."""
    max_abs = W.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)
    scale = max_abs / 127.0
    q = torch.clamp(torch.round(W / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(1)


def dequantize_int8_per_channel(q, scale):
    return q.float() * scale.unsqueeze(1)


class FakeQuantLinear(nn.Module):
    """Linear whose weight is int8 fake-quantized in the forward pass.

    Forward uses the quantized weight; backward is straight-through (grad
    flows to the full-precision master weight). This is QAT.
    """
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

    def _fake_quant(self, W):
        max_abs = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_abs / 127.0
        q = torch.clamp(torch.round(W / scale), -127, 127)
        W_hat = q * scale
        return W + (W_hat - W).detach()  # STE

    def forward(self, x):
        return F.linear(x, self._fake_quant(self.weight), self.bias)

    @torch.no_grad()
    def packed_bytes(self):
        # int8 weight + fp32 per-channel scale
        return self.out_features * self.in_features + self.out_features * 4


# ----------------------------------------------------------------------------
# Product Quantization on weights (the real contribution)
# ----------------------------------------------------------------------------
def _assign_chunked(vectors, C, chunk=200_000):
    """Nearest-centroid assignment without materializing the full [N,K] matrix."""
    out = torch.empty(vectors.shape[0], dtype=torch.long, device=vectors.device)
    for s in range(0, vectors.shape[0], chunk):
        e = min(s + chunk, vectors.shape[0])
        out[s:e] = torch.cdist(vectors[s:e], C).argmin(dim=1)
    return out


def kmeans(vectors, K, iters=25, seed=0):
    """GPU k-means, chunked to fit 4GB. vectors: [N,d] -> (C [K,d], assign [N])."""
    g = torch.Generator(device=vectors.device).manual_seed(seed)
    N = vectors.shape[0]
    K = min(K, N)
    idx = torch.randperm(N, generator=g, device=vectors.device)[:K]
    C = vectors[idx].clone()
    d = vectors.shape[1]
    for _ in range(iters):
        assign = _assign_chunked(vectors, C)
        # vectorized centroid update via scatter-add
        sums = torch.zeros(K, d, device=vectors.device)
        counts = torch.zeros(K, device=vectors.device)
        sums.index_add_(0, assign, vectors)
        counts.index_add_(0, assign, torch.ones(N, device=vectors.device))
        nonempty = counts > 0
        C[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
    return C, _assign_chunked(vectors, C)


class PQLinear(nn.Module):
    """Product-quantized Linear with a TRAINABLE codebook.

    Weight W [out, in] is split along `in` into subvectors of dim `sub_dim`.
    Each subvector is replaced by its nearest codebook entry. Indices are
    fixed (uint8, K<=256); the codebook is a Parameter fine-tuned on the task.

    Storage: indices (out * in/sub_dim bytes) + codebook (K*sub_dim*2 bytes, fp16).
    """
    def __init__(self, linear: nn.Linear, K=256, sub_dim=2, kmeans_iters=25, seed=0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        assert self.in_features % sub_dim == 0, \
            f"in_features {self.in_features} not divisible by sub_dim {sub_dim}"
        self.sub_dim = sub_dim
        self.K = K
        self.n_sub = self.in_features // sub_dim  # subvectors per output row

        W = linear.weight.detach()               # [out, in]
        vecs = W.reshape(self.out_features * self.n_sub, sub_dim)  # [out*n_sub, d]
        C, assign = kmeans(vecs, K, iters=kmeans_iters, seed=seed)

        self.codebook = nn.Parameter(C)                          # [K, d] trainable
        self.register_buffer("indices",
                             assign.reshape(self.out_features, self.n_sub).to(torch.long))
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.bias = None

    def reconstruct(self):
        # gather is differentiable w.r.t. codebook -> fine-tuning recovers accuracy
        W = self.codebook[self.indices]                # [out, n_sub, d]
        return W.reshape(self.out_features, self.in_features)

    def forward(self, x):
        return F.linear(x, self.reconstruct(), self.bias)

    @torch.no_grad()
    def packed_bytes(self):
        idx_bytes = self.out_features * self.n_sub * (1 if self.K <= 256 else 2)
        cb_bytes = self.codebook.numel() * 2       # fp16
        bias_bytes = self.out_features * 4 if self.bias is not None else 0
        return idx_bytes + cb_bytes + bias_bytes

    def bits_per_weight(self):
        bits_idx = (8 if self.K <= 256 else 16) / self.sub_dim
        return bits_idx  # codebook overhead negligible for big matrices


# ----------------------------------------------------------------------------
# model surgery: swap Linear layers
# ----------------------------------------------------------------------------
def _iter_named_linears(model, name_filter=None):
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full = f"{name}.{child_name}" if name else child_name
                if name_filter is None or name_filter(full):
                    yield module, child_name, child, full


def convert_to_fakequant(model, name_filter=None):
    """Replace Linear layers with FakeQuantLinear (QAT). Returns count."""
    n = 0
    for parent, cname, child, full in list(_iter_named_linears(model, name_filter)):
        setattr(parent, cname, FakeQuantLinear(child).to(child.weight.device))
        n += 1
    return n


def convert_to_pq(model, name_filter=None, K=256, sub_dim=2, kmeans_iters=25, seed=0):
    """Replace Linear layers with PQLinear. Returns count."""
    n = 0
    for parent, cname, child, full in list(_iter_named_linears(model, name_filter)):
        if child.in_features % sub_dim != 0:
            continue
        setattr(parent, cname, PQLinear(child, K=K, sub_dim=sub_dim,
                                        kmeans_iters=kmeans_iters, seed=seed).to(child.weight.device))
        n += 1
    return n


# ----------------------------------------------------------------------------
# honest compressed-size accounting
# ----------------------------------------------------------------------------
@torch.no_grad()
def packed_size_mb(model, embed_int8=True):
    """Real deployable size: PQ/FakeQuant linears packed, embeddings int8,
    everything else fp16."""
    total = 0
    seen = set()
    for module in model.modules():
        if isinstance(module, (PQLinear, FakeQuantLinear)):
            total += module.packed_bytes()
            for p in module.parameters(recurse=False):
                seen.add(id(p))
            if hasattr(module, "indices"):
                seen.add(id(module.indices))
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        if embed_int8 and "embeddings" in name and p.dim() == 2:
            total += p.numel() * 1 + p.shape[0] * 4   # int8 + per-row scale
        else:
            total += p.numel() * 2                     # fp16
    return total / 1e6
