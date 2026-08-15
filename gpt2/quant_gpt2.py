"""Conv1D-aware quantization for GPT-2.

GPT-2's Linear-like layers are HuggingFace `Conv1D`, weight shape [in, out]
(transposed vs nn.Linear's [out, in]), forward = x @ weight + bias.

Reuses the k-means implementation from the DistilBERT branch's quant.py
(same repo, ../qatvq/quant.py) — only the layer wrappers differ because of
the transposed weight layout.
"""
import os
import sys
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qatvq"))
from quant import kmeans  # noqa: E402  (shared k-means, GPU-chunked)


# ----------------------------------------------------------------------------
# INT8 (per-output-channel, channel = output column since weight is [in,out])
# ----------------------------------------------------------------------------
class FakeQuantConv1D(nn.Module):
    """Conv1D whose weight is int8 fake-quantized per output-channel (STE)."""
    def __init__(self, conv: Conv1D):
        super().__init__()
        self.nf = conv.nf  # out_features
        self.weight = nn.Parameter(conv.weight.detach().clone())   # [in, out]
        self.bias = nn.Parameter(conv.bias.detach().clone())

    def _fake_quant(self, W):
        # per output-channel (dim=1) scale
        max_abs = W.abs().amax(dim=0, keepdim=True).clamp(min=1e-8)
        scale = max_abs / 127.0
        q = torch.clamp(torch.round(W / scale), -127, 127)
        W_hat = q * scale
        return W + (W_hat - W).detach()

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        Wq = self._fake_quant(self.weight)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), Wq)
        return x.view(size_out)

    @torch.no_grad()
    def packed_bytes(self):
        in_f, out_f = self.weight.shape
        return in_f * out_f + out_f * 4 + out_f * 4  # int8 W + fp32 scale + fp32 bias


# ----------------------------------------------------------------------------
# Product-VQ on Conv1D weights
# ----------------------------------------------------------------------------
class PQConv1D(nn.Module):
    """Product-quantized Conv1D with a trainable codebook.

    Internally works with W transposed to [out, in] (so subvectors are taken
    along `in`, matching the PQLinear convention), then transposes back for
    the addmm forward pass.
    """
    def __init__(self, conv: Conv1D, K=256, sub_dim=2, kmeans_iters=25, seed=0):
        super().__init__()
        in_f, out_f = conv.weight.shape           # Conv1D: [in, out]
        assert in_f % sub_dim == 0, f"in_features {in_f} not divisible by sub_dim {sub_dim}"
        self.in_features, self.out_features = in_f, out_f
        self.sub_dim = sub_dim
        self.K = K
        self.n_sub = in_f // sub_dim

        Wt = conv.weight.detach().t().contiguous()          # -> [out, in]
        vecs = Wt.reshape(out_f * self.n_sub, sub_dim)
        C, assign = kmeans(vecs, K, iters=kmeans_iters, seed=seed)

        self.codebook = nn.Parameter(C)
        self.register_buffer("indices", assign.reshape(out_f, self.n_sub).to(torch.long))
        self.bias = nn.Parameter(conv.bias.detach().clone())

    def reconstruct_conv_weight(self):
        Wt = self.codebook[self.indices].reshape(self.out_features, self.in_features)
        return Wt.t()   # back to Conv1D layout [in, out]

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_features,)
        W = self.reconstruct_conv_weight()
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), W)
        return x.view(size_out)

    @torch.no_grad()
    def packed_bytes(self):
        idx_bytes = self.out_features * self.n_sub * (1 if self.K <= 256 else 2)
        cb_bytes = self.codebook.numel() * 2
        bias_bytes = self.out_features * 4
        return idx_bytes + cb_bytes + bias_bytes

    def bits_per_weight(self):
        return (8 if self.K <= 256 else 16) / self.sub_dim


# ----------------------------------------------------------------------------
# model surgery
# ----------------------------------------------------------------------------
def _iter_named_conv1d(model, name_filter=None):
    for name, module in model.named_modules():
        for cname, child in module.named_children():
            if isinstance(child, Conv1D):
                full = f"{name}.{cname}" if name else cname
                if name_filter is None or name_filter(full):
                    yield module, cname, child, full


def convert_to_fakequant_conv1d(model, name_filter=None):
    n = 0
    for parent, cname, child, full in list(_iter_named_conv1d(model, name_filter)):
        setattr(parent, cname, FakeQuantConv1D(child).to(child.weight.device))
        n += 1
    return n


def convert_to_pq_conv1d(model, name_filter=None, K=256, sub_dim=2, kmeans_iters=25, seed=0):
    n = 0
    for parent, cname, child, full in list(_iter_named_conv1d(model, name_filter)):
        if child.weight.shape[0] % sub_dim != 0:
            continue
        setattr(parent, cname, PQConv1D(child, K=K, sub_dim=sub_dim,
                                        kmeans_iters=kmeans_iters, seed=seed
                                        ).to(child.weight.device))
        n += 1
    return n


# ----------------------------------------------------------------------------
# INT8 embeddings (wte/wpe) + honest packed size
# ----------------------------------------------------------------------------
def quantize_embedding_int8(emb: nn.Embedding):
    """Return (int8 tensor, per-row fp32 scale) for an embedding weight."""
    W = emb.weight.detach()
    max_abs = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 127.0
    q = torch.clamp(torch.round(W / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(1)


@torch.no_grad()
def packed_size_mb(model):
    """Real deployable size: PQ/FakeQuant conv1d packed, wte/wpe int8, rest fp16.
    lm_head.weight is tied to wte.weight (same Parameter) -> matched by id,
    not name, so it isn't silently double- or mis-counted."""
    total = 0
    seen = set()
    embed_ids = set()
    tr = getattr(model, "transformer", model)
    if hasattr(tr, "wte"):
        embed_ids.add(id(tr.wte.weight))
    if hasattr(tr, "wpe"):
        embed_ids.add(id(tr.wpe.weight))

    for module in model.modules():
        if isinstance(module, (PQConv1D, FakeQuantConv1D)):
            total += module.packed_bytes()
            for p in module.parameters(recurse=False):
                seen.add(id(p))
            if hasattr(module, "indices"):
                seen.add(id(module.indices))

    for name, p in model.named_parameters(remove_duplicate=True):
        if id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in embed_ids:
            total += p.numel() * 1 + p.shape[0] * 4   # int8 + per-row scale
        else:
            total += p.numel() * 2                     # fp16
    return total / 1e6
