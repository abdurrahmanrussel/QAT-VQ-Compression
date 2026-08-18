"""GPTQ and AWQ for GPT-2's Conv1D layers.

Reuses the core math from ../qatvq/gptq_awq.py unmodified (gptq_quantize
and awq_search_scale both operate on W[out,in] + calibration activations
X[n,in], layout-agnostic) -- only the layer wrapper differs, because
Conv1D's weight is stored transposed ([in,out]) vs nn.Linear's ([out,in]),
same adaptation quant_gpt2.py made for the int8/Product-VQ layers.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qatvq"))
from gptq_awq import gptq_quantize, awq_search_scale, _quantize_row  # noqa: E402


@torch.no_grad()
def collect_layer_calibration(model, layer, loader, device, n_batches=16, max_rows=4096):
    """Same idea as qatvq/gptq_awq.py's version, adapted for the GPT-2
    causal-LM batch format (input_ids/labels instead of input_ids/attention_mask)."""
    acts = []

    def hook(module, inp, out):
        acts.append(inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu())

    handle = layer.register_forward_hook(hook)
    model.eval()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        ids = batch["input_ids"].to(device)
        model(input_ids=ids)
    handle.remove()

    X = torch.cat(acts, dim=0)
    if X.shape[0] > max_rows:
        idx = torch.randperm(X.shape[0])[:max_rows]
        X = X[idx]
    return X


class GPTQConv1D(nn.Module):
    def __init__(self, conv: Conv1D, W_hat_t, scale):
        """W_hat_t, scale computed in [out,in] orientation; stored back in
        Conv1D's native [in,out] layout for the addmm forward pass."""
        super().__init__()
        self.nf = conv.nf
        q = torch.clamp(torch.round(W_hat_t / scale.unsqueeze(1)), -127, 127).to(torch.int8)
        self.register_buffer("weight_q", q.t().contiguous())   # [in, out]
        self.register_buffer("scale", scale)                    # per out-channel
        self.bias = nn.Parameter(conv.bias.detach().clone())

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        W = self.weight_q.float() * self.scale.unsqueeze(0)     # broadcast over out dim
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), W)
        return x.view(size_out)

    def packed_bytes(self):
        return self.weight_q.numel() + self.scale.numel() * 4 + self.bias.numel() * 4


class AWQConv1D(nn.Module):
    def __init__(self, conv: Conv1D, s):
        """s computed in [out,in]-oriented in_features space, applied to the
        Conv1D weight's in-dimension (dim 0 in its native [in,out] layout)."""
        super().__init__()
        self.nf = conv.nf
        Wt = conv.weight.detach().t().contiguous()               # [out, in]
        Ws = Wt * s.unsqueeze(0)
        scale = torch.stack([_quantize_row(Ws[r])[1] for r in range(Ws.shape[0])])
        q = torch.clamp(torch.round(Ws / scale.unsqueeze(1)), -127, 127).to(torch.int8)
        self.register_buffer("weight_q", q.t().contiguous())    # [in, out]
        self.register_buffer("scale", scale)
        self.register_buffer("inv_s", (1.0 / s))                 # [in]
        self.bias = nn.Parameter(conv.bias.detach().clone())

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        W = self.weight_q.float() * self.scale.unsqueeze(0)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)) * self.inv_s, W)
        return x.view(size_out)

    def packed_bytes(self):
        return self.weight_q.numel() + self.scale.numel() * 4 + \
               self.inv_s.numel() * 4 + self.bias.numel() * 4


def _iter_named_conv1d(model, name_filter=None):
    for name, module in model.named_modules():
        for cname, child in module.named_children():
            if isinstance(child, Conv1D):
                full = f"{name}.{cname}" if name else cname
                if name_filter is None or name_filter(full):
                    yield module, cname, child, full


def apply_gptq(model, loader, device, name_filter=None, n_bits=8, n_batches=16):
    for parent, cname, child, full in list(_iter_named_conv1d(model, name_filter)):
        X = collect_layer_calibration(model, child, loader, device, n_batches=n_batches).to(device)
        Wt = child.weight.detach().t().contiguous()   # [out, in]
        W_hat, scale = gptq_quantize(Wt, X, n_bits=n_bits)
        setattr(parent, cname, GPTQConv1D(child, W_hat, scale).to(device))
        del X
        torch.cuda.empty_cache()
    return model


def apply_awq(model, loader, device, name_filter=None, n_batches=16):
    for parent, cname, child, full in list(_iter_named_conv1d(model, name_filter)):
        X = collect_layer_calibration(model, child, loader, device, n_batches=n_batches).to(device)
        Wt = child.weight.detach().t().contiguous()
        s = awq_search_scale(Wt, X)
        setattr(parent, cname, AWQConv1D(child, s).to(device))
        del X
        torch.cuda.empty_cache()
    return model


@torch.no_grad()
def packed_size_mb(model, embed_int8=True):
    total = 0
    seen = set()
    for module in model.modules():
        if isinstance(module, (GPTQConv1D, AWQConv1D)):
            total += module.packed_bytes()
            for p in module.parameters(recurse=False):
                seen.add(id(p))
            for b in module.buffers(recurse=False):
                seen.add(id(b))
    embed_ids = set()
    if hasattr(model, "transformer"):
        embed_ids = {id(model.transformer.wte.weight), id(model.transformer.wpe.weight)}
    for name, p in model.named_parameters(remove_duplicate=True):
        if id(p) in seen:
            continue
        if id(p) in embed_ids:
            total += p.numel() * 1 + p.shape[0] * 4
        else:
            total += p.numel() * 2
    return total / 1e6
