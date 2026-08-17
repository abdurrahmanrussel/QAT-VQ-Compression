"""GPTQ and AWQ: real post-training quantization baselines, implemented from
scratch (same philosophy as the rest of this thesis's quant.py) rather than
pulled from auto-gptq/AutoAWQ, whose calibration pipelines are built around
causal-LM architectures and don't cleanly support DistilBERT's encoder-only
BertForSequenceClassification.

Both operate on nn.Linear weights [out, in] using calibration activations
collected by a forward hook, and are used purely as evaluation baselines
(no retraining) against which QAT-VQ is compared.

References:
  GPTQ: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
        Generative Pre-trained Transformers", ICLR 2023.
  AWQ:  Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
        Compression and Acceleration", MLSys 2024.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# calibration: collect ONE layer's input activations at a time
# ----------------------------------------------------------------------------
@torch.no_grad()
def collect_layer_calibration(model, layer, loader, device, n_batches=16, max_rows=4096):
    """Hook a single Linear layer, run calibration batches, return its
    stacked input activations [n, in_features]. Layer-at-a-time (not all
    layers hooked simultaneously) keeps peak memory to one layer's worth of
    activations regardless of model depth -- this is also how the reference
    GPTQ implementation calibrates, one layer at a time."""
    acts = []

    def hook(module, inp, out):
        acts.append(inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu())

    handle = layer.register_forward_hook(hook)
    model.eval()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        model(input_ids=input_ids, attention_mask=attn)
    handle.remove()

    X = torch.cat(acts, dim=0)
    if X.shape[0] > max_rows:
        idx = torch.randperm(X.shape[0])[:max_rows]
        X = X[idx]
    return X


# ----------------------------------------------------------------------------
# shared int8 quantizer (per-output-channel, same convention as quant.py PTQ)
# ----------------------------------------------------------------------------
def _quantize_row(w_row, n_bits=8):
    qmax = 2 ** (n_bits - 1) - 1
    scale = w_row.abs().max().clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(w_row / scale), -qmax, qmax)
    return q, scale


# ----------------------------------------------------------------------------
# GPTQ: Hessian-guided sequential quantization with error compensation
# ----------------------------------------------------------------------------
def gptq_quantize(W, X, n_bits=8, damp=0.01, blocksize=128):
    """W: [out, in] fp32 weight. X: [n_samples, in] calibration activations.
    Returns (W_hat [out,in] fp32-dequantized, scale [out]) using the GPTQ
    algorithm: build the layer Hessian from calibration data, then quantize
    input channels left-to-right, propagating each channel's rounding error
    to not-yet-quantized channels via the Hessian inverse (Optimal Brain
    Quantization-style correction) -- this is what makes GPTQ beat naive
    round-to-nearest PTQ at the same bit-width.
    """
    device = W.device
    out_f, in_f = W.shape
    W = W.clone().float()

    # 1. Hessian of the layer's quadratic reconstruction loss
    H = (2.0 / X.shape[0]) * (X.t() @ X).to(device)
    diag_mean = H.diagonal().mean()
    H += damp * diag_mean * torch.eye(in_f, device=device)

    # 2. Cholesky of H^-1 (upper), the standard GPTQ formulation
    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    # 3. per-output-channel scale fixed up front from the ORIGINAL weight
    #    (matches standard practice: quantization grid doesn't change as we
    #    go, only which value on the grid is picked, guided by error feedback)
    scale = torch.stack([_quantize_row(W[r], n_bits)[1] for r in range(out_f)])
    qmax = 2 ** (n_bits - 1) - 1

    Q = torch.zeros_like(W)
    for c0 in range(0, in_f, blocksize):
        c1 = min(c0 + blocksize, in_f)
        Wblk = W[:, c0:c1].clone()
        Qblk = torch.zeros_like(Wblk)
        Hinv_blk = Hinv[c0:c1, c0:c1]
        err_acc = torch.zeros_like(Wblk)

        for i in range(c1 - c0):
            w = Wblk[:, i]
            d = Hinv_blk[i, i]
            q = torch.clamp(torch.round(w / scale), -qmax, qmax)
            q_deq = q * scale
            Qblk[:, i] = q_deq
            err = (w - q_deq) / d
            if i < c1 - c0 - 1:
                Wblk[:, i + 1:] -= err.unsqueeze(1) * Hinv_blk[i, i + 1:].unsqueeze(0)
            err_acc[:, i] = err

        Q[:, c0:c1] = Qblk
        if c1 < in_f:
            # propagate this block's error to all remaining (later) blocks
            W[:, c1:] -= err_acc @ Hinv[c0:c1, c1:]

    return Q, scale


class GPTQLinear(nn.Module):
    """Drop-in replacement holding a GPTQ-quantized weight (already computed).
    Storage format matches PTQ (int8 + per-channel fp32 scale) for a fair
    size comparison."""
    def __init__(self, linear: nn.Linear, W_hat, scale):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        q = torch.clamp(torch.round(W_hat / scale.unsqueeze(1)), -127, 127).to(torch.int8)
        self.register_buffer("weight_q", q)
        self.register_buffer("scale", scale)
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

    def forward(self, x):
        W = self.weight_q.float() * self.scale.unsqueeze(1)
        return F.linear(x, W, self.bias)

    def packed_bytes(self):
        return self.weight_q.numel() + self.scale.numel() * 4 + \
               (self.bias.numel() * 4 if self.bias is not None else 0)


# ----------------------------------------------------------------------------
# AWQ: protect salient (high-activation) input channels before quantizing
# ----------------------------------------------------------------------------
def awq_search_scale(W, X, n_grid=20, n_calib_rows=512):
    """Search alpha in [0,1] for per-input-channel scale s = act_mag^alpha
    minimizing OUTPUT reconstruction error on calibration data -- i.e. the
    actual AWQ objective (Lin et al., eq. for the grid search), not raw
    weight MSE. Raw weight MSE is dominated by whichever row has the
    largest single value and barely responds to channel scaling; the real
    benefit only shows up once you weight by how much each channel actually
    gets used (X), which is the whole point of "activation-aware".
    Returns s [in_features].
    """
    if X.shape[0] > n_calib_rows:
        idx = torch.randperm(X.shape[0])[:n_calib_rows]
        X = X[idx]
    act_mag = X.abs().mean(dim=0).clamp(min=1e-5)   # [in]
    Y = X @ W.t()                                    # true output, calibration data

    best_s, best_err = None, float("inf")
    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = act_mag.pow(alpha)
        s = s / s.mean()                             # keep weight scale stable
        Ws = W * s.unsqueeze(0)
        scale = torch.stack([_quantize_row(Ws[r])[1] for r in range(Ws.shape[0])])
        q = torch.clamp(torch.round(Ws / scale.unsqueeze(1)), -127, 127)
        Wq_hat = (q * scale.unsqueeze(1)) / s.unsqueeze(0)   # dequant + undo scale
        Yq = X @ Wq_hat.t()
        err = ((Y - Yq) ** 2).mean().item()
        if err < best_err:
            best_err, best_s = err, s
    return best_s


class AWQLinear(nn.Module):
    """y = (x / s) @ (W*s)^T_int8 -- mathematically identical to the
    original layer pre-quantization; the scale only changes which values
    land where on the int8 grid, protecting salient input channels."""
    def __init__(self, linear: nn.Linear, s):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        Ws = linear.weight.detach() * s.unsqueeze(0)
        scale = torch.stack([_quantize_row(Ws[r])[1] for r in range(Ws.shape[0])])
        q = torch.clamp(torch.round(Ws / scale.unsqueeze(1)), -127, 127).to(torch.int8)
        self.register_buffer("weight_q", q)
        self.register_buffer("scale", scale)
        self.register_buffer("inv_s", (1.0 / s))
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

    def forward(self, x):
        W = self.weight_q.float() * self.scale.unsqueeze(1)
        return F.linear(x * self.inv_s, W, self.bias)

    def packed_bytes(self):
        return self.weight_q.numel() + self.scale.numel() * 4 + self.inv_s.numel() * 4 + \
               (self.bias.numel() * 4 if self.bias is not None else 0)


# ----------------------------------------------------------------------------
# model surgery helpers
# ----------------------------------------------------------------------------
def _iter_named_linears(model, name_filter=None):
    for name, module in model.named_modules():
        for cname, child in module.named_children():
            if isinstance(child, nn.Linear):
                full = f"{name}.{cname}" if name else cname
                if name_filter is None or name_filter(full):
                    yield module, cname, child, full


def apply_gptq(model, loader, device, name_filter=None, n_bits=8, n_batches=16):
    for parent, cname, child, full in list(_iter_named_linears(model, name_filter)):
        X = collect_layer_calibration(model, child, loader, device, n_batches=n_batches).to(device)
        W_hat, scale = gptq_quantize(child.weight.detach(), X, n_bits=n_bits)
        setattr(parent, cname, GPTQLinear(child, W_hat, scale).to(device))
        del X
        torch.cuda.empty_cache()
    return model


def apply_awq(model, loader, device, name_filter=None, n_batches=16):
    for parent, cname, child, full in list(_iter_named_linears(model, name_filter)):
        X = collect_layer_calibration(model, child, loader, device, n_batches=n_batches).to(device)
        s = awq_search_scale(child.weight.detach(), X)
        setattr(parent, cname, AWQLinear(child, s).to(device))
        del X
        torch.cuda.empty_cache()
    return model


@torch.no_grad()
def packed_size_mb(model, embed_int8=True):
    total = 0
    seen = set()
    for module in model.modules():
        if isinstance(module, (GPTQLinear, AWQLinear)):
            total += module.packed_bytes()
            for p in module.parameters(recurse=False):
                seen.add(id(p))
            for b in module.buffers(recurse=False):
                seen.add(id(b))
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        if embed_int8 and "embeddings" in name and p.dim() == 2:
            total += p.numel() * 1 + p.shape[0] * 4
        else:
            total += p.numel() * 2
    return total / 1e6
