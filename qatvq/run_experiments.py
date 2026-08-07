"""Run every compression variant, evaluate, dump metrics + curve data.

Variants:
  baseline    fine-tuned DistilBERT (fp32/fp16)
  ptq         int8 weights per-channel, NO retraining (simulated)
  qat         int8 fake-quant weights, fine-tuned (STE)
  qatvq       IMPROVED hybrid: product-quant encoder weights (K=256) +
              trainable codebook fine-tune + int8 embeddings
Writes artifacts/results.json
"""
import os
import copy
import json
import argparse
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification

from common import (get_tokenizer, load_sst2, make_loaders, evaluate,
                    model_disk_size_mb, print_report, MODEL_NAME, DEVICE)
import quant

ART = os.path.join(os.path.dirname(__file__), "artifacts")
ENCODER = lambda n: "transformer.layer" in n   # encoder linears only
FT_SUBSET = 20000                               # recovery fine-tune subset (speed)


def load_baseline():
    m = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    m.load_state_dict(torch.load(os.path.join(ART, "baseline.pt"), map_location="cpu"))
    return m.to(DEVICE)


def freeze_embeddings(model):
    for name, p in model.named_parameters():
        if "embeddings" in name:
            p.requires_grad_(False)


def finetune(model, train_loader, val_loader, epochs=1, lr=1e-5, tag=""):
    freeze_embeddings(model)  # saves VRAM + embeddings kept int8 anyway
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    for ep in range(epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=True):
                out = model(input_ids=ids, attention_mask=attn, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if (i + 1) % 400 == 0:
                print(f"  [{tag}] ep{ep+1} step {i+1}/{len(train_loader)} loss {loss.item():.4f}")
        m = evaluate(model, val_loader)
        print(f"  [{tag}] epoch {ep+1} val acc {m['accuracy']*100:.2f}%")
    return model


@torch.no_grad()
def apply_ptq_int8(model):
    """Simulated per-channel int8 PTQ: quantize+dequantize encoder linear weights."""
    for parent, cname, child, full in list(quant._iter_named_linears(model, ENCODER)):
        q, scale = quant.quantize_int8_per_channel(child.weight.data)
        child.weight.data = quant.dequantize_int8_per_channel(q, scale)
    return model


def ptq_packed_mb(model):
    """int8 encoder linears + int8 embeddings + fp16 rest."""
    total = 0
    enc_param_ids = set()
    for parent, cname, child, full in quant._iter_named_linears(model, ENCODER):
        total += child.out_features * child.in_features + child.out_features * 4
        enc_param_ids.add(id(child.weight))
        if child.bias is not None:
            total += child.out_features * 4
            enc_param_ids.add(id(child.bias))
    for name, p in model.named_parameters():
        if id(p) in enc_param_ids:
            continue
        if "embeddings" in name and p.dim() == 2:
            total += p.numel() + p.shape[0] * 4
        else:
            total += p.numel() * 2
    return total / 1e6


def record(results, name, metrics, size_mb):
    print_report(name, metrics, size_mb)
    results[name] = {
        "accuracy": metrics["accuracy"], "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"], "size_mb": size_mb,
        "tp": metrics["tp"], "fn": metrics["fn"],
        "tn": metrics["tn"], "fp": metrics["fp"],
        "latency_ms_per_batch": metrics["latency_ms_per_batch"],
        "probs": metrics["_probs"].tolist(), "labels": metrics["_labels"].tolist(),
    }


def main(sub_dim, K, qat_epochs, vq_epochs, vq_lr):
    tok = get_tokenizer()
    train_ds, val_ds = load_sst2(tok, train_subset=FT_SUBSET)
    train_loader, val_loader = make_loaders(train_ds, val_ds, bs_train=16)
    print(f"fine-tune subset: {len(train_ds)} samples, val: {len(val_ds)}")
    results = {}

    # 1. baseline -------------------------------------------------------------
    base = load_baseline()
    m = evaluate(base, val_loader)
    record(results, "Baseline", m, model_disk_size_mb(base))

    # 2. PTQ ------------------------------------------------------------------
    ptq_model = apply_ptq_int8(copy.deepcopy(base))
    m = evaluate(ptq_model, val_loader)
    record(results, "PTQ", m, ptq_packed_mb(ptq_model))
    del ptq_model; torch.cuda.empty_cache()

    # 3. QAT ------------------------------------------------------------------
    qat_model = copy.deepcopy(base)
    n = quant.convert_to_fakequant(qat_model, ENCODER)
    print(f"QAT: converted {n} linears to fake-quant int8")
    finetune(qat_model, train_loader, val_loader, epochs=qat_epochs, lr=1e-5, tag="QAT")
    m = evaluate(qat_model, val_loader)
    record(results, "QAT-INT8", m, quant.packed_size_mb(qat_model))
    del qat_model; torch.cuda.empty_cache()

    # 4. QAT+VQ (improved hybrid) --------------------------------------------
    vq_model = copy.deepcopy(base)
    n = quant.convert_to_pq(vq_model, ENCODER, K=K, sub_dim=sub_dim, kmeans_iters=25)
    bpw = None
    for mod in vq_model.modules():
        if isinstance(mod, quant.PQLinear):
            bpw = mod.bits_per_weight(); break
    print(f"QAT+VQ: converted {n} linears to PQ (K={K}, sub_dim={sub_dim}, "
          f"~{bpw:.1f} bits/weight)")
    m0 = evaluate(vq_model, val_loader)
    print(f"  QAT+VQ pre-finetune acc {m0['accuracy']*100:.2f}%")
    finetune(vq_model, train_loader, val_loader, epochs=vq_epochs, lr=vq_lr, tag="QAT+VQ")
    m = evaluate(vq_model, val_loader)
    record(results, "QAT+VQ", m, quant.packed_size_mb(vq_model))
    torch.save(vq_model.state_dict(), os.path.join(ART, "qatvq.pt"))
    del vq_model; torch.cuda.empty_cache()

    with open(os.path.join(ART, "results.json"), "w") as f:
        json.dump({"config": {"sub_dim": sub_dim, "K": K}, "results": results}, f, indent=2)
    print("\nSaved results.json")

    # verdict
    r = results
    print("\n================ VERDICT ================")
    print(f"PTQ    : acc {r['PTQ']['accuracy']*100:.2f}%  size {r['PTQ']['size_mb']:.1f}MB")
    print(f"QAT+VQ : acc {r['QAT+VQ']['accuracy']*100:.2f}%  size {r['QAT+VQ']['size_mb']:.1f}MB")
    win = (r['QAT+VQ']['accuracy'] >= r['PTQ']['accuracy'] - 0.005 and
           r['QAT+VQ']['size_mb'] < r['PTQ']['size_mb'])
    print("RESULT:", "HYBRID WINS (>=PTQ acc, smaller)" if win else "not yet - tune sub_dim/epochs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub_dim", type=int, default=2)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--qat_epochs", type=int, default=1)
    ap.add_argument("--vq_epochs", type=int, default=2)
    ap.add_argument("--vq_lr", type=float, default=2e-5)
    a = ap.parse_args()
    main(a.sub_dim, a.K, a.qat_epochs, a.vq_epochs, a.vq_lr)
