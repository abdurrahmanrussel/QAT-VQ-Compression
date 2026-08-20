"""Run every compression variant for GPT-2 (small or medium) on WikiText-103.

Same recipe as ../gpt2/run_experiments.py, parameterized by --model so the
identical pipeline runs at both sizes -> a clean scaling comparison (same
dataset, same method, only model size differs).
Writes artifacts/<model>/results.json
"""
import os
import sys
import copy
import json
import argparse
import torch
from transformers import GPT2LMHeadModel

from common_wt103 import (get_tokenizer, load_wikitext103, make_loaders, evaluate_ppl,
                          model_disk_size_mb, print_report, DEVICE)
from train_baseline import make_optimizer  # 8-bit AdamW when available, see its docstring

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpt2"))
import quant_gpt2 as q  # noqa: E402  (Conv1D int8 + Product-VQ, reused as-is)

ROOT_ART = os.path.join(os.path.dirname(__file__), "artifacts")
is_mlp = lambda n: ".mlp.c_" in n
is_attn = lambda n: ".attn.c_" in n
is_any_conv = lambda n: is_mlp(n) or is_attn(n)


def load_baseline(model_name, art):
    m = GPT2LMHeadModel.from_pretrained(model_name)
    m.load_state_dict(torch.load(os.path.join(art, "baseline.pt"), map_location="cpu"))
    return m.to(DEVICE)


def freeze_embeddings(model):
    for name, p in model.named_parameters():
        if name in ("transformer.wte.weight", "transformer.wpe.weight"):
            p.requires_grad_(False)


def finetune(model, train_loader, val_loader, epochs, lr, tag=""):
    freeze_embeddings(model)
    model.gradient_checkpointing_enable()
    opt = make_optimizer([p for p in model.parameters() if p.requires_grad], lr)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    for ep in range(epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=True):
                loss = model(input_ids=ids, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if (i + 1) % 400 == 0:
                print(f"  [{tag}] ep{ep+1} step {i+1}/{len(train_loader)} loss {loss.item():.4f}")
        torch.cuda.empty_cache()
        m = evaluate_ppl(model, val_loader)
        print(f"  [{tag}] epoch {ep+1} val ppl {m['perplexity']:.2f}")
    model.gradient_checkpointing_disable()
    return model


@torch.no_grad()
def apply_ptq_int8(model):
    for parent, cname, child, full in list(q._iter_named_conv1d(model, is_any_conv)):
        W = child.weight.data  # [in, out]
        max_abs = W.abs().amax(dim=0, keepdim=True).clamp(min=1e-8)
        scale = max_abs / 127.0
        qw = torch.clamp(torch.round(W / scale), -127, 127)
        child.weight.data = qw * scale
    return model


def ptq_packed_mb(model):
    total = 0
    conv_ids = set()
    for parent, cname, child, full in q._iter_named_conv1d(model, is_any_conv):
        in_f, out_f = child.weight.shape
        total += in_f * out_f + out_f * 4 + out_f * 4
        conv_ids.add(id(child.weight)); conv_ids.add(id(child.bias))
    embed_ids = {id(model.transformer.wte.weight), id(model.transformer.wpe.weight)}
    for name, p in model.named_parameters(remove_duplicate=True):
        if id(p) in conv_ids:
            continue
        if id(p) in embed_ids:
            total += p.numel() + p.shape[0] * 4
        else:
            total += p.numel() * 2
    return total / 1e6


def record(results, name, metrics, size_mb):
    print_report(name, metrics, size_mb)
    results[name] = {"perplexity": metrics["perplexity"], "avg_loss": metrics["avg_loss"],
                     "size_mb": size_mb, "latency_ms_per_batch": metrics["latency_ms_per_batch"]}


def main(model_name, sub_dim, K, qat_epochs, ft_lr, seeds, bs_train, bs_eval, train_subset_chars):
    art = os.path.join(ROOT_ART, model_name.replace("/", "_"))
    tok = get_tokenizer(model_name)
    train_blocks, val_blocks, _ = load_wikitext103(tok, train_subset_chars=train_subset_chars)
    train_loader, val_loader = make_loaders(train_blocks, val_blocks, bs_train=bs_train, bs_eval=bs_eval)
    results = {}

    # 1. baseline
    base = load_baseline(model_name, art)
    m = evaluate_ppl(base, val_loader)
    record(results, "Baseline", m, model_disk_size_mb(base))

    # 2. PTQ
    ptq_model = apply_ptq_int8(copy.deepcopy(base))
    m = evaluate_ppl(ptq_model, val_loader)
    record(results, "PTQ", m, ptq_packed_mb(ptq_model))
    del ptq_model; torch.cuda.empty_cache()

    # 3. QAT
    qat_model = copy.deepcopy(base)
    n = q.convert_to_fakequant_conv1d(qat_model, is_any_conv)
    print(f"QAT: converted {n} Conv1D to fake-quant int8")
    finetune(qat_model, train_loader, val_loader, epochs=qat_epochs, lr=ft_lr, tag="QAT")
    m = evaluate_ppl(qat_model, val_loader)
    record(results, "QAT-INT8", m, q.packed_size_mb(qat_model))
    del qat_model; torch.cuda.empty_cache()

    # 4. QAT+VQ: MLP -> PQ (best of seeds, no fine-tune), attention -> int8
    best = None
    for s in seeds:
        vq_model = copy.deepcopy(base)
        q.convert_to_pq_conv1d(vq_model, is_mlp, K=K, sub_dim=sub_dim, kmeans_iters=25, seed=s)
        q.convert_to_fakequant_conv1d(vq_model, is_attn)
        ppl = evaluate_ppl(vq_model, val_loader)["perplexity"]
        print(f"  QAT+VQ seed {s}: val ppl {ppl:.2f}")
        if best is None or ppl < best[1]:          # LOWER perplexity is better
            best = (s, ppl, vq_model)
        else:
            del vq_model
        torch.cuda.empty_cache()
    s, _, vq_model = best
    m = evaluate_ppl(vq_model, val_loader)
    size = q.packed_size_mb(vq_model)
    record(results, "QAT+VQ", m, size)
    torch.save(vq_model.state_dict(), os.path.join(art, "qatvq.pt"))

    with open(os.path.join(art, "results.json"), "w") as f:
        json.dump({"model": model_name, "config": {"sub_dim": sub_dim, "K": K, "seed": s},
                   "results": results}, f, indent=2)
    print(f"\nSaved {art}/results.json")

    print("\n================ VERDICT ================")
    print(f"Baseline: ppl {results['Baseline']['perplexity']:.2f}  {results['Baseline']['size_mb']:.1f}MB")
    print(f"PTQ     : ppl {results['PTQ']['perplexity']:.2f}  {results['PTQ']['size_mb']:.1f}MB")
    print(f"QAT+VQ  : ppl {results['QAT+VQ']['perplexity']:.2f}  {results['QAT+VQ']['size_mb']:.1f}MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, choices=["gpt2", "gpt2-medium", "gpt2-large"])
    ap.add_argument("--sub_dim", type=int, default=2)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--qat_epochs", type=int, default=1)
    ap.add_argument("--ft_lr", type=float, default=1e-5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bs_train", type=int, default=8)
    ap.add_argument("--bs_eval", type=int, default=8)
    ap.add_argument("--train_subset_chars", type=int, default=20_000_000)
    a = ap.parse_args()
    main(a.model, a.sub_dim, a.K, a.qat_epochs, a.ft_lr, a.seeds,
         a.bs_train, a.bs_eval, a.train_subset_chars)
