"""Fine-tune the QAT+VQ codebook for GPT-2/GPT-2-Medium on WikiText-103.

Parameterized version of ../gpt2/finetune_vq.py. On the WikiText-2 branch,
fine-tuning closed part of the ppl gap (26.44 -> 26.13) -- worth trying at
both scales here too, best-checkpoint guarded so it can only help.
"""
import os
import sys
import copy
import json
import argparse
import torch
from torch.optim import AdamW
from transformers import GPT2LMHeadModel

from common_wt103 import (get_tokenizer, load_wikitext103, make_loaders, evaluate_ppl,
                          print_report, DEVICE)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpt2"))
import quant_gpt2 as q  # noqa: E402

ROOT_ART = os.path.join(os.path.dirname(__file__), "artifacts")
is_mlp = lambda n: ".mlp.c_" in n
is_attn = lambda n: ".attn.c_" in n


def build(model_name, baseline_state, seed, K, sub_dim):
    m = GPT2LMHeadModel.from_pretrained(model_name)
    m.load_state_dict(baseline_state)
    m = m.to(DEVICE)
    q.convert_to_pq_conv1d(m, is_mlp, K=K, sub_dim=sub_dim, kmeans_iters=25, seed=seed)
    q.convert_to_fakequant_conv1d(m, is_attn)
    return m


def freeze_embeddings(model):
    for name, p in model.named_parameters():
        if name in ("transformer.wte.weight", "transformer.wpe.weight"):
            p.requires_grad_(False)


def finetune_best(model, train_loader, val_loader, epochs, lr, tag=""):
    freeze_embeddings(model)
    model.gradient_checkpointing_enable()
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_ppl = evaluate_ppl(model, val_loader)["perplexity"]
    best_state = copy.deepcopy(model.state_dict())
    print(f"  [{tag}] pre-finetune (epoch 0) val ppl {best_ppl:.2f}")

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
        ppl = evaluate_ppl(model, val_loader)["perplexity"]
        print(f"  [{tag}] epoch {ep+1} val ppl {ppl:.2f}  (best {min(best_ppl,ppl):.2f})")
        if ppl < best_ppl:
            best_ppl = ppl
            best_state = copy.deepcopy(model.state_dict())
    model.gradient_checkpointing_disable()
    model.load_state_dict(best_state)
    return model, best_ppl


def main(model_name, seed, K, sub_dim, epochs, lr, bs_train, bs_eval, train_subset_chars):
    art = os.path.join(ROOT_ART, model_name.replace("/", "_"))
    tok = get_tokenizer(model_name)
    train_blocks, val_blocks, _ = load_wikitext103(tok, train_subset_chars=train_subset_chars)
    train_loader, val_loader = make_loaders(train_blocks, val_blocks, bs_train=bs_train, bs_eval=bs_eval)
    baseline_state = torch.load(os.path.join(art, "baseline.pt"), map_location="cpu")

    model = build(model_name, baseline_state, seed, K, sub_dim)
    model, best_ppl = finetune_best(model, train_loader, val_loader, epochs, lr, tag="QAT+VQ*")

    metrics = evaluate_ppl(model, val_loader)
    size = q.packed_size_mb(model)
    print_report(f"QAT+VQ ({model_name}, fine-tuned, seed {seed})", metrics, size)
    torch.save(model.state_dict(), os.path.join(art, "qatvq.pt"))

    path = os.path.join(art, "results.json")
    with open(path) as f:
        data = json.load(f)
    data["results"]["QAT+VQ"] = {
        "perplexity": metrics["perplexity"], "avg_loss": metrics["avg_loss"],
        "size_mb": size, "latency_ms_per_batch": metrics["latency_ms_per_batch"],
    }
    data["config"]["vq_final"] = {"seed": seed, "K": K, "sub_dim": sub_dim,
                                  "finetune_epochs": epochs, "finetune_lr": lr}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    ptq = data["results"]["PTQ"]
    print("\n========= FINAL VERDICT =========")
    print(f"Baseline: ppl {data['results']['Baseline']['perplexity']:.2f}  "
          f"{data['results']['Baseline']['size_mb']:.1f}MB")
    print(f"PTQ     : ppl {ptq['perplexity']:.2f}  {ptq['size_mb']:.1f}MB")
    print(f"QAT+VQ  : ppl {metrics['perplexity']:.2f}  {size:.1f}MB")
    win = metrics["perplexity"] <= ptq["perplexity"] and size < ptq["size_mb"]
    gap_pct = (metrics["perplexity"] - ptq["perplexity"]) / ptq["perplexity"] * 100
    size_pct = (ptq["size_mb"] - size) / ptq["size_mb"] * 100
    print("RESULT:", "HYBRID BEATS OR MATCHES PTQ" if win else
          f"Pareto point: {size_pct:.0f}% smaller than PTQ, {gap_pct:+.1f}% ppl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, choices=["gpt2", "gpt2-medium", "gpt2-large"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--sub_dim", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--bs_train", type=int, default=4)
    ap.add_argument("--bs_eval", type=int, default=4)
    ap.add_argument("--train_subset_chars", type=int, default=20_000_000)
    a = ap.parse_args()
    main(a.model, a.seed, a.K, a.sub_dim, a.epochs, a.lr, a.bs_train, a.bs_eval, a.train_subset_chars)
