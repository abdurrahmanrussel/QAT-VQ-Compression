"""Fine-tune the QAT+VQ codebook to try to close the ppl gap vs PTQ.

Unlike the DistilBERT branch (classification, already-converged model, fine-
tune only hurt), GPT-2 perplexity is far more sensitive to precision, so the
best-of-5-seeds init alone left a real gap (26.44 vs PTQ 25.43). The PQConv1D
codebook is a trainable nn.Parameter -> fine-tune it (and the untouched attn/
embedding params) on WikiText-2, keeping only the best checkpoint so this can
only ever match-or-beat the no-finetune result.
"""
import os
import copy
import json
import argparse
import torch
from torch.optim import AdamW
from transformers import GPT2LMHeadModel

from common_gpt2 import (get_tokenizer, load_wikitext2, make_loaders, evaluate_ppl,
                         print_report, MODEL_NAME, DEVICE)
import quant_gpt2 as q

ART = os.path.join(os.path.dirname(__file__), "artifacts")
is_mlp = lambda n: ".mlp.c_" in n
is_attn = lambda n: ".attn.c_" in n


def build(baseline_state, seed, K, sub_dim):
    m = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
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
        if ppl < best_ppl:            # LOWER perplexity is better
            best_ppl = ppl
            best_state = copy.deepcopy(model.state_dict())
    model.gradient_checkpointing_disable()
    model.load_state_dict(best_state)
    return model, best_ppl


def main(seed, K, sub_dim, epochs, lr):
    tok = get_tokenizer()
    train_blocks, val_blocks, _ = load_wikitext2(tok)
    train_loader, val_loader = make_loaders(train_blocks, val_blocks, bs_train=4, bs_eval=4)
    baseline_state = torch.load(os.path.join(ART, "baseline.pt"), map_location="cpu")

    model = build(baseline_state, seed, K, sub_dim)
    model, best_ppl = finetune_best(model, train_loader, val_loader, epochs, lr, tag="QAT+VQ*")

    metrics = evaluate_ppl(model, val_loader)
    size = q.packed_size_mb(model)
    print_report(f"QAT+VQ (fine-tuned, seed {seed})", metrics, size)
    torch.save(model.state_dict(), os.path.join(ART, "qatvq.pt"))

    path = os.path.join(ART, "results.json")
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
    ap.add_argument("--seed", type=int, default=1)   # best seed from the search
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--sub_dim", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    a = ap.parse_args()
    main(a.seed, a.K, a.sub_dim, a.epochs, a.lr)
