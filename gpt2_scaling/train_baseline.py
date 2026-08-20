"""Fine-tune GPT-2 (small or medium) on WikiText-103. Saves to
artifacts/<model>/baseline.pt -- separate subdirs so both sizes' checkpoints
coexist for the scaling comparison.
"""
import os
import argparse
import torch
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

from common_wt103 import (get_tokenizer, load_wikitext103, make_loaders, evaluate_ppl,
                          model_disk_size_mb, print_report, DEVICE)

ROOT_ART = os.path.join(os.path.dirname(__file__), "artifacts")


def train(model_name, epochs=1, bs=8, grad_accum=1, lr=5e-5, train_subset_chars=20_000_000):
    art = os.path.join(ROOT_ART, model_name.replace("/", "_"))
    os.makedirs(art, exist_ok=True)

    tok = get_tokenizer(model_name)
    train_blocks, val_blocks, _ = load_wikitext103(tok, train_subset_chars=train_subset_chars)
    print(f"model {model_name}: train blocks {train_blocks.shape[0]}  val blocks {val_blocks.shape[0]}")
    train_loader, val_loader = make_loaders(train_blocks, val_blocks, bs_train=bs)

    model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    opt = AdamW(model.parameters(), lr=lr)
    total_steps = (len(train_loader) // grad_accum) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        running = 0.0
        for i, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=True):
                loss = model(input_ids=ids, labels=labels).loss / grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * grad_accum
            if (i + 1) % grad_accum == 0:
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            if (i + 1) % 400 == 0:
                print(f"  ep{ep+1} step {i+1}/{len(train_loader)} loss {running/400:.4f}")
                running = 0.0
        torch.cuda.empty_cache()
        m = evaluate_ppl(model, val_loader)
        print(f"[epoch {ep+1}] val ppl {m['perplexity']:.2f}")

    model.gradient_checkpointing_disable()
    path = os.path.join(art, "baseline.pt")
    torch.save(model.state_dict(), path)
    m = evaluate_ppl(model, val_loader)
    size = model_disk_size_mb(model)
    print_report(f"Baseline ({model_name})", m, size)
    return model, m, size


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, choices=["gpt2", "gpt2-medium", "gpt2-large"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train_subset_chars", type=int, default=20_000_000)
    a = ap.parse_args()
    train(a.model, a.epochs, a.bs, a.grad_accum, a.lr, a.train_subset_chars)
