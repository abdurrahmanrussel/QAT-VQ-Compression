"""Fine-tune gpt2 on WikiText-2. Saves to artifacts/baseline.pt.

4GB GPU budget: seq 256, bs 4, grad accumulation to simulate bs 16.
"""
import os
import argparse
import torch
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

from common_gpt2 import (get_tokenizer, load_wikitext2, make_loaders, evaluate_ppl,
                         model_disk_size_mb, print_report, MODEL_NAME, DEVICE)

ART = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ART, exist_ok=True)


def train(epochs=1, bs=4, grad_accum=4, lr=5e-5, amp=True):
    tok = get_tokenizer()
    train_blocks, val_blocks, _ = load_wikitext2(tok)
    print(f"train blocks {train_blocks.shape[0]}  val blocks {val_blocks.shape[0]}")
    train_loader, val_loader = make_loaders(train_blocks, val_blocks, bs_train=bs)

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.gradient_checkpointing_enable()  # save VRAM on 4GB card

    opt = AdamW(model.parameters(), lr=lr)
    steps_per_epoch = len(train_loader) // grad_accum
    total = steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    for ep in range(epochs):
        model.train()
        running = 0.0
        opt.zero_grad()
        for i, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = model(input_ids=ids, labels=labels).loss / grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * grad_accum
            if (i + 1) % grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
            if (i + 1) % (grad_accum * 100) == 0:
                print(f"  ep{ep+1} step {i+1}/{len(train_loader)} loss {running/(grad_accum*100):.4f}")
                running = 0.0
        torch.cuda.empty_cache()
        m = evaluate_ppl(model, val_loader)
        print(f"[epoch {ep+1}] val ppl {m['perplexity']:.2f}")

    model.gradient_checkpointing_disable()
    path = os.path.join(ART, "baseline.pt")
    torch.save(model.state_dict(), path)
    m = evaluate_ppl(model, val_loader)
    size = model_disk_size_mb(model)
    print_report("Baseline (fp32)", m, size)
    return model, m, size


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()
    train(a.epochs, a.bs, a.grad_accum, a.lr)
