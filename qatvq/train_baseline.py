"""Fine-tune DistilBERT baseline on SST-2. Saves to artifacts/baseline.pt.

Fits GTX 1650 (4GB): batch 16, seq 128, fp32. AMP optional.
"""
import os
import argparse
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from common import (get_tokenizer, load_sst2, make_loaders, evaluate,
                    model_disk_size_mb, print_report, MODEL_NAME, DEVICE)

ART = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ART, exist_ok=True)


def train(epochs=2, bs=16, lr=2e-5, subset=None, amp=True):
    tok = get_tokenizer()
    train_ds, val_ds = load_sst2(tok, train_subset=subset)
    train_loader, val_loader = make_loaders(train_ds, val_ds, bs_train=bs)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(DEVICE)

    opt = AdamW(model.parameters(), lr=lr)
    total = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    for ep in range(epochs):
        model.train()
        running = 0.0
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if (i + 1) % 200 == 0:
                print(f"  ep{ep+1} step {i+1}/{len(train_loader)} loss {running/200:.4f}")
                running = 0.0
        m = evaluate(model, val_loader)
        print(f"[epoch {ep+1}] val acc {m['accuracy']*100:.2f}%")

    # save in fp32
    path = os.path.join(ART, "baseline.pt")
    torch.save(model.state_dict(), path)
    m = evaluate(model, val_loader)
    size = model_disk_size_mb(model, path)
    print_report("Baseline (fp32)", m, size)
    return model, m, size


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()
    train(a.epochs, a.bs, a.lr, a.subset, amp=not a.no_amp)
