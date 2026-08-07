"""Shared utilities: data, evaluation, model-size measurement.

QAT-VQ thesis reproduction + improvement.
DistilBERT on SST-2, targeting: beat PTQ at higher compression.
"""
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64   # SST-2 sentences are short (avg ~25 tokens); 64 covers >99%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_sst2(tokenizer, train_subset=None):
    """Return tokenized train/val datasets. SST-2 test labels are hidden,
    so we evaluate on the validation split (872 samples) like the community does."""
    ds = load_dataset("nyu-mll/glue", "sst2")

    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length",
                         max_length=MAX_LEN)

    ds = ds.map(tok, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    ds.set_format(type="torch", columns=cols)
    train = ds["train"]
    if train_subset:
        train = train.select(range(train_subset))
    return train, ds["validation"]


def make_loaders(train, val, bs_train=16, bs_eval=64):
    return (DataLoader(train, batch_size=bs_train, shuffle=True),
            DataLoader(val, batch_size=bs_eval))


@torch.no_grad()
def evaluate(model, loader, device=DEVICE):
    """Return dict: accuracy, roc_auc, pr_auc, confusion (tn,fp,fn,tp), latency_ms/batch."""
    model.eval()
    all_logits, all_labels = [], []
    t0 = time.time()
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["label"]
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits if hasattr(out, "logits") else out
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels)
        n_batches += 1
    latency_ms = (time.time() - t0) / max(n_batches, 1) * 1000.0

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = logits.argmax(dim=1).numpy()

    acc = float((preds == labels).mean())
    try:
        roc = float(roc_auc_score(labels, probs))
        pr = float(average_precision_score(labels, probs))
    except ValueError:
        roc = pr = float("nan")
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": acc, "roc_auc": roc, "pr_auc": pr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "latency_ms_per_batch": latency_ms,
        "_probs": probs, "_labels": labels,  # for curve plots
    }


def model_disk_size_mb(model, path=None):
    """Serialize state_dict, measure real bytes on disk."""
    import tempfile
    if path is None:
        path = tempfile.mktemp(suffix=".pt")
        state = model.state_dict() if hasattr(model, "state_dict") else model
        torch.save(state, path)
        size = os.path.getsize(path) / 1e6
        os.remove(path)
        return size
    torch.save(model.state_dict(), path)
    return os.path.getsize(path) / 1e6


def print_report(name, metrics, size_mb):
    print(f"\n=== {name} ===")
    print(f"  accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  roc_auc  : {metrics['roc_auc']:.4f}")
    print(f"  pr_auc   : {metrics['pr_auc']:.4f}")
    print(f"  size(MB) : {size_mb:.2f}")
    print(f"  conf     : TP={metrics['tp']} FN={metrics['fn']} "
          f"TN={metrics['tn']} FP={metrics['fp']}")
    print(f"  latency  : {metrics['latency_ms_per_batch']:.1f} ms/batch")
