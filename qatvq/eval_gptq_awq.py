"""Evaluate GPTQ and AWQ as extra baselines against the existing
Baseline/PTQ/QAT-INT8/QAT+VQ results, on the same fine-tuned DistilBERT
checkpoint and SST-2 validation set. Post-training only (no retraining) --
these are real SOTA-family methods, implemented from scratch (see
gptq_awq.py), used purely to check QAT+VQ against literature-standard
baselines rather than only our own PTQ/QAT.
"""
import os
import copy
import json
import argparse
import torch
from transformers import AutoModelForSequenceClassification

from common import get_tokenizer, load_sst2, make_loaders, evaluate, print_report, MODEL_NAME, DEVICE
import gptq_awq as ga

ART = os.path.join(os.path.dirname(__file__), "artifacts")
ENCODER = lambda n: "transformer.layer" in n   # same scope as PTQ/QAT (encoder linears only)


def load_baseline():
    m = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    m.load_state_dict(torch.load(os.path.join(ART, "baseline.pt"), map_location="cpu"))
    return m.to(DEVICE)


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


def main(n_calib_batches):
    tok = get_tokenizer()
    train_ds, val_ds = load_sst2(tok)
    train_loader, val_loader = make_loaders(train_ds, val_ds, bs_train=16)
    base = load_baseline()

    with open(os.path.join(ART, "results.json")) as f:
        data = json.load(f)
    results = data["results"]

    # GPTQ
    gptq_model = ga.apply_gptq(copy.deepcopy(base), train_loader, DEVICE,
                               name_filter=ENCODER, n_batches=n_calib_batches)
    m = evaluate(gptq_model, val_loader)
    record(results, "GPTQ", m, ga.packed_size_mb(gptq_model))
    del gptq_model; torch.cuda.empty_cache()

    # AWQ
    awq_model = ga.apply_awq(copy.deepcopy(base), train_loader, DEVICE,
                             name_filter=ENCODER, n_batches=n_calib_batches)
    m = evaluate(awq_model, val_loader)
    record(results, "AWQ", m, ga.packed_size_mb(awq_model))
    del awq_model; torch.cuda.empty_cache()

    with open(os.path.join(ART, "results.json"), "w") as f:
        json.dump(data, f, indent=2)
    print("\nSaved results.json (added GPTQ, AWQ)")

    print("\n================ COMPARISON ================")
    for name in ["PTQ", "GPTQ", "AWQ", "QAT+VQ"]:
        r = results[name]
        print(f"{name:8s}: acc {r['accuracy']*100:.2f}%  size {r['size_mb']:.1f}MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_calib_batches", type=int, default=16)
    a = ap.parse_args()
    main(a.n_calib_batches)
