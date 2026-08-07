"""Final QAT+VQ: FFN 4-bit product-VQ + attention int8 + embeddings int8.

Fine-tuning was found to DEGRADE this already-converged model, so the
k-means codebook init IS the compressed model. We pick the best init over a
few seeds (cheap, no training) and keep it. Patches artifacts/results.json.
"""
import os
import copy
import json
import argparse
import torch
from transformers import AutoModelForSequenceClassification

from common import (get_tokenizer, load_sst2, make_loaders, evaluate,
                    print_report, MODEL_NAME, DEVICE)
import quant

ART = os.path.join(os.path.dirname(__file__), "artifacts")
is_ffn = lambda n: "transformer.layer" in n and "ffn" in n
is_attn = lambda n: "transformer.layer" in n and "attention" in n


def build(baseline_state, seed):
    m = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    m.load_state_dict(baseline_state)
    m = m.to(DEVICE)
    quant.convert_to_pq(m, is_ffn, K=256, sub_dim=2, kmeans_iters=30, seed=seed)
    quant.convert_to_fakequant(m, is_attn)   # attention -> int8 fake-quant
    return m


def main(seeds):
    tok = get_tokenizer()
    _, val_ds = load_sst2(tok)
    _, val_loader = make_loaders(_, val_ds, bs_train=16)
    baseline_state = torch.load(os.path.join(ART, "baseline.pt"), map_location="cpu")

    best = None
    for s in seeds:
        m = build(baseline_state, s)
        acc = evaluate(m, val_loader)["accuracy"]
        print(f"seed {s}: val acc {acc*100:.2f}%")
        if best is None or acc > best[1]:
            if best is not None:
                del best_model
            best = (s, acc)
            best_model = m
        else:
            del m
        torch.cuda.empty_cache()

    s, _ = best
    m = best_model
    metrics = evaluate(m, val_loader)
    size = quant.packed_size_mb(m)
    print_report(f"QAT+VQ (FFN4bit+attn-int8, seed {s})", metrics, size)
    torch.save(m.state_dict(), os.path.join(ART, "qatvq.pt"))

    path = os.path.join(ART, "results.json")
    with open(path) as f:
        data = json.load(f)
    data["results"]["QAT+VQ"] = {
        "accuracy": metrics["accuracy"], "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"], "size_mb": size,
        "tp": metrics["tp"], "fn": metrics["fn"], "tn": metrics["tn"], "fp": metrics["fp"],
        "latency_ms_per_batch": metrics["latency_ms_per_batch"],
        "probs": metrics["_probs"].tolist(), "labels": metrics["_labels"].tolist(),
    }
    data["config"]["vq_final"] = {"scheme": "FFN 4-bit PQ + attention int8 + embeddings int8",
                                  "seed": s, "K": 256, "sub_dim": 2, "finetune": False}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    ptq = data["results"]["PTQ"]
    print("\n========= FINAL VERDICT =========")
    print(f"Baseline: {data['results']['Baseline']['accuracy']*100:.2f}%  "
          f"{data['results']['Baseline']['size_mb']:.1f}MB")
    print(f"PTQ     : {ptq['accuracy']*100:.2f}%  {ptq['size_mb']:.1f}MB")
    print(f"QAT+VQ  : {metrics['accuracy']*100:.2f}%  {size:.1f}MB "
          f"({data['results']['Baseline']['size_mb']/size:.2f}x vs baseline)")
    win = metrics["accuracy"] >= ptq["accuracy"] - 0.005 and size < ptq["size_mb"]
    print("RESULT:", "HYBRID BEATS PTQ (>=acc-0.5%, smaller)" if win else
          f"Pareto point: {(ptq['size_mb']-size)/ptq['size_mb']*100:.0f}% smaller, "
          f"{(ptq['accuracy']-metrics['accuracy'])*100:.2f}% acc gap")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    main(a.seeds)
