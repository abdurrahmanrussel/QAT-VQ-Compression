"""Evaluate GPTQ and AWQ as extra baselines against Baseline/PTQ/QAT-INT8/
QAT+VQ, on the same fine-tuned GPT-2 checkpoint and WikiText-2. Post-training
only (no retraining) -- mirrors qatvq/eval_gptq_awq.py.
"""
import os
import copy
import json
import argparse
import torch
from transformers import GPT2LMHeadModel

from common_gpt2 import (get_tokenizer, load_wikitext2, make_loaders, evaluate_ppl,
                         print_report, MODEL_NAME, DEVICE)
import gptq_awq_conv1d as ga

ART = os.path.join(os.path.dirname(__file__), "artifacts")
is_mlp = lambda n: ".mlp.c_" in n
is_attn = lambda n: ".attn.c_" in n
is_any_conv = lambda n: is_mlp(n) or is_attn(n)


def load_baseline():
    m = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    m.load_state_dict(torch.load(os.path.join(ART, "baseline.pt"), map_location="cpu"))
    return m.to(DEVICE)


def record(results, name, metrics, size_mb):
    print_report(name, metrics, size_mb)
    results[name] = {"perplexity": metrics["perplexity"], "avg_loss": metrics["avg_loss"],
                     "size_mb": size_mb, "latency_ms_per_batch": metrics["latency_ms_per_batch"]}


def main(n_calib_batches):
    tok = get_tokenizer()
    train_blocks, val_blocks, _ = load_wikitext2(tok)
    train_loader, val_loader = make_loaders(train_blocks, val_blocks, bs_train=4, bs_eval=4)
    base = load_baseline()

    with open(os.path.join(ART, "results.json")) as f:
        data = json.load(f)
    results = data["results"]

    gptq_model = ga.apply_gptq(copy.deepcopy(base), train_loader, DEVICE,
                               name_filter=is_any_conv, n_batches=n_calib_batches)
    m = evaluate_ppl(gptq_model, val_loader)
    record(results, "GPTQ", m, ga.packed_size_mb(gptq_model))
    del gptq_model; torch.cuda.empty_cache()

    awq_model = ga.apply_awq(copy.deepcopy(base), train_loader, DEVICE,
                             name_filter=is_any_conv, n_batches=n_calib_batches)
    m = evaluate_ppl(awq_model, val_loader)
    record(results, "AWQ", m, ga.packed_size_mb(awq_model))
    del awq_model; torch.cuda.empty_cache()

    with open(os.path.join(ART, "results.json"), "w") as f:
        json.dump(data, f, indent=2)
    print("\nSaved results.json (added GPTQ, AWQ)")

    print("\n================ COMPARISON ================")
    for name in ["PTQ", "GPTQ", "AWQ", "QAT+VQ"]:
        r = results[name]
        print(f"{name:8s}: ppl {r['perplexity']:.2f}  size {r['size_mb']:.1f}MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_calib_batches", type=int, default=16)
    a = ap.parse_args()
    main(a.n_calib_batches)
