"""Generate per-model figures/tables AND the combined scaling comparison
(GPT-2 124M vs GPT-2-Medium 355M, both on WikiText-103) -- the actual
deliverable of this branch: does the QAT+VQ compression/quality trade hold
as the model gets bigger?
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = os.path.join(os.path.dirname(__file__), "artifacts")
FIG = os.path.join(ART, "figures")
os.makedirs(FIG, exist_ok=True)
ORDER = ["Baseline", "PTQ", "QAT-INT8", "QAT+VQ"]
MODELS = ["gpt2", "gpt2-medium"]
MODEL_LABEL = {"gpt2": "GPT-2 (124M)", "gpt2-medium": "GPT-2-Medium (355M)"}


def load(model_name):
    path = os.path.join(ART, model_name.replace("/", "_"), "results.json")
    with open(path) as f:
        return json.load(f)["results"]


def per_model_outputs(model_name, res):
    fig_dir = os.path.join(ART, model_name.replace("/", "_"), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in ORDER:
        r = res[name]
        ax.scatter(r["size_mb"], r["perplexity"], s=70)
        ax.annotate(name, (r["size_mb"], r["perplexity"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Model Size (MB)"); ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title(f"{MODEL_LABEL[model_name]}: Perplexity vs. Model Size"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "ppl_vs_size.png"), dpi=150); plt.close(fig)

    base = res["Baseline"]["size_mb"]
    lines = ["| Model | Perplexity | Size (MB) | Compression |",
             "|-------|-----------|-----------|-------------|"]
    for n in ORDER:
        r = res[n]
        lines.append(f"| {n} | {r['perplexity']:.2f} | {r['size_mb']:.1f} | {base/r['size_mb']:.2f}x |")
    txt = "\n".join(lines)
    with open(os.path.join(ART, model_name.replace("/", "_"), "results_table.md"), "w") as f:
        f.write(txt + "\n")
    print(f"\n=== {MODEL_LABEL[model_name]} ===")
    print(txt)


def scaling_comparison(all_res):
    """The key figure: does compression ratio / relative ppl gap hold as
    the model scales from 124M to 355M?"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(ORDER))
    width = 0.35
    colors = {"gpt2": "#7ec8e3", "gpt2-medium": "#e67e22"}

    for i, model in enumerate(MODELS):
        res = all_res[model]
        sizes = [res[n]["size_mb"] for n in ORDER]
        offset = (i - 0.5) * width
        ax1.bar(x + offset, sizes, width, label=MODEL_LABEL[model], color=colors[model])
    ax1.set_xticks(x); ax1.set_xticklabels(ORDER, rotation=15)
    ax1.set_ylabel("Model Size (MB)"); ax1.set_yscale("log")
    ax1.set_title("Size by Method, Across Scale"); ax1.legend(fontsize=8)

    for model in MODELS:
        res = all_res[model]
        base_ppl = res["Baseline"]["perplexity"]
        rel_ppl = [(res[n]["perplexity"] - base_ppl) / base_ppl * 100 for n in ORDER]
        ax2.plot(x, rel_ppl, "o-", label=MODEL_LABEL[model], color=colors[model])
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xticks(x); ax2.set_xticklabels(ORDER, rotation=15)
    ax2.set_ylabel("Perplexity vs Baseline (%)")
    ax2.set_title("Quality Cost by Method, Across Scale"); ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "scaling_comparison.png"), dpi=150)
    plt.close(fig)


def scaling_table(all_res):
    lines = ["| Model | Method | Perplexity | Size (MB) | Compression | Δppl vs Baseline |",
             "|-------|--------|-----------|-----------|-------------|-------------------|"]
    for model in MODELS:
        res = all_res[model]
        base = res["Baseline"]["size_mb"]
        base_ppl = res["Baseline"]["perplexity"]
        for n in ORDER:
            r = res[n]
            d = (r["perplexity"] - base_ppl) / base_ppl * 100
            lines.append(f"| {MODEL_LABEL[model]} | {n} | {r['perplexity']:.2f} | "
                        f"{r['size_mb']:.1f} | {base/r['size_mb']:.2f}x | {d:+.1f}% |")
    txt = "\n".join(lines)
    with open(os.path.join(ART, "scaling_table.md"), "w") as f:
        f.write(txt + "\n")
    print("\n=== Combined scaling table ===")
    print(txt)


if __name__ == "__main__":
    all_res = {}
    for model in MODELS:
        path = os.path.join(ART, model.replace("/", "_"), "results.json")
        if not os.path.exists(path):
            print(f"skip {model}: {path} not found yet")
            continue
        res = load(model)
        all_res[model] = res
        per_model_outputs(model, res)

    if len(all_res) == len(MODELS):
        scaling_comparison(all_res)
        scaling_table(all_res)
        print(f"\nScaling figures -> {FIG}")
    else:
        print("\nRun both models' pipelines before generating the scaling comparison.")
