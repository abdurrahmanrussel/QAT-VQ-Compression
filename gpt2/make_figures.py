"""Generate GPT-2/WikiText-2 figures + comparison table from results.json."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = os.path.join(os.path.dirname(__file__), "artifacts")
FIG = os.path.join(ART, "figures")
os.makedirs(FIG, exist_ok=True)
ORDER = ["Baseline", "PTQ", "QAT-INT8", "GPTQ", "AWQ", "QAT+VQ"]


def load():
    with open(os.path.join(ART, "results.json")) as f:
        return json.load(f)["results"]


def scatter_ppl_size(res):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in ORDER:
        r = res[name]
        ax.scatter(r["size_mb"], r["perplexity"], s=70)
        ax.annotate(name, (r["size_mb"], r["perplexity"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Model Size (MB)"); ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Perplexity vs. Model Size"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ppl_vs_size.png"), dpi=150); plt.close(fig)


def bar_compare(res):
    base = res["Baseline"]["size_mb"]
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    sizes = [res[n]["size_mb"] for n in ORDER]
    ppls = [res[n]["perplexity"] for n in ORDER]
    x = np.arange(len(ORDER))
    ax1.bar(x, sizes, color="#7ec8e3")
    for i, s in enumerate(sizes):
        ax1.text(i, s, f"{s:.1f}\n{base/s:.2f}x", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Model Size (MB)"); ax1.set_xticks(x); ax1.set_xticklabels(ORDER)
    ax2 = ax1.twinx()
    ax2.plot(x, ppls, "o-", color="#e67e22")
    for i, p in enumerate(ppls):
        ax2.text(i, p, f"{p:.1f}", color="#e67e22", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Perplexity")
    ax1.set_title("GPT-2 / WikiText-2: Size & Perplexity by Compression Method")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "size_ppl_bar.png"), dpi=150); plt.close(fig)


def table(res):
    base = res["Baseline"]["size_mb"]
    lines = ["| Model | Perplexity | Size (MB) | Compression |",
             "|-------|-----------|-----------|-------------|"]
    for n in ORDER:
        r = res[n]
        lines.append(f"| {n} | {r['perplexity']:.2f} | {r['size_mb']:.1f} | {base/r['size_mb']:.2f}x |")
    txt = "\n".join(lines)
    with open(os.path.join(ART, "results_table.md"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    res = load()
    scatter_ppl_size(res)
    bar_compare(res)
    table(res)
    print(f"\nFigures -> {FIG}")
