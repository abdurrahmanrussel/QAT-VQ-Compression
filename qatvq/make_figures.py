"""Generate thesis figures + comparison table from artifacts/results.json."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                             confusion_matrix)

ART = os.path.join(os.path.dirname(__file__), "artifacts")
FIG = os.path.join(ART, "figures")
os.makedirs(FIG, exist_ok=True)
ORDER = ["Baseline", "PTQ", "QAT-INT8", "QAT+VQ"]


def load():
    with open(os.path.join(ART, "results.json")) as f:
        return json.load(f)["results"]


def cm_fig(r, name):
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Neg", "Pos"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Neg", "Pos"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name} Confusion Matrix")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"cm_{name.replace('+','').replace('-','')}.png"), dpi=150)
    plt.close(fig)


def curves(res):
    # ROC
    fig, ax = plt.subplots(figsize=(5, 4))
    for name in ORDER:
        r = res[name]
        fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "roc_all.png"), dpi=150); plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(5, 4))
    for name in ORDER:
        r = res[name]
        prec, rec, _ = precision_recall_curve(r["labels"], r["probs"])
        ax.plot(rec, prec, label=f"{name} (AUC={auc(rec,prec):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pr_all.png"), dpi=150); plt.close(fig)


def scatter_acc_size(res):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in ORDER:
        r = res[name]
        ax.scatter(r["size_mb"], r["accuracy"] * 100, s=70)
        ax.annotate(name, (r["size_mb"], r["accuracy"] * 100),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Model Size (MB)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Model Size"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "acc_vs_size.png"), dpi=150); plt.close(fig)


def bar_compare(res):
    base = res["Baseline"]["size_mb"]
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    sizes = [res[n]["size_mb"] for n in ORDER]
    accs = [res[n]["accuracy"] * 100 for n in ORDER]
    x = np.arange(len(ORDER))
    ax1.bar(x, sizes, color="#7ec8e3", label="Size (MB)")
    for i, s in enumerate(sizes):
        ax1.text(i, s, f"{s:.1f}\n{base/s:.2f}x", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Model Size (MB)"); ax1.set_xticks(x); ax1.set_xticklabels(ORDER)
    ax2 = ax1.twinx()
    ax2.plot(x, accs, "o-", color="#e67e22", label="Accuracy (%)")
    for i, a in enumerate(accs):
        ax2.text(i, a, f"{a:.1f}%", color="#e67e22", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Accuracy (%)"); ax2.set_ylim(min(accs) - 3, 100)
    ax1.set_title("Size & Accuracy by Compression Method")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "size_acc_bar.png"), dpi=150); plt.close(fig)


def table(res):
    base = res["Baseline"]["size_mb"]
    lines = ["| Model | Accuracy | Size (MB) | Compression | ROC-AUC | PR-AUC |",
             "|-------|----------|-----------|-------------|---------|--------|"]
    for n in ORDER:
        r = res[n]
        lines.append(f"| {n} | {r['accuracy']*100:.2f}% | {r['size_mb']:.1f} | "
                     f"{base/r['size_mb']:.2f}x | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} |")
    txt = "\n".join(lines)
    with open(os.path.join(ART, "results_table.md"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    res = load()
    for n in ORDER:
        cm_fig(res[n], n)
    curves(res)
    scatter_acc_size(res)
    bar_compare(res)
    table(res)
    print(f"\nFigures -> {FIG}")
