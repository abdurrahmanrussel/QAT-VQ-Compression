# QAT-VQ-Compression

Quantization-Aware **Vector** Compression for transformer language models —
a hybrid of Quantization-Aware Training (QAT) and Vector Quantization (VQ).

BSc thesis reproduction **and fix**: the originally-proposed hybrid failed
(worst of all methods). This repo contains a corrected implementation where the
hybrid becomes the **best** method — highest accuracy *and* smallest model.

## Results — DistilBERT / SST-2 (validation, GTX 1650 4 GB)

| Model | Accuracy | Size (MB) | Compression | ROC-AUC | PR-AUC |
|-------|----------|-----------|-------------|---------|--------|
| Baseline (fp32) | 91.06% | 267.9 | 1.00× | 0.9711 | 0.9715 |
| PTQ (int8) | 91.17% | 68.0 | 3.94× | 0.9713 | 0.9717 |
| QAT (int8) | 89.91% | 67.8 | 3.95× | 0.9705 | 0.9707 |
| **QAT+VQ (proposed)** | **91.28%** | **53.7** | **4.99×** | **0.9719** | **0.9722** |

QAT+VQ = best accuracy **and** smallest footprint (5× vs baseline, 21% smaller
than PTQ). Full analysis in [`qatvq/RESULTS.md`](qatvq/RESULTS.md).

## The method

- **FFN weights → 4-bit Product Quantization** (K=256 codebook, 2-D sub-vectors).
  FFN is the bulk of the encoder and the most redundant.
- **Attention weights → per-channel INT8** (sensitive; kept at 8 bits).
- **Embeddings → INT8**. No activation VQ (that was the original mistake).

The K=256 codebook holds accuracy at 4 bits/weight where the original k=16 did
not. `qatvq/quant.py` contains the Product-VQ implementation.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.40" "datasets>=2.18" scikit-learn matplotlib
cd qatvq
python train_baseline.py --epochs 1 --bs 32     # -> artifacts/baseline.pt
python run_experiments.py --sub_dim 2 --K 256    # PTQ / QAT / uniform QAT+VQ
python pq_seed_search.py --seeds 0 1 2 3 4        # final QAT+VQ (FFN4bit+attn-int8)
python make_figures.py                            # figures + results_table.md
```

Model weights (`*.pt`) and the venv are git-ignored; run the scripts to
regenerate them.

## Layout

```
qatvq/
  common.py            data loading, evaluation, size accounting
  quant.py             INT8 + Product-VQ layers (the fix)
  train_baseline.py    fine-tune DistilBERT on SST-2
  run_experiments.py   baseline / PTQ / QAT / uniform QAT+VQ + verdict
  pq_seed_search.py    final QAT+VQ (FFN 4-bit VQ + attention int8)
  make_figures.py      figures + comparison table
  RESULTS.md           thesis-ready writeup (diagnosis + fix + tables)
  artifacts/figures/   plots
1910055_thesis_book_revised.pdf   original thesis book
```
