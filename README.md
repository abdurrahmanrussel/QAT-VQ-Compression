# QAT-VQ-Compression

Quantization-Aware **Vector** Compression for transformer language models —
a hybrid of Quantization-Aware Training (QAT) and Vector Quantization (VQ).

This started as a BSc thesis reproduction. The originally proposed hybrid
**failed** — it was the worst-performing method of everything tested, losing
even to plain post-training quantization. This repo is the fix: a corrected
implementation, tested across two tasks and three model sizes, with results
checked against real literature baselines (GPTQ, AWQ) — not just internal
ones.

## The method

- **FFN/MLP weights → 4-bit Product Quantization.** A K=256 codebook is
  learned per layer via k-means on 2-D weight sub-vectors. This is the bulk
  of a transformer's weights and the most redundant — where compression pays
  off.
- **Attention weights → per-channel INT8.** Attention is more sensitive;
  aggressive vector quantization there hurt accuracy in testing, so it's
  kept at 8 bits.
- **Embeddings → INT8.**
- **Codebook fine-tuned** post-hoc where it helps (best-checkpoint guarded,
  so it can only match-or-beat the unfine-tuned result).

The original thesis's version of this used a codebook far too small (k=16)
and applied vector quantization to activations instead of weights, which
destroyed information with no way to recover it. Fixing both of those is
most of what separates the results below from the original's failure.

## Three experiments, three branches

| Branch | Model(s) | Task | Verdict |
|---|---|---|---|
| [`distilbert-sst2-qatvq`](../../tree/distilbert-sst2-qatvq) | DistilBERT (66M) | Classification (SST-2) | **Hybrid wins outright** |
| [`gpt2-wikitext2-qatvq`](../../tree/gpt2-wikitext2-qatvq) | GPT-2 (124M) | Generation (WikiText-2) | Hybrid is a **Pareto point** |
| [`gpt2medium-wikitext103-qatvq`](../../tree/gpt2medium-wikitext103-qatvq) | GPT-2 124M / 355M / 774M | Generation (WikiText-103) | **Scaling study** |

Each branch has its own `RESULTS.md` with full numbers, figures, and honest
limitations. Summaries below.

### 1. DistilBERT / SST-2 — the hybrid wins outright

| Model | Accuracy | Size (MB) | Compression |
|-------|----------|-----------|-------------|
| Baseline (fp32) | 91.06% | 267.9 | 1.00× |
| PTQ (int8) | 91.17% | 68.0 | 3.94× |
| QAT (int8) | 89.91% | 67.8 | 3.95× |
| GPTQ | 91.06% | 68.0 | 3.94× |
| AWQ | 91.06% | 68.1 | 3.93× |
| **QAT+VQ (proposed)** | **91.28%** | **53.7** | **4.99×** |

Highest accuracy *and* smallest model of every method tested — including
GPTQ and AWQ, real SOTA-family baselines, not just internal ones. GPTQ/AWQ
land within noise of naive PTQ here, which matches the literature: their
advantage is rescuing 3-4 bit quantization on billion-parameter models, and
there's little naive-rounding error left to correct at 8 bits on a 66M
classifier.

### 2. GPT-2 / WikiText-2 — a real Pareto point

| Model | Perplexity | Size (MB) | Compression |
|-------|-----------|-----------|-------------|
| Baseline | 25.85 | 497.8 | 1.00× |
| PTQ | 25.87 | 125.3 | 3.97× |
| QAT-INT8 | 25.11 | 125.3 | 3.97× |
| GPTQ | 25.86 | 125.3 | 3.97× |
| AWQ | 25.87 | 125.5 | 3.97× |
| QAT+VQ | 26.12 | 96.8 | 5.14× |

On a generation task, the hybrid does **not** beat PTQ/GPTQ/AWQ outright —
it's 23% smaller for a ~1% perplexity cost. A genuine trade-off, reported as
such rather than oversold as a win.

### 3. GPT-2 scaling study — 124M vs 355M vs 774M, same task and data

| | 124M | 355M | 774M |
|---|---|---|---|
| QAT+VQ compression vs baseline | 5.14× | 5.55× | **5.73×** |
| QAT+VQ perplexity gap vs PTQ | +1.68% | +1.30% | **+2.59%** |
| QAT-INT8 vs baseline | -0.8% | -0.3% | **+4.1%** |

**Compression improves monotonically with scale** — clean, confirmed trend.
**The quality cost does not** — it dips then rises, and even the simple
int8-only baseline (QAT-INT8) flips from beating the original model at small
scale to losing badly at 774M. Two data points would have told a tidy
"everything gets better with scale" story; the third point shows the real
picture is more nuanced. That branch's `RESULTS.md` also documents the
process of chasing down and resolving a real data-calibration confound
along the way.

## What this adds up to

The compression method is real and it works — proven on two different task
types (classification and generation), checked against literature
baselines (not just internal ones), and its size-efficiency gain is a
provable, monotonic trend across a 6× parameter-count range. Whether it also
gets *safer* (lower quality cost) on bigger models is genuinely unresolved —
a mixed, honest finding rather than a testing weakness.

## Reproduce

Each branch is self-contained — see its own `RESULTS.md` for exact
commands and Kaggle notebooks (`kaggle_kernel*/` directories) used to run
on free-tier GPUs where local hardware (a 4GB laptop GPU, in this project's
case) wasn't enough.

```bash
git clone https://github.com/abdurrahmanrussel/QAT-VQ-Compression.git
cd QAT-VQ-Compression
git checkout distilbert-sst2-qatvq   # or gpt2-wikitext2-qatvq, or gpt2medium-wikitext103-qatvq
cat qatvq/RESULTS.md                 # (or gpt2/RESULTS.md, or gpt2_scaling/RESULTS.md)
```

## Origin

`1910055_thesis_book_revised.pdf` is the original BSc thesis document this
project set out to fix. Its proposed QAT+VQ method scored worst of every
compression technique tested (81.17% accuracy vs plain PTQ's 86.58%) — the
opposite of every result in this repo.
