# GPT-2 Scaling Study — 124M vs 355M vs 774M on WikiText-103

Third experiment in the QAT-VQ series, asking a question the other two
branches couldn't answer alone: **does the QAT+VQ compression/quality
trade-off get better or worse as the model scales up?** Same method, same
dataset (WikiText-103), three model sizes — directly comparable, unlike the
GPT-2/WikiText-2 branch which also changed the dataset alongside the task.
Ran end-to-end on Kaggle (Tesla T4, single GPU per run).

## Headline finding: compression keeps improving with scale; the quality
## gap does not shrink monotonically

| | GPT-2 (124M) | GPT-2-Medium (355M) | GPT-2-Large (774M) |
|---|---|---|---|
| QAT+VQ perplexity gap vs PTQ | +1.68% | +1.30% | **+2.89%** |
| QAT+VQ size vs PTQ | 22.7% smaller | 28.4% smaller | **30.4% smaller** |
| QAT+VQ compression vs baseline | 5.14× | 5.55× | **5.73×** |

With only two points (124M, 355M) the earlier version of this doc claimed
the quality gap "narrows with scale." **The third point breaks that claim** —
the gap widens again at 774M, ending up worse than even the 124M starting
point. The compression-ratio trend, in contrast, *does* hold cleanly across
all three sizes (5.14× → 5.55× → 5.73×, monotonic). Reporting both findings
as they actually are, not force-fitting the original two-point story.

**A real confound, disclosed rather than hidden:** GPT-2-Large's calibration/
fine-tuning subset was cut to 10M characters (vs 20M for the other two
sizes), purely for Kaggle compute-budget reasons — this run alone took
~9.1 hours on a single T4 even at that reduced size, after two earlier OOM
failures forced a switch to 8-bit AdamW (see below). Less fine-tuning data
relative to a 2.2×-bigger model is a plausible explanation for the larger
model's *worse-than-expected* QAT-INT8 and QAT+VQ numbers — this is **not**
a fully controlled scaling comparison at the 774M point, and that should be
weighed before treating the quality-gap reversal as a genuine model-scale
effect rather than a data-budget artifact.

## Full results

**GPT-2 (124M), WikiText-103** (20M-char calibration subset):

| Model | Perplexity | Size (MB) | Compression |
|-------|-----------|-----------|-------------|
| Baseline | 24.33 | 497.8 | 1.00× |
| PTQ | 24.35 | 125.3 | 3.97× |
| QAT-INT8 | 24.14 | 125.3 | 3.97× |
| QAT+VQ | 24.76 | 96.8 | 5.14× |

**GPT-2-Medium (355M), WikiText-103** (20M-char calibration subset):

| Model | Perplexity | Size (MB) | Compression |
|-------|-----------|-----------|-------------|
| Baseline | 18.40 | 1419.4 | 1.00× |
| PTQ | 18.41 | 356.7 | 3.98× |
| QAT-INT8 | 18.34 | 356.7 | 3.98× |
| QAT+VQ | 18.65 | 255.6 | 5.55× |

**GPT-2-Large (774M), WikiText-103** (10M-char calibration subset — half the
other two, see confound note above):

| Model | Perplexity | Size (MB) | Compression |
|-------|-----------|-----------|-------------|
| Baseline | 16.62 | 3096.3 | 1.00× |
| PTQ | 16.62 | 777.3 | 3.98× |
| QAT-INT8 | 17.35 | 777.3 | **+4.4% vs baseline** |
| QAT+VQ | 17.10 | 540.5 | 5.73× |

Perplexity keeps dropping with scale (24→18→17, expected — bigger models are
just better language models), and PTQ stays essentially free of cost at
every size. But **QAT-INT8, which was actually *better* than baseline at
124M and 355M (fine-tuning recovering more than the quantization cost),
flips to notably worse at 774M** (+4.4%) — the same reduced-calibration-data
story likely applies here too, not just to QAT+VQ.

## Method (identical at all three scales)

- MLP weights → 4-bit Product-VQ (K=256 codebook, 2-D sub-vectors), best-of-3
  seeds, then codebook fine-tuned (best-checkpoint guarded).
- Attention weights → per-channel INT8.
- Embeddings → INT8.
- `quant_gpt2.py` is reused completely unmodified from the GPT-2/WikiText-2
  branch — Conv1D layer surgery doesn't care about hidden size, which is
  exactly what makes this a clean method comparison across scales.

Codebook fine-tuning helped at all three scales (matches the WikiText-2
branch's finding, not the DistilBERT branch's): GPT-2-Large's QAT+VQ improved
from a pre-finetune 19.0-ish ppl (best seed) to 17.10 ppl over 2 fine-tune
epochs — the largest fine-tuning recovery of the three sizes, though again
this is entangled with the smaller calibration set used at this scale.

### The 774M run needed 8-bit AdamW to fit in 16GB

Two earlier attempts (batch size 2, then 1) OOM'd at almost the exact same
memory mark regardless of batch size — the giveaway that it wasn't an
activation/batch-size problem at all. Plain AdamW's fp32 optimizer state
(weights + gradients + 2 Adam moments ≈ 4× parameter memory) for 774M
parameters alone is ~12.4 GB, saturating a T4's ~14.5 GB usable budget
before any batch data even enters the picture. Switching to 8-bit AdamW
(`bitsandbytes`) cut optimizer memory roughly 4× and let the run complete —
applied everywhere the (near-)full model is trained: baseline fine-tuning,
the QAT fine-tune step, and the codebook fine-tune step (the latter two also
train nearly all parameters via `FakeQuantConv1D`/`PQConv1D` wrapping, same
OOM exposure as full fine-tuning).

## Honest framing

QAT+VQ does not beat PTQ outright at any of the three scales — it remains a
Pareto point (meaningfully smaller, at some perplexity cost), not a win. The
size trend across scale is real and clean: compression keeps improving as
the model grows (5.14× → 5.55× → 5.73×). The quality-gap trend is *not*
clean once a third point is added — it dips then rises, and the 774M point
is confounded by a smaller calibration subset than the other two sizes used.
**The honest claim from this branch is: compression improves with scale;
whether the quality cost does too is unresolved** — a controlled re-run of
GPT-2-Large with the same 20M-char subset as the other two (more Kaggle GPU
time than was budgeted here) would be needed to settle it cleanly.

## Reproduce

```bash
cd gpt2_scaling
python train_baseline.py --model gpt2 --epochs 1 --bs 8
python run_experiments.py --model gpt2 --sub_dim 2 --K 256 --seeds 0 1 2
python finetune_vq.py --model gpt2 --seed <best> --epochs 2 --lr 5e-6

python train_baseline.py --model gpt2-medium --epochs 1 --bs 4 --grad_accum 2
python run_experiments.py --model gpt2-medium --sub_dim 2 --K 256 --seeds 0 1 2 --bs_train 4 --bs_eval 4
python finetune_vq.py --model gpt2-medium --seed <best> --epochs 2 --lr 5e-6 --bs_train 2 --bs_eval 2

# gpt2-large needs bitsandbytes (pip install bitsandbytes) for 8-bit AdamW,
# and fits a T4's 16GB only at bs=1 even with it.
python train_baseline.py --model gpt2-large --epochs 1 --bs 1 --grad_accum 8 --train_subset_chars 10000000
python run_experiments.py --model gpt2-large --sub_dim 2 --K 256 --seeds 0 1 2 --bs_train 1 --bs_eval 1 --train_subset_chars 10000000
python finetune_vq.py --model gpt2-large --seed <best> --epochs 2 --lr 5e-6 --bs_train 1 --bs_eval 1 --train_subset_chars 10000000

python make_figures.py   # per-model figures + combined scaling_comparison.png
```

See `../kaggle_kernel_scaling/` (124M + 355M, one session) and
`../kaggle_kernel_scaling_large/` (774M only, reuses the other two's
already-committed results for the combined comparison) for the Kaggle
notebooks used.

## Figures

- `artifacts/figures/scaling_comparison.png` — the key deliverable: size and
  relative-perplexity-cost by method, side by side across all three sizes.
- `artifacts/gpt2/figures/ppl_vs_size.png`, `artifacts/gpt2-medium/figures/ppl_vs_size.png`,
  `artifacts/gpt2-large/figures/ppl_vs_size.png` — per-model Pareto scatter.
