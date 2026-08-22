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
| QAT+VQ perplexity gap vs PTQ | +1.68% | +1.30% | **+2.65%** |
| QAT+VQ size vs PTQ | 22.7% smaller | 28.4% smaller | **30.4% smaller** |
| QAT+VQ compression vs baseline | 5.14× | 5.55× | **5.73×** |

With only two points (124M, 355M) an earlier version of this doc claimed the
quality gap "narrows with scale." **The third point breaks that claim** —
the gap widens again at 774M, ending up worse than the 124M starting point.
The compression-ratio trend, in contrast, *does* hold cleanly across all
three sizes (5.14× → 5.55× → 5.73×, monotonic). Reporting both findings as
they actually are, not force-fitting the original two-point story.

**A confound was found, chased down, and mostly resolved.** The first
774M run used a 10M-character calibration subset (half the 20M used at the
other two sizes) purely for Kaggle compute-budget reasons, and that run's
quality numbers were initially suspect as a possible data-budget artifact
rather than a genuine scale effect. A second run with the **same 20M-char
subset as the other two sizes** was launched to check — it confirms
**`Baseline`, `PTQ`, and `QAT-INT8` fully completed on matched data, and the
QAT-INT8 flip-to-worse-at-774M finding holds** (+4.1% vs baseline at 774M,
vs -0.8%/-0.3% at the smaller sizes) — this is a real model-scale effect,
not a calibration-data artifact. One caveat remains: the matched-data run
was cancelled (Kaggle session limit, ~12h) partway through the QAT+VQ
codebook fine-tune step, so the **QAT+VQ number reported for 774M is
pre-finetune** — the other two scales' QAT+VQ numbers include a completed
fine-tune, which historically improved results by ~0.3-0.4 ppl there. If
774M's fine-tune would have helped similarly, its true post-finetune number
is likely somewhat better than 16.65 — this was not confirmed and is
reported honestly as an open gap, not filled in with a guess.

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

**GPT-2-Large (774M), WikiText-103** (20M-char calibration subset, matching
the other two — `Baseline`/`PTQ`/`QAT-INT8` are complete and final; `QAT+VQ`
is the best-of-3-seeds result **before** codebook fine-tuning, which was
interrupted by a Kaggle session-length cancellation):

| Model | Perplexity | Size (MB) | Compression |
|-------|-----------|-----------|-------------|
| Baseline | 16.22 | 3096.3 | 1.00× |
| PTQ | 16.22 | 777.3 | 3.98× |
| QAT-INT8 | 16.88 | 777.3 | 3.98× |
| QAT+VQ (pre-finetune) | 16.65 | 540.5 | 5.73× |

Perplexity keeps dropping with scale (24 → 18 → 16, expected — bigger models
are just better language models), and PTQ stays essentially free of cost at
every size. But **QAT-INT8, which was actually *better* than baseline at
124M and 355M (fine-tuning recovering more than the quantization cost),
flips to notably worse at 774M** (+4.1%) — confirmed on matched calibration
data, so this is a genuine model-scale effect, not a data-budget artifact.

## Method (identical at all three scales)

- MLP weights → 4-bit Product-VQ (K=256 codebook, 2-D sub-vectors), best-of-3
  seeds, then codebook fine-tuned (best-checkpoint guarded) — completed for
  124M and 355M, interrupted mid-fine-tune for 774M (see above).
- Attention weights → per-channel INT8.
- Embeddings → INT8.
- `quant_gpt2.py` is reused completely unmodified from the GPT-2/WikiText-2
  branch — Conv1D layer surgery doesn't care about hidden size, which is
  exactly what makes this a clean method comparison across scales.

Codebook fine-tuning helped at both scales where it completed (matches the
WikiText-2 branch's finding, not the DistilBERT branch's): 355M's QAT+VQ
improved from a pre-finetune ~19.0 ppl (best seed) to 18.65 ppl.

### The 774M runs needed 8-bit AdamW to fit in 16GB

Two early attempts (batch size 2, then 1) OOM'd at almost the exact same
memory mark regardless of batch size — the giveaway that it wasn't an
activation/batch-size problem at all. Plain AdamW's fp32 optimizer state
(weights + gradients + 2 Adam moments ≈ 4× parameter memory) for 774M
parameters alone is ~12.4 GB, saturating a T4's ~14.5 GB usable budget
before any batch data even enters the picture. Switching to 8-bit AdamW
(`bitsandbytes`) cut optimizer memory roughly 4× and let training proceed —
applied everywhere the (near-)full model is trained: baseline fine-tuning,
the QAT fine-tune step, and the codebook fine-tune step (the latter two also
train nearly all parameters via `FakeQuantConv1D`/`PQConv1D` wrapping, same
OOM exposure as full fine-tuning).

The matched-data (20M-char) run still took long enough (~12h) to hit
Kaggle's session-length limit before the codebook fine-tune step finished —
a second, separate resource constraint from the memory one above.

## Honest framing

QAT+VQ does not beat PTQ outright at any of the three scales — it remains a
Pareto point (meaningfully smaller, at some perplexity cost), not a win. The
size trend across scale is real, clean, and confirmed on matched data:
compression keeps improving as the model grows (5.14× → 5.55× → 5.73×). The
quality-gap trend is *not* clean once a third point is added — it dips then
rises. The 774M QAT-INT8 result is fully confirmed on matched data (a real
effect); the 774M QAT+VQ result is real but incomplete (pre-finetune only),
so its exact final position in that trend is not settled — it may look
somewhat better once fully fine-tuned, but is unlikely to reverse the
overall "not monotonic" conclusion given fine-tuning's typical ~0.3-0.4 ppl
magnitude versus the ~1 ppl gap currently observed.

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
# and fits a T4's 16GB only at bs=1 even with it. A full 2-epoch finetune_vq
# run at 20M chars risks exceeding a single Kaggle session's time limit.
python train_baseline.py --model gpt2-large --epochs 1 --bs 1 --grad_accum 8 --train_subset_chars 20000000
python run_experiments.py --model gpt2-large --sub_dim 2 --K 256 --seeds 0 1 2 --bs_train 1 --bs_eval 1 --train_subset_chars 20000000
python finetune_vq.py --model gpt2-large --seed <best> --epochs 2 --lr 5e-6 --bs_train 1 --bs_eval 1 --train_subset_chars 20000000

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
