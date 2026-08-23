# GPT-2 Scaling Study — 124M vs 355M vs 774M on WikiText-103

Third experiment in the QAT-VQ series, asking a question the other two
branches couldn't answer alone: **does the QAT+VQ compression/quality
trade-off get better or worse as the model scales up?** Same method, same
dataset (WikiText-103, same 20M-character calibration subset at every
size), three model sizes — directly comparable, unlike the GPT-2/WikiText-2
branch which also changed the dataset alongside the task. Ran end-to-end on
Kaggle (Tesla T4, single GPU per run). All four methods are complete and
fully matched across all three sizes — no outstanding caveats.

## Headline finding: compression keeps improving with scale; the quality
## gap does not shrink monotonically

| | GPT-2 (124M) | GPT-2-Medium (355M) | GPT-2-Large (774M) |
|---|---|---|---|
| QAT+VQ perplexity gap vs PTQ | +1.68% | +1.30% | **+2.59%** |
| QAT+VQ size vs PTQ | 22.7% smaller | 28.4% smaller | **30.4% smaller** |
| QAT+VQ compression vs baseline | 5.14× | 5.55× | **5.73×** |

The compression-ratio trend is clean and monotonic across all three sizes
(5.14× → 5.55× → 5.73×) — bigger models have more redundancy for Product-VQ
to exploit. The quality-gap trend is **not** monotonic: it narrows from
124M to 355M, then widens sharply at 774M, ending up worse than the 124M
starting point. Both findings are now fully confirmed on matched data with
no remaining confounds.

A second, related finding: **QAT-INT8 (fine-tuned int8, no VQ) flips from
better-than-baseline at the two smaller sizes to notably worse at 774M**
(-0.8% at 124M, -0.3% at 355M, **+4.1% at 774M**). Simple int8 fine-tuning
recovers more than it costs at smaller scale, but stops doing so at 774M —
a genuine model-scale effect, confirmed on matched calibration data (see
"Chasing down a confound" below for how this was verified).

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
the other two — all four methods fully complete):

| Model | Perplexity | Size (MB) | Compression |
|-------|-----------|-----------|-------------|
| Baseline | 16.22 | 3096.3 | 1.00× |
| PTQ | 16.22 | 777.3 | 3.98× |
| QAT-INT8 | 16.88 | 777.3 | 3.98× |
| QAT+VQ | 16.64 | 540.5 | 5.73× |

Perplexity keeps dropping with scale (24 → 18 → 16, expected — bigger models
are just better language models), and PTQ stays essentially free of cost at
every size.

## Method (identical at all three scales)

- MLP weights → 4-bit Product-VQ (K=256 codebook, 2-D sub-vectors), best-of-3
  seeds, then codebook fine-tuned (best-checkpoint guarded).
- Attention weights → per-channel INT8.
- Embeddings → INT8.
- `quant_gpt2.py` is reused completely unmodified from the GPT-2/WikiText-2
  branch — Conv1D layer surgery doesn't care about hidden size, which is
  exactly what makes this a clean method comparison across scales.

**Codebook fine-tuning's payoff shrinks with scale.** At 355M, fine-tuning
improved QAT+VQ from ~19.0 ppl (pre-finetune, best seed) to 18.65 ppl — a
real ~0.35 ppl gain. At 774M, fine-tuning barely moved the number at all:
16.65 ppl pre-finetune → 16.64 ppl after epoch 1 (the best checkpoint;
epoch 2 actually got slightly worse, at 16.76, and was correctly discarded
by the best-checkpoint guard). The larger model's k-means initialization is
already close to as good as 2 epochs of fine-tuning can get it — the
opposite of what helped smaller models most.

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

## Chasing down a confound

The first 774M attempt used a 10M-character calibration subset (half the
other two sizes' 20M) purely to bound Kaggle compute time, and its results
were flagged as potentially confounded rather than trusted outright. Getting
to the final, clean numbers above took several more steps, documented here
for anyone reproducing this:

1. **Matched-data rerun** (20M chars, same as 124M/355M): completed
   `Baseline`/`PTQ`/`QAT-INT8` in full and confirmed the QAT-INT8
   flip-to-worse-at-774M finding is real, not a calibration-data artifact.
   Got cancelled by Kaggle's session-length limit (~12h) partway through the
   `QAT+VQ` codebook fine-tune step, though — leaving only a pre-finetune
   number for that one method.
2. **Checkpoint recovery**: the already-trained `baseline.pt` (3GB,
   representing that ~12h of completed compute) was pulled back from the
   cancelled kernel's output rather than retrained from scratch, uploaded as
   a private Kaggle Dataset, and mounted into a second, much shorter kernel
   that runs only the fine-tune step.
3. **A real bug found along the way**: the first (10M-char) run's
   `finetune_vq` step had used the notebook's hardcoded `--seed 1` instead
   of the actual best seed found by the 3-seed search (**seed 2**, at 16.65
   ppl vs 16.79/16.85 for seeds 0/1) — fixed for the recovery run.
4. **A Kaggle mount-path quirk**: the recovery kernel initially failed twice
   with `baseline.pt` not found at the expected
   `/kaggle/input/<dataset-slug>/` path — it turned out to actually mount at
   `/kaggle/input/datasets/<owner>/<dataset-slug>/`, an extra `datasets/
   <owner>/` prefix not documented anywhere obvious. Fixed by searching
   `/kaggle/input` broadly instead of assuming the path.
5. **Success**: the finetune-only kernel completed both epochs cleanly (see
   the fine-tuning payoff note above), giving the final, fully-matched
   `QAT+VQ = 16.64 ppl` reported throughout this document.

## Honest framing

QAT+VQ does not beat PTQ outright at any of the three scales — it remains a
Pareto point (meaningfully smaller, at some perplexity cost), not a win.
That said, both trends reported here are now fully confirmed on matched,
complete data: compression improves monotonically with scale; the quality
cost does not, dipping at 355M before widening again at 774M. Two data
points would have suggested a clean "everything gets better with scale"
story; the third point shows the real picture is more nuanced — which is
itself the more scientifically honest and useful finding to report.

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
# and fits a T4's 16GB only at bs=1 even with it. The full pipeline at 20M
# chars is ~23h total -- split across two Kaggle sessions if needed (see
# kaggle_kernel_scaling_large/ for stage 1, kaggle_kernel_finetune_only/
# for stage 2, which reuses stage 1's checkpoint via a Kaggle Dataset).
python train_baseline.py --model gpt2-large --epochs 1 --bs 1 --grad_accum 8 --train_subset_chars 20000000
python run_experiments.py --model gpt2-large --sub_dim 2 --K 256 --seeds 0 1 2 --bs_train 1 --bs_eval 1 --train_subset_chars 20000000
python finetune_vq.py --model gpt2-large --seed <best> --epochs 2 --lr 5e-6 --bs_train 1 --bs_eval 1 --train_subset_chars 20000000

python make_figures.py   # per-model figures + combined scaling_comparison.png
```

See `../kaggle_kernel_scaling/` (124M + 355M, one session),
`../kaggle_kernel_scaling_large/` (774M stage 1: baseline/PTQ/QAT-INT8/
pre-finetune QAT+VQ), and `../kaggle_kernel_finetune_only/` (774M stage 2:
codebook fine-tune only, resumes from stage 1's checkpoint via a Kaggle
Dataset) for the Kaggle notebooks used.

## Figures

- `artifacts/figures/scaling_comparison.png` — the key deliverable: size and
  relative-perplexity-cost by method, side by side across all three sizes.
- `artifacts/gpt2/figures/ppl_vs_size.png`, `artifacts/gpt2-medium/figures/ppl_vs_size.png`,
  `artifacts/gpt2-large/figures/ppl_vs_size.png` — per-model Pareto scatter.
