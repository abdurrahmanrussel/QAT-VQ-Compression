# GPT-2 Scaling Study — GPT-2 (124M) vs GPT-2-Medium (355M) on WikiText-103

Third experiment in the QAT-VQ series. The DistilBERT branch showed the
hybrid winning outright; the GPT-2/WikiText-2 branch showed it as a tight
Pareto point. This branch asks: **does that trade-off hold as the model
scales up?** — same method, same dataset, two model sizes, directly
comparable (unlike GPT-2-small/WikiText-2 vs this, which also changes the
dataset).

## Setup
- Models: `gpt2` (124M, already validated on WikiText-2) and `gpt2-medium`
  (355M), both HuggingFace `Conv1D`-based, same architecture family.
- Dataset: `Salesforce/wikitext` `wikitext-103-raw-v1` — training text capped
  at ~20M chars (~5M tokens, still ~2x WikiText-2's full train set) for
  compute budget; validation/test use the FULL WikiText-103 splits (real,
  harder benchmark than WikiText-2, similar size to before so eval is
  apples-to-apples on scale).
- Metric: perplexity (test/val split), same as the GPT-2/WikiText-2 branch.

## Method (identical to the GPT-2/WikiText-2 branch, reused as-is)
- MLP (`mlp.c_fc`, `mlp.c_proj`) → 4-bit Product-VQ (K=256).
- Attention (`attn.c_attn`, `attn.c_proj`) → per-channel INT8.
- Embeddings (`wte`, `wpe`) → INT8.
- Codebook fine-tuned (best-of-3-seeds init, then best-checkpoint-guarded
  fine-tune) — this branch's `run_experiments.py`/`finetune_vq.py` are
  `--model`-parameterized so the exact same code runs both sizes.
- `quant_gpt2.py` is reused unmodified from the `gpt2-wikitext2-qatvq`
  branch (Conv1D layer surgery doesn't care about hidden size).

## What "success" looks like
Not necessarily QAT+VQ beating PTQ outright (it didn't on GPT-2 small
either) — the interesting result is whether the **gap to PTQ shrinks, holds,
or grows** as the model scales from 124M → 355M, and whether the
compression ratio stays similar. Either direction is a real, reportable
finding for a scaling section.

## Compute budget
Kaggle free tier: T4×2 (16GB each), 30h/week GPU quota. GPT-2-medium is
~3x the compute of GPT-2-small per step; batch size is halved accordingly
(bs=4 baseline/PTQ path, bs=2 for the codebook fine-tune) to fit VRAM.
Both models run sequentially in one Kaggle kernel — see `../kaggle_kernel_scaling/`.

## Files
```
gpt2_scaling/
  common_wt103.py      data (wikitext-103), perplexity eval, size accounting
                        -- parameterized by model_name, unlike gpt2/common_gpt2.py
  train_baseline.py    fine-tune gpt2 or gpt2-medium on wikitext-103
  run_experiments.py   baseline / PTQ / QAT / QAT+VQ (best-of-3-seeds), --model flag
  finetune_vq.py       codebook fine-tune, --model flag
  make_figures.py      per-model figures + the combined scaling comparison figure
  artifacts/
    gpt2/               results.json, figures/ for the 124M run
    gpt2-medium/        results.json, figures/ for the 355M run
    scaling_table.md    combined comparison table (both models, all methods)
    figures/scaling_comparison.png   the key deliverable
```
