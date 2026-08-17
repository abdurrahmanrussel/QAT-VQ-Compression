# GPT-2 Scaling Study — 124M vs 355M on WikiText-103

Third experiment in the QAT-VQ series, asking a question the other two
branches couldn't answer alone: **does the QAT+VQ compression/quality
trade-off get better or worse as the model scales up?** Same method, same
dataset (WikiText-103), two model sizes — directly comparable, unlike the
GPT-2/WikiText-2 branch which also changed the dataset alongside the task.
Ran end-to-end on Kaggle (Tesla T4×2).

## Headline finding: the trade-off improves with scale

| | GPT-2 (124M) | GPT-2-Medium (355M) | Direction |
|---|---|---|---|
| QAT+VQ perplexity gap vs PTQ | +1.68% | +1.30% | **narrows** |
| QAT+VQ size vs PTQ | 22.7% smaller | 28.4% smaller | **improves** |
| QAT+VQ compression vs baseline | 5.14× | 5.55× | **improves** |

At 355M params, QAT+VQ gets *both* a smaller relative accuracy cost *and* a
better compression ratio than at 124M. This is the direction you want to see
for a compression method aimed at larger, more practically-deployed models —
the bigger the model, the more redundancy there is for Product-VQ to exploit,
while the (small, fixed) codebook overhead matters proportionally less.

## Full results

**GPT-2 (124M), WikiText-103:**

| Model | Perplexity | Size (MB) | Compression |
|-------|-----------|-----------|-------------|
| Baseline | 24.33 | 497.8 | 1.00× |
| PTQ | 24.35 | 125.3 | 3.97× |
| QAT-INT8 | 24.14 | 125.3 | 3.97× |
| QAT+VQ | 24.76 | 96.8 | 5.14× |

**GPT-2-Medium (355M), WikiText-103:**

| Model | Perplexity | Size (MB) | Compression |
|-------|-----------|-----------|-------------|
| Baseline | 18.40 | 1419.4 | 1.00× |
| PTQ | 18.41 | 356.7 | 3.98× |
| QAT-INT8 | 18.34 | 356.7 | 3.98× |
| QAT+VQ | 18.65 | 255.6 | 5.55× |

(Both models see much lower perplexity than the WikiText-2 branch's ~25-26
range — WikiText-103's larger, more varied training text, even capped to a
~5M-token subset, gives a stronger starting point than WikiText-2's full
~2.4M-token train set.)

## Method (identical at both scales)

- MLP weights → 4-bit Product-VQ (K=256 codebook, 2-D sub-vectors), best-of-3
  seeds, then codebook fine-tuned (best-checkpoint guarded).
- Attention weights → per-channel INT8.
- Embeddings → INT8.
- `quant_gpt2.py` is reused completely unmodified from the GPT-2/WikiText-2
  branch — Conv1D layer surgery doesn't care about hidden size, which is
  exactly what makes this a clean scaling comparison rather than a new
  implementation each time.

Codebook fine-tuning helped at both scales (matches the WikiText-2 branch's
finding, not the DistilBERT branch's): GPT-2-Medium's QAT+VQ improved from
19.03 ppl (pre-finetune, best seed) to 18.65 ppl over 2 fine-tune epochs.

## Honest framing

QAT+VQ still does not beat PTQ outright at either scale (same as the
WikiText-2 branch) — it remains a Pareto point, not a win. What's new here is
evidence that the point moves in the *right* direction as models get bigger,
which is the more important claim for real-world relevance: practical
deployment targets are 1B+ parameter models, not 124M ones, and this trend
suggests the method's value proposition strengthens exactly where it matters
most. Two data points is a trend, not a proof — a third size (e.g. GPT-2-Large,
774M, or a ~1B model) would make this a stronger claim.

## Reproduce

```bash
cd gpt2_scaling
python train_baseline.py --model gpt2 --epochs 1 --bs 8
python run_experiments.py --model gpt2 --sub_dim 2 --K 256 --seeds 0 1 2
python finetune_vq.py --model gpt2 --seed <best> --epochs 2 --lr 5e-6

python train_baseline.py --model gpt2-medium --epochs 1 --bs 4 --grad_accum 2
python run_experiments.py --model gpt2-medium --sub_dim 2 --K 256 --seeds 0 1 2 --bs_train 4 --bs_eval 4
python finetune_vq.py --model gpt2-medium --seed <best> --epochs 2 --lr 5e-6 --bs_train 2 --bs_eval 2

python make_figures.py   # per-model figures + combined scaling_comparison.png
```

See `../kaggle_kernel_scaling/` for the Kaggle notebook that ran both sizes
sequentially in one session.

## Figures

- `artifacts/figures/scaling_comparison.png` — the key deliverable: size and
  relative-perplexity-cost by method, side by side across both model sizes.
- `artifacts/gpt2/figures/ppl_vs_size.png`, `artifacts/gpt2-medium/figures/ppl_vs_size.png`
  — per-model Pareto scatter.
