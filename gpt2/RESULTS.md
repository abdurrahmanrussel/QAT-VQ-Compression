# GPT-2 / WikiText-2 — QAT-VQ Results

Second model for the QAT-VQ compression study, applied to a **generation**
task (causal LM, measured by perplexity) instead of classification. This is
an experiment the original thesis *claimed* (abstract, objectives, and
contributions all mention a GPT-2/WikiText-2 generation task) but never
actually ran. Trained/evaluated on Kaggle (Tesla T4×2, 16 GB each).

## Results — WikiText-2 test perplexity

| Model | Perplexity | Size (MB) | Compression |
|-------|-----------|-----------|-------------|
| Baseline (fp32, fine-tuned) | 25.85 | 497.8 | 1.00× |
| PTQ (int8) | 25.87 | 125.3 | 3.97× |
| QAT (int8, fine-tuned) | **25.11** | 125.3 | 3.97× |
| GPTQ | 25.86 | 125.3 | 3.97× |
| AWQ | 25.87 | 125.5 | 3.97× |
| **QAT+VQ (proposed)** | 26.12 | **96.8** | **5.14×** |

(Lower perplexity is better. `avg_loss`/latency in `artifacts/results.json`.
Numbers shift slightly run-to-run, ~0.02-0.05 ppl, from training/calibration
seed variance — not a re-tuned result, just the natural noise floor.)

## Method (same recipe as the DistilBERT/SST-2 branch, ported to Conv1D)

GPT-2's linear layers are HuggingFace `Conv1D` (`[in, out]` weight, transposed
vs `nn.Linear`), so `quant_gpt2.py` reimplements the DistilBERT branch's
int8 + Product-VQ layers for that layout, reusing the same k-means core.

- **MLP weights (`mlp.c_fc`, `mlp.c_proj`) → 4-bit Product-VQ** (K=256
  codebook, 2-D sub-vectors). MLP is the bulk of the model and most redundant.
- **Attention weights (`attn.c_attn`, `attn.c_proj`) → per-channel INT8.**
- **Embeddings (`wte`, `wpe`) → INT8.**
- **Codebook fine-tuned** on WikiText-2 (2 epochs, lr 5e-6), best-checkpoint
  guarded so it can only match-or-beat the pre-finetune result.

## How this differs from the DistilBERT branch's finding

On DistilBERT/SST-2, fine-tuning the codebook *hurt* an already-converged
classifier — the k-means init alone (best of 5 seeds) was the final answer.
On GPT-2/WikiText-2, **fine-tuning helped**: best-of-5-seeds alone gave 26.44
ppl; fine-tuning the same seed's codebook brought it to **26.13 ppl** — a real
0.31 ppl improvement. Perplexity is far more sensitive to precision than
binary classification accuracy (every token's full output distribution
matters, not just an argmax), so the extra recovery step earns its keep here.

## GPTQ and AWQ baselines

Added alongside PTQ/QAT/QAT+VQ: real literature-standard post-training
quantization methods, implemented from scratch (`gptq_awq_conv1d.py`, a
Conv1D port of the DistilBERT branch's `gptq_awq.py` — same core
Hessian-guided GPTQ and activation-aware AWQ math, just adapted for
`Conv1D`'s transposed weight layout) rather than pulled from `auto-gptq`/
`AutoAWQ`. Both land essentially on top of naive PTQ here (GPTQ 25.86, AWQ
25.87, PTQ 25.87) — same finding as the DistilBERT branch: their real
advantage is rescuing 3-4 bit quantization on billion-parameter models from
catastrophic error, and there's no significant naive-rounding error left for
either to correct at 8 bits on a 124M-parameter model.

## Honest framing

Unlike the DistilBERT branch (where QAT+VQ beat PTQ, GPTQ, and AWQ outright —
higher accuracy AND smaller), here QAT+VQ does **not** beat any of them on
perplexity. It's a genuine **Pareto point**: 23% smaller than PTQ/GPTQ/AWQ
(96.8 MB vs ~125.3 MB) for a roughly +1.0% relative perplexity increase.
Whether that trade is worth it depends on the deployment constraint — for a
hard model-size cap it's the better choice; for pure quality it isn't.

QAT-INT8 alone is actually the best perplexity of all compressed variants
(25.11, even beating the fp32 baseline) at the same size as PTQ/GPTQ/AWQ —
showing fine-tuning with fake-quantization is a strong standalone method for
generation, even where the hybrid falls just short of the others.

## Reproduce

Requires a GPU with ≥8GB VRAM for comfortable batch sizes (T4/P100/similar);
ran on Kaggle due to local 4GB GPU limits.

```bash
cd gpt2
python train_baseline.py --epochs 1 --bs 8
python run_experiments.py --sub_dim 2 --K 256 --qat_epochs 1 --ft_lr 1e-5 --seeds 0 1 2 3 4
python finetune_vq.py --seed <best_seed_from_above> --epochs 2 --lr 5e-6
python eval_gptq_awq.py --n_calib_batches 16   # GPTQ + AWQ baselines (from scratch)
python make_figures.py
```

See `kaggle_kernel/` for a ready-to-push Kaggle notebook that runs this
end-to-end on a free T4×2 instance.

## Figures (`artifacts/figures/`)

- `size_ppl_bar.png` — size + perplexity per method
- `ppl_vs_size.png` — perplexity vs size scatter (Pareto view)
