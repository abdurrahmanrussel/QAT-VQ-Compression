# GPT-2 / WikiText-2 branch — resume point

Branch: `gpt2-wikitext2-qatvq`. GPU freed, nothing running.

## Results so far (SST-2-style pipeline, ported to generation)
| Model | Perplexity | Size (MB) | Compression |
|-------|-----------|-----------|-------------|
| Baseline | 25.40 | 497.8 | 1.0× |
| PTQ | 25.43 | 125.3 | 3.97× |
| QAT-INT8 | 24.95 | 125.3 | 3.97× |
| QAT+VQ (best-of-5-seeds, no finetune) | 26.44 | 96.8 | 5.14× |

Unlike the DistilBERT branch, seed selection alone left a real gap (all 5 seeds
landed 26.44-26.58, i.e. systematic, not variance) — perplexity is more
precision-sensitive than classification accuracy. Honest framing: QAT+VQ is a
**Pareto point** (23% smaller than PTQ, +4% ppl), not yet a clean win.

## In progress when paused
Was running `finetune_vq.py --seed 1 --epochs 2 --lr 5e-6` — fine-tunes the
PQConv1D codebook (already trainable) with a best-checkpoint guard (can only
match-or-beat 26.44, never worse). Killed at step 400/2359 epoch 1, **no
checkpoint saved yet** — rerun from scratch.

## To resume
```bash
cd /home/md-abdur-rahman/code/thesis && source .venv/bin/activate && cd gpt2
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python finetune_vq.py --seed 1 --epochs 2 --lr 5e-6
```
Takes ~15-20 min on GTX 1650. Watch for `FINAL VERDICT` block at the end —
compares fine-tuned QAT+VQ vs PTQ (25.43 ppl / 125.3 MB).

If fine-tune still doesn't close the gap: that's fine — write it up honestly as
a Pareto point (smaller-for-slightly-worse), which is still a valid, defensible
result for the thesis. Then:
```bash
python make_figures.py     # perplexity vs size, compression bar
```

## Then: figures + RESULTS.md + commit/push
Mirror the DistilBERT branch's `RESULTS.md` — diagnosis (N/A here, this is a
new experiment not a fix), method, table, honest caveats. Commit + push to
`gpt2-wikitext2-qatvq`.

## Files
- `common_gpt2.py`, `quant_gpt2.py` (Conv1D-aware, reuses `../qatvq/quant.py` kmeans)
- `train_baseline.py`, `run_experiments.py`, `finetune_vq.py`, `make_figures.py`
- `artifacts/baseline.pt`, `results.json`, logs
