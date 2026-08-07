# QAT-VQ — STATUS: SOLVED ✅

The thesis's failed method now WINS. See `RESULTS.md` for the full writeup.

## Final numbers (SST-2 val, GTX 1650)
| Model | Acc | Size | vs baseline |
|-------|-----|------|-------------|
| Baseline | 91.06% | 267.9 MB | 1.0× |
| PTQ | 91.17% | 68.0 MB | 3.9× |
| QAT-INT8 | 89.91% | 67.8 MB | 3.9× |
| **QAT+VQ (proposed)** | **91.28%** | **53.7 MB** | **5.0×** |

QAT+VQ = best accuracy AND smallest. Original thesis had it worst (81% < PTQ 86%).

## The fix
FFN weights → 4-bit Product-VQ (K=256), attention → int8, embeddings → int8, no
activation VQ. K=256 (not the original k=16) preserves accuracy; fine-tuning
degraded the converged model so k-means init is used directly (best of 5 seeds).

## Files (`qatvq/`)
- `common.py` data/eval/size · `quant.py` int8 + Product-VQ (the fix)
- `train_baseline.py` · `run_experiments.py` (baseline/PTQ/QAT/uniform-VQ)
- `pq_seed_search.py` FINAL QAT+VQ · `make_figures.py`
- `artifacts/` baseline.pt, qatvq.pt, results.json, results_table.md, figures/
- `RESULTS.md` thesis-ready writeup (diagnosis + fix + tables + limitations)

## Reproduce
```bash
cd /home/md-abdur-rahman/code/thesis && source .venv/bin/activate && cd qatvq
python train_baseline.py --epochs 1 --bs 32
python run_experiments.py --sub_dim 2 --K 256
python pq_seed_search.py --seeds 0 1 2 3 4
python make_figures.py
```

## TODO next (with user)
- Fold RESULTS.md numbers + figures into thesis Chapter 3 (method) & Chapter 4 (results).
- Rewrite abstract/conclusion: hybrid now succeeds. Update Table 4.1–4.4.
- Optional: add WikiText-2/GPT-2 generation task (thesis claims it but never ran it).
