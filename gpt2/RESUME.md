# GPT-2 / WikiText-2 branch — STATUS: DONE ✅

Ran end-to-end on Kaggle (Tesla T4×2, 16GB). See `RESULTS.md` for the full
writeup with honest framing.

## Final numbers (WikiText-2 test perplexity)
| Model | Perplexity | Size | Compression |
|-------|-----------|------|-------------|
| Baseline | 25.82 | 497.8 MB | 1.0× |
| PTQ | 25.84 | 125.3 MB | 3.97× |
| QAT-INT8 | **25.07** (best overall) | 125.3 MB | 3.97× |
| QAT+VQ | 26.13 | **96.8 MB** | **5.14×** |

QAT+VQ is a **Pareto point** here (23% smaller than PTQ, +1.1% ppl), not an
outright win like the DistilBERT branch. Codebook fine-tuning helped this
time (26.44 → 26.13), unlike DistilBERT where it hurt — perplexity is more
precision-sensitive than classification accuracy.

## What happened getting here (for future reference)
Local 4GB GPU OOM'd on GPT-2 training even after batch/seq tuning, so moved
to Kaggle. Hit a chain of setup issues worth remembering:
1. GitHub push token pasted in chat expired/was invalid — regenerated fine-grained PAT.
2. Kaggle kernel needs **phone verification** on the account for internet+GPU kernels.
3. `enable_gpu: true` in kernel-metadata.json is **deprecated** — does nothing.
   Must pass `--accelerator <name>` to `kaggle kernels push`.
4. Accelerator names are NOT what you'd guess (`P100`, `GPU_T4X2` all silently
   ignored, no error). Real values (found in `kagglesdk` source docstring):
   `NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`.
5. Kaggle's current preinstalled PyTorch (cu128) dropped support for P100
   (sm_60) — minimum supported is sm_70. **T4 (sm_75) works, P100 doesn't.**
6. The actual GPT-2 scripts (`common_gpt2.py`, `quant_gpt2.py`, etc.) had only
   ever been committed as `PLAN.md` — the real code was local-only. Every
   Kaggle run failed instantly with "No such file" until this was pushed.
7. `kaggle kernels output` only returns logs/files once a run reaches a
   terminal state (COMPLETE/ERROR) — can't peek mid-run.
8. The notebook's own git-push-back cell failed on a Kaggle-internal secrets
   service ConnectionError (unrelated to our code) — results were pulled back
   manually via `kaggle kernels output` instead.

## Files
- `common_gpt2.py`, `quant_gpt2.py` (Conv1D-aware int8 + Product-VQ)
- `train_baseline.py`, `run_experiments.py`, `finetune_vq.py`, `make_figures.py`
- `RESULTS.md` — thesis-ready writeup
- `artifacts/` results.json, results_table.md, figures/ (checkpoints gitignored)
- `../kaggle_kernel/` — the notebook + metadata used for the Kaggle run

## TODO next (with user)
- Fold RESULTS.md numbers + figures into thesis (new chapter or extend Ch.4
  with a "generation task" section, since this is genuinely new work vs the
  original thesis which never ran this experiment).
- Consider revoking/regenerating the GitHub PAT used for Kaggle (was pasted
  in chat during setup).
