# GPT-2 / WikiText-2 — QAT+VQ compression (branch gpt2-wikitext2-qatvq)

Second model, same compression idea as the DistilBERT branch, applied to a
**generation** task the thesis claims but never ran.

## Setup
- Model: `gpt2` (124M). Fits GTX 1650 4 GB at seq 256, small batch + grad accum.
- Data: `wikitext` / `wikitext-2-raw-v1`. Task: causal language modelling.
- Metric: **perplexity** (exp of mean token NLL) on the test split — plus model
  size (MB) and compression ratio. (No accuracy/AUC — this is generation.)

## Variants (mirror DistilBERT branch)
| Variant | Weights |
|---------|---------|
| Baseline | fp32/fp16, fine-tuned on WikiText-2 |
| PTQ | int8 all linears + embeddings |
| QAT | int8 fake-quant, fine-tuned |
| **QAT+VQ** | MLP `c_fc`/`c_proj` → 4-bit Product-VQ (K=256); attention `c_attn`/`c_proj` → int8; `wte`/`wpe` → int8 |

## Key adaptation vs DistilBERT
GPT-2 linear layers are HuggingFace **`Conv1D`**, weight shape `[in_features,
out_features]` (transposed vs `nn.Linear` `[out,in]`). So:
- `quant.py` surgery + PQ/int8 must handle `Conv1D` (transpose or a dedicated
  wrapper). Product-VQ splits along the `in` dimension = dim 0 here.
- Layer name filters: MLP = `mlp.c_fc`, `mlp.c_proj`; attention = `attn.c_attn`,
  `attn.c_proj`; embeddings = `wte`, `wpe`.

## Expected story
QAT+VQ should retain perplexity close to baseline at ~5× compression, showing
the method generalises from classification (SST-2) to generation (WikiText-2) —
the cross-task claim the thesis makes but never demonstrated.

## Files (to build)
```
gpt2/
  common_gpt2.py       data (wikitext-2), perplexity eval, size accounting
  quant_gpt2.py        Conv1D-aware int8 + Product-VQ (reuses qatvq/quant.py kmeans)
  train_baseline.py    fine-tune gpt2 on wikitext-2
  run_experiments.py   baseline / PTQ / QAT / QAT+VQ + perplexity table
  make_figures.py      perplexity vs size, compression bar
```
