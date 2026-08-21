| Model | Method | Perplexity | Size (MB) | Compression | Δppl vs Baseline |
|-------|--------|-----------|-----------|-------------|-------------------|
| GPT-2 (124M) | Baseline | 24.33 | 497.8 | 1.00x | +0.0% |
| GPT-2 (124M) | PTQ | 24.35 | 125.3 | 3.97x | +0.1% |
| GPT-2 (124M) | QAT-INT8 | 24.14 | 125.3 | 3.97x | -0.8% |
| GPT-2 (124M) | QAT+VQ | 24.76 | 96.8 | 5.14x | +1.7% |
| GPT-2-Medium (355M) | Baseline | 18.40 | 1419.4 | 1.00x | +0.0% |
| GPT-2-Medium (355M) | PTQ | 18.41 | 356.7 | 3.98x | +0.1% |
| GPT-2-Medium (355M) | QAT-INT8 | 18.34 | 356.7 | 3.98x | -0.3% |
| GPT-2-Medium (355M) | QAT+VQ | 18.65 | 255.6 | 5.55x | +1.3% |
| GPT-2-Large (774M) | Baseline | 16.62 | 3096.3 | 1.00x | +0.0% |
| GPT-2-Large (774M) | PTQ | 16.62 | 777.3 | 3.98x | +0.0% |
| GPT-2-Large (774M) | QAT-INT8 | 17.35 | 777.3 | 3.98x | +4.4% |
| GPT-2-Large (774M) | QAT+VQ | 17.10 | 540.5 | 5.73x | +2.9% |
