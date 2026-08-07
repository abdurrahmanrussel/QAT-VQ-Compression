#!/usr/bin/env bash
# Full overnight pipeline: baseline -> all compression variants -> figures.
# Unbuffered, timestamped, single log. Stops on first error.
set -eo pipefail
cd /home/md-abdur-rahman/code/thesis
source .venv/bin/activate
cd qatvq
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

ts() { date '+%H:%M:%S'; }
step() { echo ""; echo "########## [$(ts)] $1 ##########"; }

step "STEP 1/3  baseline fine-tune (1 epoch, bs32, seq64)"
python train_baseline.py --epochs 1 --bs 32

step "STEP 2/3  compression variants (PTQ / QAT / QAT+VQ) + verdict"
python run_experiments.py --sub_dim 2 --K 256 --qat_epochs 1 --vq_epochs 2 --vq_lr 2e-5

step "STEP 3/3  figures + comparison table"
python make_figures.py

step "ALL DONE"
echo "results table:"
cat artifacts/results_table.md
echo "PIPELINE_COMPLETE_OK"
