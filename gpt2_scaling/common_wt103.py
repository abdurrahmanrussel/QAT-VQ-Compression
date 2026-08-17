"""Shared utilities for the scaling study: GPT-2 vs GPT-2-Medium on WikiText-103.

Mirrors ../gpt2/common_gpt2.py but parameterized by model name (so the same
code runs both model sizes) and points at WikiText-103 (bigger, more credible
benchmark than WikiText-2). Validation/test splits of WikiText-103 are
similar in size to WikiText-2's (~200-250k tokens), so only the *training*
text needs a practical cap for free-GPU compute budgets -- eval is on the
real, full WikiText-103 val/test data.
"""
import os
import time
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

SEQ_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return tok


def load_wikitext103(tokenizer, train_subset_chars=20_000_000):
    """Concatenate WikiText-103-raw text, chunk into fixed-length blocks.
    Train text is capped (default ~20M chars ~ 5M tokens ~ 20k blocks -- still
    ~2x WikiText-2's full train set) for compute budget on free-tier GPUs;
    validation/test use the FULL WikiText-103 splits (they're already a
    practical size, ~200-250k tokens each) so evaluation is on the real,
    harder benchmark.
    Returns (train_blocks, val_blocks, test_blocks) as tensors [N, SEQ_LEN].
    """
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    def chunk(split, cap_chars=None):
        text = "\n\n".join(t for t in ds[split]["text"] if t.strip())
        if cap_chars:
            text = text[:cap_chars]
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        n_blocks = len(ids) // SEQ_LEN
        ids = ids[: n_blocks * SEQ_LEN]
        return ids.view(n_blocks, SEQ_LEN)

    train = chunk("train", cap_chars=train_subset_chars)
    val = chunk("validation")
    test = chunk("test")
    return train, val, test


class BlockDataset(torch.utils.data.Dataset):
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return self.blocks.shape[0]

    def __getitem__(self, i):
        ids = self.blocks[i]
        return {"input_ids": ids, "labels": ids.clone()}


def make_loaders(train_blocks, val_blocks, bs_train=4, bs_eval=4):
    return (DataLoader(BlockDataset(train_blocks), batch_size=bs_train, shuffle=True),
            DataLoader(BlockDataset(val_blocks), batch_size=bs_eval))


@torch.no_grad()
def evaluate_ppl(model, loader, device=DEVICE):
    """Return dict: perplexity, avg loss, latency_ms/batch."""
    torch.cuda.empty_cache()
    model.eval()
    total_loss, total_tokens = 0.0, 0
    t0 = time.time()
    n_batches = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=ids, labels=labels)
        n_tok = ids.numel()
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
        n_batches += 1
    latency_ms = (time.time() - t0) / max(n_batches, 1) * 1000.0
    avg_loss = total_loss / total_tokens
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    return {"perplexity": ppl, "avg_loss": avg_loss, "latency_ms_per_batch": latency_ms}


def model_disk_size_mb(model):
    import tempfile
    path = tempfile.mktemp(suffix=".pt")
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1e6
    os.remove(path)
    return size


def print_report(name, metrics, size_mb):
    print(f"\n=== {name} ===")
    print(f"  perplexity : {metrics['perplexity']:.2f}")
    print(f"  size(MB)   : {size_mb:.2f}")
    print(f"  latency    : {metrics['latency_ms_per_batch']:.1f} ms/batch")
