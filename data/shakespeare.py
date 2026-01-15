"""
TinyShakespeare dataset for Signal-Aware Data Parallel Training.

Simple, fast, and small - perfect for testing the signal-aware DP concept.
~300k train tokens, ~36k val tokens.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Try tiktoken first (faster), fall back to transformers
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent


def download_shakespeare() -> str:
    """Download TinyShakespeare if not present, return the text."""
    import requests

    data_dir = get_data_dir()
    input_file = data_dir / "input.txt"

    if not input_file.exists():
        print(f"Downloading TinyShakespeare to {input_file}...")
        response = requests.get(DATA_URL)
        response.raise_for_status()
        input_file.write_text(response.text, encoding="utf-8")
        print("Download complete.")

    return input_file.read_text(encoding="utf-8")


def prepare_shakespeare(force: bool = False) -> tuple[Path, Path]:
    """
    Prepare TinyShakespeare train/val binary files.

    Returns:
        (train_bin_path, val_bin_path)
    """
    data_dir = get_data_dir()
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"

    if train_bin.exists() and val_bin.exists() and not force:
        return train_bin, val_bin

    # Download and load text
    data = download_shakespeare()
    n = len(data)

    # 90/10 train/val split
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # Encode with tiktoken GPT-2 BPE
    if HAS_TIKTOKEN:
        enc = tiktoken.get_encoding("gpt2")
        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        train_ids = tokenizer.encode(train_data, add_special_tokens=False)
        val_ids = tokenizer.encode(val_data, add_special_tokens=False)

    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")

    # Save as binary
    np.array(train_ids, dtype=np.uint16).tofile(train_bin)
    np.array(val_ids, dtype=np.uint16).tofile(val_bin)

    return train_bin, val_bin


def load_shakespeare_tokens(split: str = "train") -> torch.Tensor:
    """
    Load pre-tokenized Shakespeare data.

    Args:
        split: "train" or "val"

    Returns:
        1D tensor of token IDs
    """
    train_bin, val_bin = prepare_shakespeare()
    bin_path = train_bin if split == "train" else val_bin

    # Load from binary
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    tokens = torch.from_numpy(tokens.astype(np.int64))

    return tokens


class ShakespeareDataset(torch.utils.data.Dataset):
    """
    Simple Shakespeare dataset for causal LM training.
    """

    def __init__(self, split: str, seq_len: int):
        """
        Args:
            split: "train" or "val"
            seq_len: Sequence length for each sample
        """
        self.tokens = load_shakespeare_tokens(split)
        self.seq_len = seq_len
        self.n_sequences = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.tokens[start:end]
        input_ids = chunk[:-1]
        targets = chunk[1:]

        return input_ids, targets


def create_shakespeare_dataloaders(
    cfg,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """
    Create train and validation dataloaders for Shakespeare.

    This is a simpler alternative to the WikiText dataloaders.
    """
    from data.sharding import make_rank_dataset
    from data.collate import collate_fn

    # Prepare data
    prepare_shakespeare()

    # Load tokens
    train_tokens = load_shakespeare_tokens("train")
    val_tokens = load_shakespeare_tokens("val")

    # Apply rank-specific corruption
    train_tokens = make_rank_dataset(train_tokens, cfg.corruption, rank, world_size)

    # Create datasets
    train_dataset = ShakespeareDataset.__new__(ShakespeareDataset)
    train_dataset.tokens = train_tokens
    train_dataset.seq_len = cfg.data.seq_len
    train_dataset.n_sequences = (len(train_tokens) - 1) // cfg.data.seq_len

    val_dataset = ShakespeareDataset("val", cfg.data.seq_len)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # GPT-2 vocab size
    vocab_size = 50257

    return train_loader, val_loader, vocab_size
