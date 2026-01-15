"""
Dataset utilities for Signal-Aware Data Parallel Training.
Loads text, tokenizes it, and yields fixed-length causal-LM sequences.
"""

from typing import Optional
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


def load_raw_text(dataset_name: str, dataset_config: str, split: str = "train"):
    """
    Load raw text dataset from HuggingFace datasets.

    Args:
        dataset_name: Dataset name (e.g., "wikitext")
        dataset_config: Dataset configuration (e.g., "wikitext-103-raw-v1")
        split: Dataset split ("train", "validation", "test")

    Returns:
        HuggingFace dataset object
    """
    ds = load_dataset(dataset_name, dataset_config, split=split)
    return ds


def build_token_stream(
    ds,
    tokenizer,
    text_column: str = "text",
    max_tokens: Optional[int] = None,
    chunk_size: int = 10000,
) -> torch.Tensor:
    """
    Tokenize dataset and return a single 1D token stream.

    Args:
        ds: HuggingFace dataset
        tokenizer: Tokenizer object
        text_column: Name of text column in dataset
        max_tokens: Maximum number of tokens to return (None = all)
        chunk_size: Number of examples to process at once (memory optimization)

    Returns:
        1D tensor of token IDs
    """
    all_tokens = []
    eos_token_id = tokenizer.eos_token_id

    for i in range(0, len(ds), chunk_size):
        chunk = ds[i : i + chunk_size]
        texts = chunk[text_column]

        # Filter empty texts
        texts = [t for t in texts if t.strip()]

        for text in texts:
            # Tokenize with EOS separator
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if tokens:
                all_tokens.extend(tokens)
                all_tokens.append(eos_token_id)

            # Early exit if we have enough tokens
            if max_tokens is not None and len(all_tokens) >= max_tokens:
                all_tokens = all_tokens[:max_tokens]
                return torch.tensor(all_tokens, dtype=torch.long)

    return torch.tensor(all_tokens, dtype=torch.long)


class CausalLMDataset(Dataset):
    """
    Dataset for causal language modeling.

    Takes a token stream and yields fixed-length sequences with
    input_ids and targets (next token shifted).
    """

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        """
        Args:
            tokens: 1D tensor of token IDs
            seq_len: Sequence length for each sample
        """
        self.tokens = tokens
        self.seq_len = seq_len

        # Number of complete sequences we can form
        # Need seq_len + 1 tokens for each sample (input + 1 for target)
        self.n_sequences = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            (input_ids, targets) both of shape (seq_len,)
        """
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for the target of last position

        chunk = self.tokens[start:end]
        input_ids = chunk[:-1]  # First seq_len tokens
        targets = chunk[1:]  # Shifted by 1

        return input_ids, targets


def get_tokenizer(tokenizer_name: str = "gpt2"):
    """
    Load and configure tokenizer.

    Args:
        tokenizer_name: Name of pretrained tokenizer

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Ensure we have a pad token (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def create_dataloaders(
    cfg,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """
    Create train and validation dataloaders.

    Args:
        cfg: Configuration object
        rank: Process rank
        world_size: Total number of processes
        device: Device to load data to

    Returns:
        (train_loader, val_loader, tokenizer)
    """
    from data.sharding import make_rank_dataset
    from data.collate import collate_fn

    # Load tokenizer
    tokenizer = get_tokenizer(cfg.data.tokenizer_name)

    # Update vocab size in model config
    cfg.model.vocab_size = len(tokenizer)

    # Load and tokenize training data
    train_ds = load_raw_text(
        cfg.data.dataset_name,
        cfg.data.dataset_config,
        split="train",
    )
    train_tokens = build_token_stream(
        train_ds,
        tokenizer,
        max_tokens=cfg.data.max_train_tokens,
    )

    # Apply rank-specific corruption/sharding
    train_tokens = make_rank_dataset(train_tokens, cfg.corruption, rank, world_size)

    # Create dataset
    train_dataset = CausalLMDataset(train_tokens, cfg.data.seq_len)

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,  # Keep it simple for now
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Load validation data (same across all ranks, no corruption)
    val_ds = load_raw_text(
        cfg.data.dataset_name,
        cfg.data.dataset_config,
        split="validation",
    )
    val_tokens = build_token_stream(val_ds, tokenizer)
    val_dataset = CausalLMDataset(val_tokens, cfg.data.seq_len)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return train_loader, val_loader, tokenizer
