"""
Non-IID data sharding for Signal-Aware Data Parallel Training.

This module implements the core experimental setup: creating different
quality data shards for different workers to test signal-aware weighting.
"""

import random
from typing import Optional
import torch

from config import CorruptionConfig


def make_rank_dataset(
    base_tokens: torch.Tensor,
    cfg: CorruptionConfig,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """
    Create rank-specific token stream based on non-IID configuration.

    Args:
        base_tokens: Original token stream
        cfg: Corruption configuration
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        Token stream for this rank (may be corrupted)
    """
    if cfg.non_iid_mode == "iid":
        # IID mode: all ranks get the same clean data
        # Sharding happens via different batch sampling
        return base_tokens.clone()

    elif cfg.non_iid_mode == "clean_vs_corrupt":
        # Clean vs corrupt: one rank gets corrupted data
        if rank == cfg.corrupt_rank:
            return apply_corruption(base_tokens, cfg)
        else:
            return base_tokens.clone()

    elif cfg.non_iid_mode == "dup_skew":
        # Duplication skew: one rank gets highly duplicated data
        if rank == cfg.corrupt_rank:
            return apply_duplication(base_tokens, cfg.dup_factor)
        else:
            return base_tokens.clone()

    else:
        raise ValueError(f"Unknown non_iid_mode: {cfg.non_iid_mode}")


def apply_corruption(
    tokens: torch.Tensor,
    cfg: CorruptionConfig,
) -> torch.Tensor:
    """
    Apply corruption to token stream.

    This is where we simulate "junk" data that should be downweighted.

    Args:
        tokens: Original token stream
        cfg: Corruption configuration

    Returns:
        Corrupted token stream
    """
    corruption_fn = {
        "token_swap": _token_swap_corruption,
        "span_drop": _span_drop_corruption,
        "random_insert": _random_insert_corruption,
        "duplicate": _duplicate_corruption,
    }.get(cfg.corruption_type)

    if corruption_fn is None:
        raise ValueError(f"Unknown corruption_type: {cfg.corruption_type}")

    return corruption_fn(tokens, cfg.corruption_prob)


def _token_swap_corruption(
    tokens: torch.Tensor,
    prob: float,
    vocab_size: int = 50257,  # GPT-2 vocab size
    special_tokens: set = {0, 50256},  # BOS, EOS
) -> torch.Tensor:
    """
    Replace tokens with random tokens with probability prob.

    This simulates data where random noise has been introduced.

    Args:
        tokens: Original tokens
        prob: Probability of replacing each token
        vocab_size: Vocabulary size for random sampling
        special_tokens: Token IDs to never corrupt

    Returns:
        Corrupted tokens
    """
    corrupted = tokens.clone()
    n_tokens = len(tokens)

    # Generate mask of which tokens to corrupt
    mask = torch.rand(n_tokens) < prob

    # Don't corrupt special tokens
    for special in special_tokens:
        mask = mask & (tokens != special)

    # Count corrupted tokens
    n_corrupt = mask.sum().item()

    # Replace with random tokens
    corrupted[mask] = torch.randint(0, vocab_size, (n_corrupt,))

    return corrupted


def _span_drop_corruption(
    tokens: torch.Tensor,
    prob: float,
    min_span: int = 3,
    max_span: int = 20,
) -> torch.Tensor:
    """
    Drop random spans of tokens, then repack to maintain length.

    This simulates data with missing context.

    Args:
        tokens: Original tokens
        prob: Probability of starting a span drop
        min_span: Minimum span length
        max_span: Maximum span length

    Returns:
        Corrupted tokens (same length, gaps filled by shifting)
    """
    tokens_list = tokens.tolist()
    result = []
    i = 0

    while i < len(tokens_list):
        if random.random() < prob:
            # Drop a span
            span_len = random.randint(min_span, max_span)
            i += span_len
        else:
            result.append(tokens_list[i])
            i += 1

    # Pad back to original length by repeating (simulates sparse data)
    original_len = len(tokens_list)
    if len(result) < original_len:
        # Repeat the sequence to fill
        while len(result) < original_len:
            result.extend(result[:original_len - len(result)])
        result = result[:original_len]

    return torch.tensor(result, dtype=tokens.dtype)


def _random_insert_corruption(
    tokens: torch.Tensor,
    prob: float,
    vocab_size: int = 50257,
) -> torch.Tensor:
    """
    Insert random tokens, then crop to maintain length.

    This simulates data with inserted garbage.

    Args:
        tokens: Original tokens
        prob: Probability of inserting after each token
        vocab_size: Vocabulary size for random sampling

    Returns:
        Corrupted tokens (same length)
    """
    tokens_list = tokens.tolist()
    result = []

    for tok in tokens_list:
        result.append(tok)
        if random.random() < prob:
            # Insert random token
            result.append(random.randint(0, vocab_size - 1))

    # Crop to original length
    return torch.tensor(result[: len(tokens_list)], dtype=tokens.dtype)


def _duplicate_corruption(
    tokens: torch.Tensor,
    prob: float,
    min_window: int = 50,
    max_window: int = 200,
) -> torch.Tensor:
    """
    Duplicate random windows of tokens.

    This simulates duplicated/redundant data that provides less signal.

    Args:
        tokens: Original tokens
        prob: Probability of duplicating each window
        min_window: Minimum window size
        max_window: Maximum window size

    Returns:
        Corrupted tokens with duplicated sections
    """
    tokens_list = tokens.tolist()
    result = []
    i = 0

    while i < len(tokens_list):
        window_size = random.randint(min_window, max_window)
        window = tokens_list[i : i + window_size]

        result.extend(window)

        if random.random() < prob:
            # Duplicate this window
            result.extend(window)

        i += window_size

    # Crop to original length
    return torch.tensor(result[: len(tokens_list)], dtype=tokens.dtype)


def apply_duplication(
    tokens: torch.Tensor,
    dup_factor: int,
) -> torch.Tensor:
    """
    Create heavily duplicated version of tokens.

    This simulates a shard with low diversity data.

    Args:
        tokens: Original tokens
        dup_factor: How many times to repeat chunks

    Returns:
        Token stream with repeated chunks
    """
    # Split into chunks and repeat each
    chunk_size = len(tokens) // (dup_factor * 2)
    result = []

    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i : i + chunk_size].tolist()
        # Repeat this chunk
        for _ in range(dup_factor):
            result.extend(chunk)

    # Crop to original length
    return torch.tensor(result[: len(tokens)], dtype=tokens.dtype)


def describe_corruption(cfg: CorruptionConfig, rank: int) -> str:
    """
    Return a human-readable description of what corruption is applied to this rank.

    Args:
        cfg: Corruption configuration
        rank: Process rank

    Returns:
        Description string
    """
    if cfg.non_iid_mode == "iid":
        return "IID (no corruption)"
    elif cfg.non_iid_mode == "clean_vs_corrupt":
        if rank == cfg.corrupt_rank:
            return f"CORRUPTED ({cfg.corruption_type}, prob={cfg.corruption_prob})"
        else:
            return "CLEAN"
    elif cfg.non_iid_mode == "dup_skew":
        if rank == cfg.corrupt_rank:
            return f"DUPLICATED (factor={cfg.dup_factor})"
        else:
            return "CLEAN"
    return "UNKNOWN"
