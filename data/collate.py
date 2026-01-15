"""
Collate functions for Signal-Aware Data Parallel Training.
"""

import torch
from typing import Sequence


def collate_fn(batch: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate batch items into tensors.

    Since our dataset returns fixed-size tensors, collation is simply stacking.

    Args:
        batch: List of (input_ids, targets) tuples

    Returns:
        (input_ids, targets) both of shape (batch_size, seq_len)
    """
    input_ids = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return input_ids, targets
