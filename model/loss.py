"""
Loss functions for Signal-Aware Data Parallel Training.
"""

import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute causal language modeling loss.

    Args:
        logits: Model output logits, shape (batch, seq_len, vocab_size)
        targets: Target token indices, shape (batch, seq_len)
        ignore_index: Index to ignore in loss computation (e.g., padding)
        reduction: Loss reduction method ("mean", "sum", "none")

    Returns:
        Scalar loss (or per-token loss if reduction="none")
    """
    # Flatten for cross entropy
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction=reduction,
    )

    return loss


def per_sample_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Compute per-sample loss for analysis.

    Args:
        logits: Model output logits, shape (batch, seq_len, vocab_size)
        targets: Target token indices, shape (batch, seq_len)
        ignore_index: Index to ignore in loss computation

    Returns:
        Loss per sample, shape (batch,)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Compute per-token loss
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    per_token_loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction="none",
    )

    # Reshape and average per sample
    per_token_loss = per_token_loss.view(batch_size, seq_len)

    # Mask ignored tokens
    mask = targets != ignore_index
    masked_loss = per_token_loss * mask

    # Average over sequence (accounting for varying valid lengths)
    per_sample = masked_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return per_sample
