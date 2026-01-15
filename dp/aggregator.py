"""
Gradient aggregation for Signal-Aware Data Parallel Training.

This module implements the custom DP gradient synchronization that replaces
the uniform average with weighted aggregation.

Key functions:
- sync_grads_uniform: Baseline uniform averaging (for comparison)
- sync_grads_weighted: Signal-aware weighted averaging (the innovation)
"""

from typing import Optional
import torch
import torch.distributed as dist
import torch.nn as nn


def sync_grads_uniform(
    model: nn.Module,
    world_size: int,
    process_group: Optional[dist.ProcessGroup] = None,
):
    """
    Synchronize gradients with uniform averaging (baseline).

    This is equivalent to what DDP does by default:
    g_global = (1/N) * sum(g_i)

    Args:
        model: Model with gradients to synchronize
        world_size: Number of workers
        process_group: Distributed process group (None for default)
    """
    if not dist.is_initialized():
        return

    # All-reduce each gradient tensor
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=process_group)
            p.grad.div_(world_size)


def sync_grads_weighted(
    model: nn.Module,
    weight_rank: float,
    process_group: Optional[dist.ProcessGroup] = None,
):
    """
    Synchronize gradients with weighted averaging.

    This is the key innovation:
    g_global = sum(w_i * g_i) where sum(w_i) = 1

    Each rank multiplies its gradient by its weight before all-reduce SUM.
    The result is automatically the weighted global gradient.

    Args:
        model: Model with gradients to synchronize
        weight_rank: This rank's weight (should sum to 1 across all ranks)
        process_group: Distributed process group (None for default)
    """
    if not dist.is_initialized():
        return

    # Scale local gradients by weight
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(weight_rank)

    # All-reduce SUM (weighted gradients sum to weighted global gradient)
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=process_group)


def get_global_grad_vector(model: nn.Module) -> Optional[torch.Tensor]:
    """
    Flatten all gradients into a single vector.

    Useful for computing gradient statistics or cosine similarity.

    Args:
        model: Model with gradients

    Returns:
        Flattened gradient vector or None if no gradients
    """
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.data.view(-1))

    if not grads:
        return None

    return torch.cat(grads)


def compute_grad_norm(model: nn.Module) -> float:
    """
    Compute total L2 norm of gradients.

    Args:
        model: Model with gradients

    Returns:
        Gradient norm
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


def clip_grad_norm(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """
    Clip gradients by global norm.

    Args:
        model: Model with gradients
        max_norm: Maximum gradient norm
        norm_type: Type of norm (default L2)

    Returns:
        Original gradient norm (before clipping)
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0

    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    return total_norm.item()


class GradientAggregator:
    """
    Stateful gradient aggregator that handles both uniform and weighted modes.

    This class encapsulates the gradient synchronization logic and provides
    a unified interface for both aggregation modes.
    """

    def __init__(
        self,
        model: nn.Module,
        world_size: int,
        agg_mode: str = "uniform",
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Args:
            model: Model to synchronize gradients for
            world_size: Number of workers
            agg_mode: Aggregation mode ("uniform" or "signal_weighted")
            process_group: Distributed process group
        """
        self.model = model
        self.world_size = world_size
        self.agg_mode = agg_mode
        self.process_group = process_group

    def sync_gradients(self, weights: Optional[torch.Tensor] = None, rank: int = 0):
        """
        Synchronize gradients according to aggregation mode.

        Args:
            weights: Weight tensor for weighted mode (shape [world_size])
            rank: Current process rank
        """
        if self.agg_mode == "uniform":
            sync_grads_uniform(self.model, self.world_size, self.process_group)
        elif self.agg_mode == "signal_weighted":
            if weights is None:
                raise ValueError("weights required for signal_weighted mode")
            weight_rank = weights[rank].item()
            sync_grads_weighted(self.model, weight_rank, self.process_group)
        else:
            raise ValueError(f"Unknown agg_mode: {self.agg_mode}")

    def get_grad_stats(self) -> dict:
        """
        Compute gradient statistics for logging.

        Returns:
            Dictionary with gradient statistics
        """
        grad_norm = compute_grad_norm(self.model)

        # Compute per-layer norms
        layer_norms = {}
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                layer_norms[f"grad_norm/{name}"] = p.grad.norm().item()

        return {
            "grad_norm_total": grad_norm,
            **layer_norms,
        }
