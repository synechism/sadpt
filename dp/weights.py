"""
Weight computation for Signal-Aware Data Parallel Training.

Turn gathered per-rank signals into stable weights for gradient aggregation.
This is where signals are transformed into actionable trust scores.
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F


@dataclass
class WeightState:
    """State for weight computation with smoothing."""

    weights: Optional[torch.Tensor] = None
    step_count: int = 0

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            "weights": self.weights.cpu() if self.weights is not None else None,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict, device: torch.device = None):
        """Load state from checkpoint."""
        if state["weights"] is not None:
            self.weights = state["weights"]
            if device is not None:
                self.weights = self.weights.to(device)
        self.step_count = state["step_count"]


def compute_weights(
    signals: torch.Tensor,
    temp: float = 1.0,
    w_min: float = 0.1,
    w_max: float = 0.9,
    ema_beta: float = 0.9,
    state: Optional[WeightState] = None,
) -> tuple[torch.Tensor, WeightState]:
    """
    Convert signals to weights with smoothing and clamping.

    The weight computation pipeline:
    1. Normalize signals (zero mean, unit variance)
    2. Apply softmax with temperature
    3. Clamp to [w_min, w_max] range
    4. Renormalize to sum to 1
    5. EMA smooth with previous weights

    Args:
        signals: Signal values from all ranks, shape [world_size]
        temp: Softmax temperature (higher = more uniform)
        w_min: Minimum weight per worker (prevent zeroing)
        w_max: Maximum weight per worker (prevent collapse)
        ema_beta: EMA coefficient for weight smoothing
        state: Previous weight state for smoothing

    Returns:
        (weights, updated_state) where weights sum to 1
    """
    if state is None:
        state = WeightState()

    world_size = signals.shape[0]
    device = signals.device

    # Step 1: Normalize signals
    signals_normalized = normalize_signals(signals)

    # Step 2: Apply softmax with temperature
    weights = F.softmax(signals_normalized / temp, dim=0)

    # Step 3: Clamp to bounds
    weights = weights.clamp(min=w_min, max=w_max)

    # Step 4: Renormalize to sum to 1
    weights = weights / weights.sum()

    # Step 5: EMA smooth with previous weights
    if state.weights is not None and state.step_count > 0:
        weights = ema_beta * state.weights + (1 - ema_beta) * weights
        # Renormalize after EMA (in case of numerical drift)
        weights = weights / weights.sum()

    # Update state
    state.weights = weights.clone()
    state.step_count += 1

    return weights, state


def normalize_signals(signals: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize signals to zero mean and unit variance.

    Args:
        signals: Raw signal values
        eps: Small constant for numerical stability

    Returns:
        Normalized signals
    """
    # Handle single-element case (single worker)
    if signals.numel() <= 1:
        return torch.zeros_like(signals)

    mean = signals.mean()
    std = signals.std(unbiased=False)  # Use biased std to avoid NaN with small N

    if std < eps:
        # All signals are the same - return zeros (will result in uniform weights)
        return torch.zeros_like(signals)

    return (signals - mean) / (std + eps)


def compute_uniform_weights(world_size: int, device: torch.device) -> torch.Tensor:
    """
    Return uniform weights (baseline).

    Args:
        world_size: Number of workers
        device: Device for tensor

    Returns:
        Uniform weights tensor
    """
    return torch.ones(world_size, device=device) / world_size


def analyze_weight_distribution(weights: torch.Tensor) -> dict:
    """
    Compute statistics about weight distribution for logging.

    Args:
        weights: Weight tensor

    Returns:
        Dictionary with distribution statistics
    """
    return {
        "weight_min": weights.min().item(),
        "weight_max": weights.max().item(),
        "weight_std": weights.std().item(),
        "weight_entropy": -(weights * (weights + 1e-10).log()).sum().item(),
        "weight_effective_n": 1.0 / (weights ** 2).sum().item(),  # Effective number of workers
    }
