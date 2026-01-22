"""
Signal computation for Signal-Aware Data Parallel Training.

Compute per-rank scalar "signals" each step that reflect gradient quality / shard quality.
These signals are used to determine how much to trust each worker's gradients.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class BaseSignal(ABC):
    """Base class for signal computation."""

    @abstractmethod
    def update(self, **kwargs) -> float:
        """
        Update signal state and return current signal value.

        Higher signal = more trustworthy gradients.

        Returns:
            Signal value (higher is better)
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset signal state."""
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        pass

    @abstractmethod
    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        pass


class LossEMASignal(BaseSignal):
    """
    Signal based on exponential moving average of loss.

    Simple and robust: lower loss EMA = higher signal (better gradients).

    The intuition is that workers with consistently lower loss are seeing
    cleaner data and producing more useful gradients.
    """

    def __init__(self, beta: float = 0.99, eps: float = 1e-8):
        """
        Args:
            beta: EMA smoothing factor (higher = more smoothing)
            eps: Small constant for numerical stability
        """
        self.beta = beta
        self.eps = eps
        self.ema_loss: Optional[float] = None
        self.step_count = 0

    def update(self, loss: float, **kwargs) -> float:
        """
        Update loss EMA and return signal.

        Args:
            loss: Current step loss value

        Returns:
            Signal value (negative EMA loss, so higher is better)
        """
        self.step_count += 1

        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = self.beta * self.ema_loss + (1 - self.beta) * loss

        # Return negative EMA loss (higher signal = lower loss = better)
        # Add small constant to avoid issues with very small losses
        signal = -self.ema_loss

        return signal

    def reset(self):
        """Reset signal state."""
        self.ema_loss = None
        self.step_count = 0

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            "ema_loss": self.ema_loss,
            "step_count": self.step_count,
            "beta": self.beta,
        }

    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.ema_loss = state["ema_loss"]
        self.step_count = state["step_count"]
        self.beta = state.get("beta", self.beta)


class GradNormStabilitySignal(BaseSignal):
    """
    Signal based on gradient norm stability.

    Tracks EMA of gradient norm and its variance.
    High mean / low variance = stable, informative gradients.

    signal = E[||g||] / (Var(||g||) + eps)
    """

    def __init__(self, beta: float = 0.99, eps: float = 1e-8):
        """
        Args:
            beta: EMA smoothing factor
            eps: Small constant for numerical stability
        """
        self.beta = beta
        self.eps = eps
        self.ema_norm: Optional[float] = None
        self.ema_norm_sq: Optional[float] = None
        self.step_count = 0

    def update(self, model: nn.Module = None, grad_norm: float = None, **kwargs) -> float:
        """
        Update gradient norm statistics and return signal.

        Args:
            model: Model to compute gradient norm from (if grad_norm not provided)
            grad_norm: Pre-computed gradient norm (optional)

        Returns:
            Signal value (stability measure, higher is better)
        """
        self.step_count += 1

        # Compute gradient norm if not provided
        if grad_norm is None:
            if model is None:
                raise ValueError("Must provide either model or grad_norm")
            grad_norm = self._compute_grad_norm(model)

        # Update EMAs
        if self.ema_norm is None:
            self.ema_norm = grad_norm
            self.ema_norm_sq = grad_norm ** 2
        else:
            self.ema_norm = self.beta * self.ema_norm + (1 - self.beta) * grad_norm
            self.ema_norm_sq = self.beta * self.ema_norm_sq + (1 - self.beta) * (grad_norm ** 2)

        # Compute variance: E[X^2] - E[X]^2
        variance = max(0, self.ema_norm_sq - self.ema_norm ** 2)

        # Signal: mean / (var + eps)
        signal = self.ema_norm / (variance + self.eps)

        return signal

    def _compute_grad_norm(self, model: nn.Module) -> float:
        """Compute total gradient norm of model parameters."""
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        return total_norm_sq ** 0.5

    def reset(self):
        """Reset signal state."""
        self.ema_norm = None
        self.ema_norm_sq = None
        self.step_count = 0

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            "ema_norm": self.ema_norm,
            "ema_norm_sq": self.ema_norm_sq,
            "step_count": self.step_count,
            "beta": self.beta,
        }

    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.ema_norm = state["ema_norm"]
        self.ema_norm_sq = state["ema_norm_sq"]
        self.step_count = state["step_count"]
        self.beta = state.get("beta", self.beta)


class CosineToGlobalSignal(BaseSignal):
    """
    Signal based on cosine similarity to previous global gradient.

    Measures how aligned each worker's gradient is with the consensus.
    Workers with gradients pointing in similar directions as the global
    gradient are considered more trustworthy.
    """

    def __init__(self, beta: float = 0.99, eps: float = 1e-8):
        """
        Args:
            beta: EMA smoothing factor for the signal
            eps: Small constant for numerical stability
        """
        self.beta = beta
        self.eps = eps
        self.prev_global_grad: Optional[torch.Tensor] = None
        self.ema_cosine: Optional[float] = None
        self.step_count = 0

    def update(
        self,
        model: nn.Module = None,
        global_grad: torch.Tensor = None,
        **kwargs,
    ) -> float:
        """
        Update cosine similarity signal.

        Args:
            model: Model with current local gradients
            global_grad: Flattened global gradient from previous step

        Returns:
            Signal value (EMA of cosine similarity, higher is better)
        """
        self.step_count += 1

        # Flatten local gradients
        local_grad = self._flatten_grads(model)

        # Compute cosine similarity to previous global gradient
        if self.prev_global_grad is not None and local_grad is not None:
            cosine = self._cosine_similarity(local_grad, self.prev_global_grad)
        else:
            cosine = 1.0  # Neutral on first step

        # Update EMA
        if self.ema_cosine is None:
            self.ema_cosine = cosine
        else:
            self.ema_cosine = self.beta * self.ema_cosine + (1 - self.beta) * cosine

        # Store current global gradient for next step
        if global_grad is not None:
            self.prev_global_grad = global_grad.clone()

        return self.ema_cosine

    def _flatten_grads(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Flatten all gradients into a single vector."""
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.view(-1))

        if not grads:
            return None

        return torch.cat(grads)

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = a.norm()
        norm_b = b.norm()

        if norm_a < self.eps or norm_b < self.eps:
            return 0.0

        return (a @ b / (norm_a * norm_b)).item()

    def reset(self):
        """Reset signal state."""
        self.prev_global_grad = None
        self.ema_cosine = None
        self.step_count = 0

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            "ema_cosine": self.ema_cosine,
            "step_count": self.step_count,
            "beta": self.beta,
            # Note: prev_global_grad not saved (recomputed from training)
        }

    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.ema_cosine = state["ema_cosine"]
        self.step_count = state["step_count"]
        self.beta = state.get("beta", self.beta)
        self.prev_global_grad = None  # Will be recomputed


class CosineToMeanSignal(BaseSignal):
    """
    Signal based on cosine similarity to the CURRENT unweighted mean gradient.

    This is the correct approach for identifying divergent workers:
    1. All-reduce to get unweighted sum of gradients
    2. Compare each worker's local gradient to this mean
    3. Workers with divergent gradients get lower cosine similarity

    Unlike CosineToGlobalSignal, this compares to the current consensus,
    not the previous step's weighted gradient.
    """

    def __init__(self, beta: float = 0.9, eps: float = 1e-8):
        """
        Args:
            beta: EMA smoothing factor (lower = faster adaptation)
            eps: Small constant for numerical stability
        """
        self.beta = beta
        self.eps = eps
        self.ema_cosine: Optional[float] = None
        self.step_count = 0

    def update(
        self,
        model: nn.Module = None,
        mean_grad: torch.Tensor = None,
        **kwargs,
    ) -> float:
        """
        Update cosine similarity signal.

        Args:
            model: Model with current local gradients
            mean_grad: Unweighted mean gradient (computed via all-reduce in training loop)

        Returns:
            Signal value (EMA of cosine similarity, higher is better)
        """
        self.step_count += 1

        # Flatten local gradients
        local_grad = self._flatten_grads(model)

        # Compute cosine similarity to unweighted mean
        if mean_grad is not None and local_grad is not None:
            cosine = self._cosine_similarity(local_grad, mean_grad)
        else:
            cosine = 1.0  # Neutral if no mean available

        # Update EMA (use lower beta for faster adaptation)
        if self.ema_cosine is None:
            self.ema_cosine = cosine
        else:
            self.ema_cosine = self.beta * self.ema_cosine + (1 - self.beta) * cosine

        return self.ema_cosine

    def _flatten_grads(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Flatten all gradients into a single vector."""
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.view(-1))

        if not grads:
            return None

        return torch.cat(grads)

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = a.norm()
        norm_b = b.norm()

        if norm_a < self.eps or norm_b < self.eps:
            return 0.0

        return (a @ b / (norm_a * norm_b)).item()

    def reset(self):
        """Reset signal state."""
        self.ema_cosine = None
        self.step_count = 0

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            "ema_cosine": self.ema_cosine,
            "step_count": self.step_count,
            "beta": self.beta,
        }

    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.ema_cosine = state["ema_cosine"]
        self.step_count = state["step_count"]
        self.beta = state.get("beta", self.beta)


class GradientDirectionStabilitySignal(BaseSignal):
    """
    Signal based on gradient direction stability over time.

    This measures how consistent each worker's own gradient direction is.
    Works with any number of workers (including 2) because it doesn't
    compare to consensus - it measures self-consistency.

    Key insight: Corrupted data produces noisy gradients that fluctuate
    in direction, while clean data produces consistent gradient directions.

    Algorithm:
    1. Keep EMA of normalized gradient direction
    2. Each step, compute cosine similarity of current gradient to EMA direction
    3. Higher cosine = more stable direction = higher signal
    """

    def __init__(self, beta: float = 0.9, eps: float = 1e-8):
        """
        Args:
            beta: EMA smoothing factor for direction tracking (lower = faster adaptation)
            eps: Small constant for numerical stability
        """
        self.beta = beta
        self.eps = eps
        self.ema_direction: Optional[torch.Tensor] = None
        self.ema_stability: Optional[float] = None
        self.step_count = 0

    def update(
        self,
        model: nn.Module = None,
        **kwargs,
    ) -> float:
        """
        Update gradient direction stability signal.

        Args:
            model: Model with current local gradients

        Returns:
            Signal value (EMA of direction stability, higher is better)
        """
        self.step_count += 1

        # Get current gradient
        local_grad = self._flatten_grads(model)
        if local_grad is None:
            return self.ema_stability if self.ema_stability is not None else 1.0

        # Normalize to unit vector (direction only)
        grad_norm = local_grad.norm()
        if grad_norm < self.eps:
            return self.ema_stability if self.ema_stability is not None else 1.0
        normalized_grad = local_grad / grad_norm

        # First step: initialize EMA direction
        if self.ema_direction is None:
            self.ema_direction = normalized_grad.clone()
            self.ema_stability = 1.0
            return self.ema_stability

        # Compute cosine similarity to EMA direction (stability measure)
        stability = self._cosine_similarity(normalized_grad, self.ema_direction)

        # Update EMA of stability score
        if self.ema_stability is None:
            self.ema_stability = stability
        else:
            self.ema_stability = self.beta * self.ema_stability + (1 - self.beta) * stability

        # Update EMA direction (track the moving average direction)
        self.ema_direction = self.beta * self.ema_direction + (1 - self.beta) * normalized_grad
        # Re-normalize EMA direction to unit vector
        ema_norm = self.ema_direction.norm()
        if ema_norm > self.eps:
            self.ema_direction = self.ema_direction / ema_norm

        return self.ema_stability

    def _flatten_grads(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Flatten all gradients into a single vector."""
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.view(-1))

        if not grads:
            return None

        return torch.cat(grads)

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = a.norm()
        norm_b = b.norm()

        if norm_a < self.eps or norm_b < self.eps:
            return 0.0

        return (a @ b / (norm_a * norm_b)).item()

    def reset(self):
        """Reset signal state."""
        self.ema_direction = None
        self.ema_stability = None
        self.step_count = 0

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            "ema_stability": self.ema_stability,
            "step_count": self.step_count,
            "beta": self.beta,
            # Note: ema_direction not saved (recomputed from training)
        }

    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.ema_stability = state["ema_stability"]
        self.step_count = state["step_count"]
        self.beta = state.get("beta", self.beta)
        self.ema_direction = None  # Will be recomputed


def create_signal(signal_type: str, beta: float = 0.99) -> BaseSignal:
    """
    Factory function to create signal objects.

    Args:
        signal_type: Type of signal ("loss_ema", "grad_norm_stability", "cosine_to_prev_global",
                     "cosine_to_mean", "direction_stability")
        beta: EMA smoothing factor

    Returns:
        Signal object
    """
    signals = {
        "loss_ema": LossEMASignal,
        "grad_norm_stability": GradNormStabilitySignal,
        "cosine_to_prev_global": CosineToGlobalSignal,
        "cosine_to_mean": CosineToMeanSignal,
        "direction_stability": GradientDirectionStabilitySignal,
    }

    if signal_type not in signals:
        raise ValueError(f"Unknown signal type: {signal_type}. Available: {list(signals.keys())}")

    return signals[signal_type](beta=beta)
