"""
Checkpoint utilities for Signal-Aware Data Parallel Training.

Save and resume training runs with full state restoration.
"""

import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Any,
    signal_state: Optional[dict] = None,
    weight_state: Optional[dict] = None,
    extra: Optional[dict] = None,
    rank: int = 0,
):
    """
    Save training checkpoint (rank 0 only).

    Args:
        path: Path to save checkpoint
        model: Model (will unwrap DDP if needed)
        optimizer: Optimizer
        step: Current training step
        config: Configuration
        signal_state: Signal object state dict
        weight_state: Weight computation state dict
        extra: Additional state to save
        rank: Process rank (only 0 saves)
    """
    if rank != 0:
        return

    # Unwrap DDP model
    model_to_save = model
    if hasattr(model, "module"):
        model_to_save = model.module

    # Build checkpoint dict
    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": _dataclass_to_dict(config) if hasattr(config, "__dataclass_fields__") else config,
    }

    if signal_state is not None:
        checkpoint["signal_state"] = signal_state

    if weight_state is not None:
        checkpoint["weight_state"] = weight_state

    if extra is not None:
        checkpoint.update(extra)

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save with temp file for atomicity
    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.rename(temp_path, path)

    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = None,
    strict: bool = True,
) -> dict:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load weights into (can be DDP-wrapped)
        optimizer: Optimizer to load state into (optional)
        device: Device to load to
        strict: Whether to strictly enforce state dict matching

    Returns:
        Checkpoint dict with all saved state
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Load model state (unwrap DDP if needed)
    model_to_load = model
    if hasattr(model, "module"):
        model_to_load = model.module

    model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from {path} (step {checkpoint.get('step', '?')})")

    return checkpoint


def get_latest_checkpoint(ckpt_dir: str, prefix: str = "ckpt_") -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Args:
        ckpt_dir: Directory containing checkpoints
        prefix: Checkpoint filename prefix

    Returns:
        Path to latest checkpoint or None
    """
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None

    checkpoints = list(ckpt_path.glob(f"{prefix}*.pt"))
    if not checkpoints:
        return None

    # Sort by step number in filename
    def get_step(p):
        try:
            # Expect format: ckpt_00001000.pt
            return int(p.stem.split("_")[-1])
        except (ValueError, IndexError):
            return 0

    checkpoints.sort(key=get_step)
    return str(checkpoints[-1])


def _dataclass_to_dict(obj) -> dict:
    """Convert nested dataclass to dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in vars(obj).items()}
    return obj


class CheckpointManager:
    """
    Manages checkpoint saving with rotation.

    Keeps only the last N checkpoints to save disk space.
    """

    def __init__(
        self,
        ckpt_dir: str,
        prefix: str = "ckpt_",
        max_keep: int = 5,
        rank: int = 0,
    ):
        """
        Args:
            ckpt_dir: Directory for checkpoints
            prefix: Checkpoint filename prefix
            max_keep: Maximum checkpoints to keep
            rank: Process rank
        """
        self.ckpt_dir = Path(ckpt_dir)
        self.prefix = prefix
        self.max_keep = max_keep
        self.rank = rank
        self.saved_checkpoints = []

        if rank == 0:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        **kwargs,
    ):
        """
        Save checkpoint and rotate old ones.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current step (used in filename)
            **kwargs: Additional args passed to save_checkpoint
        """
        if self.rank != 0:
            return

        # Build checkpoint path
        ckpt_name = f"{self.prefix}{step:08d}.pt"
        ckpt_path = str(self.ckpt_dir / ckpt_name)

        # Save checkpoint
        save_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            step=step,
            rank=self.rank,
            **kwargs,
        )

        # Track saved checkpoints
        self.saved_checkpoints.append(ckpt_path)

        # Rotate old checkpoints
        while len(self.saved_checkpoints) > self.max_keep:
            old_ckpt = self.saved_checkpoints.pop(0)
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
                print(f"Removed old checkpoint: {old_ckpt}")

    def get_latest(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        return get_latest_checkpoint(str(self.ckpt_dir), self.prefix)
