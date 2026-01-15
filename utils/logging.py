"""
Logging utilities for Signal-Aware Data Parallel Training.

Structured logging so results are convincing and reproducible.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch


class MetricLogger:
    """
    Structured metric logging to JSONL files.

    Each step log includes:
    - step, lr, global_train_loss, rank_train_loss
    - signals vector (from all ranks)
    - weights vector
    - eval_loss (periodically)
    """

    def __init__(
        self,
        log_dir: str,
        rank: int = 0,
        world_size: int = 1,
        run_name: Optional[str] = None,
    ):
        """
        Args:
            log_dir: Directory to write logs to
            rank: Process rank
            world_size: Total number of processes
            run_name: Optional name for this run
        """
        self.log_dir = Path(log_dir)
        self.rank = rank
        self.world_size = world_size
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create log directory
        if rank == 0:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log file paths
        self.metrics_file = self.log_dir / f"{self.run_name}_metrics.jsonl"
        self.config_file = self.log_dir / f"{self.run_name}_config.json"

        # Timing
        self.start_time = time.time()
        self.step_start_time = time.time()

        # Accumulated metrics
        self._step_metrics = {}

    def log_config(self, config: Any):
        """
        Log configuration to JSON file (rank 0 only).

        Args:
            config: Configuration object (dataclass or dict)
        """
        if self.rank != 0:
            return

        if hasattr(config, "__dict__"):
            config_dict = self._dataclass_to_dict(config)
        else:
            config_dict = config

        config_dict["run_name"] = self.run_name
        config_dict["world_size"] = self.world_size
        config_dict["start_time"] = datetime.now().isoformat()

        with open(self.config_file, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    def _dataclass_to_dict(self, obj) -> dict:
        """Convert nested dataclass to dict."""
        if hasattr(obj, "__dataclass_fields__"):
            return {k: self._dataclass_to_dict(v) for k, v in vars(obj).items()}
        return obj

    def log_step(
        self,
        step: int,
        lr: float,
        train_loss: float,
        global_train_loss: Optional[float] = None,
        signals: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        eval_loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        **extra_metrics,
    ):
        """
        Log metrics for a training step.

        Args:
            step: Current training step
            lr: Learning rate
            train_loss: Local training loss
            global_train_loss: Global (reduced) training loss
            signals: Signal values from all ranks
            weights: Weight values for all ranks
            eval_loss: Evaluation loss (if computed this step)
            grad_norm: Gradient norm
            **extra_metrics: Additional metrics to log
        """
        if self.rank != 0:
            return

        # Compute timing
        step_time = time.time() - self.step_start_time
        total_time = time.time() - self.start_time
        self.step_start_time = time.time()

        # Build metrics dict
        metrics = {
            "step": step,
            "lr": lr,
            "train_loss": train_loss,
            "step_time_s": step_time,
            "total_time_s": total_time,
        }

        if global_train_loss is not None:
            metrics["global_train_loss"] = global_train_loss

        if signals is not None:
            if isinstance(signals, torch.Tensor):
                signals = signals.cpu().tolist()
            metrics["signals"] = signals

        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.cpu().tolist()
            metrics["weights"] = weights

        if eval_loss is not None:
            metrics["eval_loss"] = eval_loss

        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm

        # Add any extra metrics
        metrics.update(extra_metrics)

        # Write to JSONL file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def log_eval(self, step: int, eval_loss: float, **extra_metrics):
        """
        Log evaluation results.

        Args:
            step: Current training step
            eval_loss: Evaluation loss
            **extra_metrics: Additional metrics
        """
        if self.rank != 0:
            return

        metrics = {
            "step": step,
            "eval_loss": eval_loss,
            "type": "eval",
            **extra_metrics,
        }

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def print_step(
        self,
        step: int,
        max_steps: int,
        train_loss: float,
        lr: float,
        grad_norm: Optional[float] = None,
        weights: Optional[torch.Tensor] = None,
        eval_loss: Optional[float] = None,
    ):
        """
        Print step summary to console (rank 0 only).

        Args:
            step: Current step
            max_steps: Total steps
            train_loss: Training loss
            lr: Learning rate
            grad_norm: Gradient norm
            weights: Weight values
            eval_loss: Evaluation loss
        """
        if self.rank != 0:
            return

        elapsed = time.time() - self.start_time
        steps_per_sec = step / elapsed if elapsed > 0 else 0

        msg = f"step {step:6d}/{max_steps} | loss {train_loss:.4f} | lr {lr:.2e}"

        if grad_norm is not None:
            msg += f" | gnorm {grad_norm:.2f}"

        if weights is not None:
            if isinstance(weights, torch.Tensor):
                w_list = weights.cpu().tolist()
            else:
                w_list = weights
            w_str = " ".join(f"{w:.3f}" for w in w_list)
            msg += f" | w [{w_str}]"

        if eval_loss is not None:
            msg += f" | eval {eval_loss:.4f}"

        msg += f" | {steps_per_sec:.1f} it/s"

        print(msg)


class WandbLogger:
    """
    Weights & Biases logger wrapper.

    Provides same interface as MetricLogger but logs to W&B.
    """

    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        config: Any = None,
        rank: int = 0,
    ):
        """
        Args:
            project: W&B project name
            run_name: Run name
            config: Configuration to log
            rank: Process rank (only rank 0 logs)
        """
        self.rank = rank
        self.run = None

        if rank == 0:
            import wandb

            self.run = wandb.init(
                project=project,
                name=run_name,
                config=self._dataclass_to_dict(config) if config else None,
            )

    def _dataclass_to_dict(self, obj) -> dict:
        """Convert nested dataclass to dict."""
        if hasattr(obj, "__dataclass_fields__"):
            return {k: self._dataclass_to_dict(v) for k, v in vars(obj).items()}
        return obj

    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to W&B."""
        if self.rank == 0 and self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self):
        """Finish W&B run."""
        if self.run is not None:
            self.run.finish()


def setup_logging(
    cfg,
    rank: int,
    world_size: int,
) -> tuple[MetricLogger, Optional[WandbLogger]]:
    """
    Setup logging based on configuration.

    Args:
        cfg: Configuration object
        rank: Process rank
        world_size: World size

    Returns:
        (metric_logger, wandb_logger or None)
    """
    run_name = cfg.runtime.wandb_run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    metric_logger = MetricLogger(
        log_dir=cfg.runtime.log_dir,
        rank=rank,
        world_size=world_size,
        run_name=run_name,
    )
    metric_logger.log_config(cfg)

    wandb_logger = None
    if cfg.runtime.use_wandb and rank == 0:
        wandb_logger = WandbLogger(
            project=cfg.runtime.wandb_project,
            run_name=run_name,
            config=cfg,
            rank=rank,
        )

    return metric_logger, wandb_logger
