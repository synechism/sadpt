"""
Signal-Aware Data Parallel Training on TinyShakespeare.

Simplified training script for quick experiments.

Usage (local single GPU):
    python train_shakespeare.py

Usage (2 GPUs with torchrun):
    torchrun --nproc_per_node=2 train_shakespeare.py --agg-mode signal_weighted --non-iid-mode clean_vs_corrupt

Usage (Modal cloud - 2 GPUs):
    modal run modal_train.py
"""

import math
import sys
from dataclasses import dataclass
from itertools import cycle
from typing import Optional

import torch
import torch.distributed as dist

from dist_utils import (
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    barrier,
    all_gather_float,
    reduce_mean_scalar,
    print0,
)
from utils.seed import seed_all
from data.shakespeare import create_shakespeare_dataloaders
from data.sharding import describe_corruption
from model.gpt import GPT, GPTConfig
from dp.signals import create_signal
from dp.weights import compute_weights, WeightState, compute_uniform_weights
from dp.aggregator import sync_grads_uniform, sync_grads_weighted, clip_grad_norm, get_global_grad_vector


@dataclass
class SimpleConfig:
    """Simplified config for Shakespeare training."""
    # Data
    seq_len: int = 256
    batch_size: int = 32

    # Model (small for quick training)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False

    # Training
    max_steps: int = 2000
    lr: float = 1e-3
    warmup_steps: int = 100
    weight_decay: float = 0.1
    clip_grad_norm: float = 1.0
    eval_every: int = 100

    # DP settings
    agg_mode: str = "uniform"  # "uniform" or "signal_weighted"
    signal_type: str = "loss_ema"  # Lower loss = more trustworthy gradients
    weight_temp: float = 1.0
    weight_ema_beta: float = 0.9
    signal_ema_beta: float = 0.99
    w_min: float = 0.1
    w_max: float = 0.9
    weight_freeze_step: int = 500  # Freeze weights after this step to prevent flip during overfitting

    # Corruption (for non-IID experiments)
    non_iid_mode: str = "iid"  # "iid" or "clean_vs_corrupt"
    corrupt_rank: int = 1
    corruption_type: str = "token_swap"
    corruption_prob: float = 0.3

    # Runtime
    seed: int = 42
    log_every: int = 10

    @property
    def corruption(self):
        """Compatibility wrapper for corruption config."""
        from config import CorruptionConfig
        return CorruptionConfig(
            non_iid_mode=self.non_iid_mode,
            corrupt_rank=self.corrupt_rank,
            corruption_type=self.corruption_type,
            corruption_prob=self.corruption_prob,
        )

    @property
    def data(self):
        """Compatibility wrapper for data config."""
        class DataWrapper:
            pass
        d = DataWrapper()
        d.seq_len = self.seq_len
        return d

    @property
    def training(self):
        """Compatibility wrapper for training config."""
        class TrainWrapper:
            pass
        t = TrainWrapper()
        t.batch_size = self.batch_size
        return t


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    """Cosine LR schedule with warmup."""
    min_lr = max_lr * 0.1
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, val_loader, device, n_batches: int = 20) -> float:
    """Quick evaluation."""
    model.eval()
    total_loss = 0.0
    count = 0
    val_iter = iter(val_loader)
    for _ in range(n_batches):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def train(cfg: SimpleConfig):
    """Main training function."""
    # Setup distributed
    r, ws, lr, dev = setup_distributed()

    print0("=" * 60)
    print0("Signal-Aware DP Training on TinyShakespeare")
    print0("=" * 60)
    print0(f"World size: {ws}, Device: {dev}")
    print0(f"Aggregation: {cfg.agg_mode}")
    print0(f"Non-IID mode: {cfg.non_iid_mode}")
    print0("=" * 60)

    # Seed
    seed_all(cfg.seed, rank=r)

    # Data description per rank
    data_desc = describe_corruption(cfg.corruption, r, ws)
    print(f"[Rank {r}] Data: {data_desc}")
    barrier()

    # Load data
    print0("Loading TinyShakespeare...")
    train_loader, val_loader, vocab_size = create_shakespeare_dataloaders(cfg, r, ws, dev)
    print0(f"Train batches: {len(train_loader)}, Vocab: {vocab_size}")

    # Create model
    print0("Creating model...")
    model_config = GPTConfig(
        block_size=cfg.seq_len,
        vocab_size=vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    model = GPT(model_config).to(dev)
    n_params = model.get_num_params()
    print0(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Broadcast parameters from rank 0 (replaces DDP constructor broadcast)
    # We don't use DDP because its backward hooks interfere with custom gradient sync
    if is_distributed():
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.lr,
        betas=(0.9, 0.95),
        device_type=dev.type,
    )

    # Signal and weight state
    signal = create_signal(cfg.signal_type, beta=cfg.signal_ema_beta)
    weight_state = WeightState()
    weights = compute_uniform_weights(ws, dev)

    # Training loop
    print0(f"Training for {cfg.max_steps} steps...")
    model.train()
    train_iter = cycle(train_loader)

    for step in range(cfg.max_steps):
        # LR schedule
        lr_val = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_val

        # Get batch
        x, y = next(train_iter)
        x, y = x.to(dev), y.to(dev)

        # Forward
        _, loss = model(x, y)
        local_loss = loss.item()

        # Backward FIRST (so gradients exist for signal computation)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Compute unweighted mean gradient for cosine_to_mean signal
        mean_grad = None
        if cfg.signal_type == "cosine_to_mean" and is_distributed():
            # Get local gradient vector
            local_grad = get_global_grad_vector(model)
            if local_grad is not None:
                # All-reduce to get sum, then divide by world_size for mean
                mean_grad = local_grad.clone()
                dist.all_reduce(mean_grad, op=dist.ReduceOp.SUM)
                mean_grad = mean_grad / ws

        # Update signal AFTER backward (when gradients exist)
        sig_val = signal.update(
            loss=local_loss,
            model=model,
            mean_grad=mean_grad,  # For cosine_to_mean signal
        )
        signals = all_gather_float(sig_val, dev)

        # Compute weights (freeze after warmup to prevent flip during overfitting)
        if cfg.agg_mode == "signal_weighted":
            if step < cfg.weight_freeze_step:
                weights, weight_state = compute_weights(
                    signals=signals,
                    temp=cfg.weight_temp,
                    w_min=cfg.w_min,
                    w_max=cfg.w_max,
                    ema_beta=cfg.weight_ema_beta,
                    state=weight_state,
                )
            elif step == cfg.weight_freeze_step:
                print0(f"Freezing weights at step {step}: {weights.tolist()}")
            # else: keep using the frozen weights from step weight_freeze_step
        else:
            weights = compute_uniform_weights(ws, dev)

        # Custom gradient sync (replaces DDP's all-reduce)
        if cfg.agg_mode == "signal_weighted":
            sync_grads_weighted(model, weights[r].item())
        else:
            sync_grads_uniform(model, ws)

        # Clip gradients
        grad_norm = clip_grad_norm(model, cfg.clip_grad_norm)

        # Step
        optimizer.step()

        # Logging
        if step % cfg.log_every == 0:
            global_loss = reduce_mean_scalar(local_loss)
            # Gather per-worker losses to verify they differ
            local_losses = all_gather_float(local_loss, dev)
            eval_loss = None
            if step % cfg.eval_every == 0 and step > 0:
                eval_loss = evaluate(model, val_loader, dev)
                if is_distributed():
                    eval_loss = reduce_mean_scalar(eval_loss)

            if r == 0:
                w_str = " ".join(f"{w:.3f}" for w in weights.tolist()) if cfg.agg_mode == "signal_weighted" else ""
                s_str = " ".join(f"{s:.4f}" for s in signals.tolist()) if cfg.agg_mode == "signal_weighted" else ""
                l_str = " ".join(f"{l:.2f}" for l in local_losses.tolist()) if is_distributed() else ""
                msg = f"step {step:5d} | loss {global_loss:.4f} | lr {lr_val:.2e} | gnorm {grad_norm:.2f}"
                if w_str:
                    msg += f" | w [{w_str}] s [{s_str}]"
                if l_str:
                    msg += f" | losses [{l_str}]"
                if eval_loss is not None:
                    msg += f" | eval {eval_loss:.4f}"
                print(msg)

    # Final eval
    print0("Final evaluation...")
    final_loss = evaluate(model, val_loader, dev, n_batches=50)
    if is_distributed():
        final_loss = reduce_mean_scalar(final_loss)
    print0(f"Final eval loss: {final_loss:.4f}")

    barrier()
    cleanup_distributed()
    print0("Done!")

    return final_loss


def parse_args() -> SimpleConfig:
    """Parse command line args into config."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg-mode", type=str, default="uniform", choices=["uniform", "signal_weighted"])
    parser.add_argument("--non-iid-mode", type=str, default="iid", choices=["iid", "clean_vs_corrupt"])
    parser.add_argument("--corruption-type", type=str, default="token_swap")
    parser.add_argument("--corruption-prob", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-freeze-step", type=int, default=500)
    args = parser.parse_args()

    cfg = SimpleConfig()
    cfg.agg_mode = args.agg_mode
    cfg.non_iid_mode = args.non_iid_mode
    cfg.corruption_type = args.corruption_type
    cfg.corruption_prob = args.corruption_prob
    cfg.max_steps = args.max_steps
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.seed = args.seed
    cfg.weight_freeze_step = args.weight_freeze_step
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
