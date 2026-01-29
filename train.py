"""
Signal-Aware Data Parallel Training - Main Training Script

This script orchestrates:
1. Distributed setup with torchrun
2. Data loading with rank-specific corruption
3. Model initialization and DDP wrapping
4. Signal-aware gradient aggregation
5. Logging and checkpointing

Usage:
    # Single GPU (testing)
    python train.py --agg-mode uniform

    # 2 GPUs with uniform DP (baseline)
    torchrun --nproc_per_node=2 train.py --agg-mode uniform

    # 2 GPUs with signal-aware DP (experiment)
    torchrun --nproc_per_node=2 train.py --agg-mode signal_weighted --non-iid-mode clean_vs_corrupt

    # Full experiment: clean vs corrupt with signal weighting
    torchrun --nproc_per_node=2 train.py \\
        --agg-mode signal_weighted \\
        --non-iid-mode clean_vs_corrupt \\
        --corruption-type token_swap \\
        --corruption-prob 0.3
"""

import math
import sys
from itertools import cycle

import torch
import torch.distributed as dist

# Local imports
from config import get_config_from_args, get_default_config
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
from utils.logging import setup_logging
from utils.checkpoint import CheckpointManager, load_checkpoint
from data.dataset import create_dataloaders
from data.sharding import describe_corruption
from model.gpt import GPT, GPTConfig
from dp.signals import create_signal
from dp.weights import compute_weights, WeightState, compute_uniform_weights, analyze_weight_distribution
from dp.aggregator import sync_grads_uniform, sync_grads_weighted, clip_grad_norm


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate

    Returns:
        Learning rate for this step
    """
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Cosine decay
    if step >= max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, val_loader, cfg, device):
    """
    Evaluate model on validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        cfg: Configuration
        device: Device

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    val_iter = iter(val_loader)
    for _ in range(cfg.training.eval_batches):
        try:
            input_ids, targets = next(val_iter)
        except StopIteration:
            break

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        _, loss = model(input_ids, targets)
        total_loss += loss.item()
        n_batches += 1

    model.train()

    if n_batches == 0:
        return float("inf")

    return total_loss / n_batches


def train(cfg):
    """
    Main training function.

    Args:
        cfg: Configuration object
    """
    # ===== 1. Setup distributed =====
    r, ws, lr, dev = setup_distributed()

    print0(f"{'='*60}")
    print0(f"Signal-Aware Data Parallel Training")
    print0(f"{'='*60}")
    print0(f"World size: {ws}")
    print0(f"Aggregation mode: {cfg.dp.agg_mode}")
    print0(f"Non-IID mode: {cfg.corruption.non_iid_mode}")
    if cfg.corruption.non_iid_mode != "iid":
        print0(f"Corruption type: {cfg.corruption.corruption_type}")
        print0(f"Corruption prob: {cfg.corruption.corruption_prob}")
    print0(f"{'='*60}")

    # ===== 2. Seed for reproducibility =====
    seed_all(cfg.runtime.seed, rank=r)

    # Print per-rank data description
    data_desc = describe_corruption(cfg.corruption, r)
    print(f"[Rank {r}] Data: {data_desc}")

    barrier()

    # ===== 3. Setup logging =====
    metric_logger, wandb_logger = setup_logging(cfg, r, ws)

    # ===== 4. Load data =====
    print0("Loading data...")
    train_loader, val_loader, tokenizer = create_dataloaders(cfg, r, ws, dev)
    print0(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print0(f"Vocab size: {len(tokenizer)}")

    # ===== 5. Create model =====
    print0("Creating model...")
    model_config = GPTConfig(
        block_size=cfg.data.seq_len,
        vocab_size=len(tokenizer),
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        dropout=cfg.model.dropout,
        bias=cfg.model.bias,
    )
    model = GPT(model_config)
    model = model.to(dev)

    n_params = model.get_num_params()
    print0(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Optionally compile model
    if cfg.runtime.compile_model and hasattr(torch, "compile"):
        print0("Compiling model...")
        model = torch.compile(model)

    # Broadcast parameters from rank 0 to ensure all ranks start identical
    # (replaces DDP's constructor broadcast - we don't use DDP because its
    # backward hooks interfere with our custom gradient aggregation)
    if is_distributed():
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    # ===== 6. Create optimizer =====
    optimizer = model.configure_optimizers(
        weight_decay=cfg.training.weight_decay,
        learning_rate=cfg.training.lr,
        betas=cfg.training.betas,
        device_type=dev.type,
    )

    # ===== 7. Setup signal and weight state =====
    signal = create_signal(cfg.dp.signal_type, beta=cfg.dp.signal_ema_beta)
    weight_state = WeightState()

    # ===== 8. Setup checkpointing =====
    ckpt_manager = CheckpointManager(
        ckpt_dir=cfg.runtime.ckpt_dir,
        max_keep=3,
        rank=r,
    )

    # Resume from checkpoint if available
    start_step = 0
    latest_ckpt = ckpt_manager.get_latest()
    if latest_ckpt is not None:
        print0(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt = load_checkpoint(latest_ckpt, model, optimizer, dev)
        start_step = ckpt.get("step", 0) + 1
        if "signal_state" in ckpt:
            signal.load_state_dict(ckpt["signal_state"])
        if "weight_state" in ckpt:
            weight_state.load_state_dict(ckpt["weight_state"], dev)

    # ===== 9. Training loop =====
    print0(f"Starting training from step {start_step}...")
    model.train()

    # Infinite data iterator
    train_iter = cycle(train_loader)

    # Current weights (start uniform)
    weights = compute_uniform_weights(ws, dev)

    for step in range(start_step, cfg.training.max_steps):
        # Get learning rate
        lr_val = get_lr(
            step,
            cfg.training.warmup_steps,
            cfg.training.max_steps,
            cfg.training.lr,
            min_lr=cfg.training.lr * 0.1,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_val

        # Get batch
        input_ids, targets = next(train_iter)
        input_ids = input_ids.to(dev)
        targets = targets.to(dev)

        # Forward pass
        _, loss = model(input_ids, targets)
        local_loss = loss.item()

        # Update local signal
        signal_value = signal.update(loss=local_loss)

        # Gather signals from all ranks
        signals = all_gather_float(signal_value, dev)

        # Compute weights from signals (freeze after warmup to prevent flip during overfitting)
        if cfg.dp.agg_mode == "signal_weighted":
            if step < cfg.dp.weight_freeze_step:
                weights, weight_state = compute_weights(
                    signals=signals,
                    temp=cfg.dp.weight_temp,
                    w_min=cfg.dp.w_min,
                    w_max=cfg.dp.w_max,
                    ema_beta=cfg.dp.weight_ema_beta,
                    state=weight_state,
                )
            elif step == cfg.dp.weight_freeze_step:
                print0(f"Freezing weights at step {step}: {weights.tolist()}")
            # else: keep using the frozen weights
        else:
            weights = compute_uniform_weights(ws, dev)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Custom gradient synchronization (replaces DDP's all-reduce)
        if cfg.dp.agg_mode == "signal_weighted":
            sync_grads_weighted(model, weights[r].item())
        else:
            sync_grads_uniform(model, ws)

        # Gradient clipping
        grad_norm = None
        if cfg.training.clip_grad_norm is not None:
            grad_norm = clip_grad_norm(model, cfg.training.clip_grad_norm)

        # Optimizer step
        optimizer.step()

        # Compute global loss for logging
        global_loss = reduce_mean_scalar(local_loss)

        # Logging
        if step % cfg.runtime.log_every == 0:
            eval_loss = None
            if step % cfg.training.eval_every == 0 and step > 0:
                eval_loss = evaluate(model, val_loader, cfg, dev)
                if is_distributed():
                    eval_loss = reduce_mean_scalar(eval_loss)

            metric_logger.log_step(
                step=step,
                lr=lr_val,
                train_loss=local_loss,
                global_train_loss=global_loss,
                signals=signals,
                weights=weights,
                eval_loss=eval_loss,
                grad_norm=grad_norm,
            )

            metric_logger.print_step(
                step=step,
                max_steps=cfg.training.max_steps,
                train_loss=global_loss,
                lr=lr_val,
                grad_norm=grad_norm,
                weights=weights if cfg.dp.agg_mode == "signal_weighted" else None,
                eval_loss=eval_loss,
            )

            if wandb_logger is not None:
                log_dict = {
                    "train/loss": global_loss,
                    "train/local_loss": local_loss,
                    "train/lr": lr_val,
                }
                if grad_norm is not None:
                    log_dict["train/grad_norm"] = grad_norm
                if eval_loss is not None:
                    log_dict["eval/loss"] = eval_loss

                # Log per-rank signals and weights
                for i, (s, w) in enumerate(zip(signals.tolist(), weights.tolist())):
                    log_dict[f"signal/rank_{i}"] = s
                    log_dict[f"weight/rank_{i}"] = w

                # Log weight distribution stats
                weight_stats = analyze_weight_distribution(weights)
                for k, v in weight_stats.items():
                    log_dict[f"weight_stats/{k}"] = v

                wandb_logger.log(log_dict, step=step)

        # Checkpointing
        if step > 0 and step % cfg.runtime.ckpt_every == 0:
            ckpt_manager.save(
                model=model,
                optimizer=optimizer,
                step=step,
                config=cfg,
                signal_state=signal.state_dict(),
                weight_state=weight_state.state_dict(),
            )

    # ===== 10. Final evaluation =====
    print0("Running final evaluation...")
    final_eval_loss = evaluate(model, val_loader, cfg, dev)
    if is_distributed():
        final_eval_loss = reduce_mean_scalar(final_eval_loss)
    print0(f"Final eval loss: {final_eval_loss:.4f}")

    # Save final checkpoint
    ckpt_manager.save(
        model=model,
        optimizer=optimizer,
        step=cfg.training.max_steps,
        config=cfg,
        signal_state=signal.state_dict(),
        weight_state=weight_state.state_dict(),
    )

    # Cleanup
    if wandb_logger is not None:
        wandb_logger.finish()

    barrier()
    cleanup_distributed()

    print0("Training complete!")


def main():
    """Entry point."""
    # Parse configuration from command line
    if len(sys.argv) > 1:
        cfg = get_config_from_args()
    else:
        cfg = get_default_config()

    train(cfg)


if __name__ == "__main__":
    main()
