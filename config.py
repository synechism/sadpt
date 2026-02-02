"""
Configuration for Signal-Aware Data Parallel Training.
All training knobs in one place with sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import argparse


@dataclass
class DataConfig:
    """Dataset and tokenization configuration."""
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    tokenizer_name: str = "gpt2"
    seq_len: int = 256
    max_train_tokens: Optional[int] = None  # Limit training tokens (None = full dataset)


@dataclass
class CorruptionConfig:
    """Non-IID / corruption configuration for data sharding."""
    non_iid_mode: Literal["iid", "clean_vs_corrupt", "dup_skew"] = "iid"
    corrupt_rank: int = 1  # Which rank gets corrupted data (legacy, for 2-GPU setup)
    corrupt_fraction: float = 0.5  # Fraction of ranks to corrupt (0.5 = half the cluster)
    corruption_type: Literal["token_swap", "span_drop", "random_insert", "duplicate"] = "token_swap"
    corruption_prob: float = 0.3  # Probability of corruption per token
    dup_factor: int = 5  # Duplication factor for dup_skew mode


@dataclass
class ModelConfig:
    """Model architecture configuration (nanoGPT style)."""
    vocab_size: int = 50304  # GPT-2 vocab padded to multiple of 64
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False  # No bias is slightly better and faster


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 8  # Per GPU batch size
    grad_accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_steps: int = 5000
    warmup_steps: int = 100
    eval_every: int = 100
    eval_batches: int = 20
    clip_grad_norm: Optional[float] = 1.0


@dataclass
class DPConfig:
    """Data Parallel weighting configuration - the heart of the project."""
    agg_mode: Literal["uniform", "signal_weighted"] = "uniform"
    signal_type: Literal["loss_ema", "grad_norm_stability", "cosine_to_prev_global"] = "loss_ema"
    weight_temp: float = 1.0  # Softmax temperature for weight computation
    weight_ema_beta: float = 0.9  # EMA smoothing for weights
    signal_ema_beta: float = 0.99  # EMA smoothing for signal computation
    w_min: float = 0.1  # Minimum weight per worker (avoid zeroing out)
    w_max: float = 0.9  # Maximum weight per worker (avoid collapse to single)
    weight_freeze_step: int = 500  # Freeze weights after this step to prevent flip during overfitting


@dataclass
class RuntimeConfig:
    """Runtime and logging configuration."""
    seed: int = 42
    log_dir: str = "logs"
    ckpt_dir: str = "checkpoints"
    log_every: int = 10
    ckpt_every: int = 1000
    use_wandb: bool = False
    wandb_project: str = "signal-aware-dp"
    wandb_run_name: Optional[str] = None
    compile_model: bool = False  # torch.compile


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self):
        # Ensure block_size matches seq_len
        pass


def get_config_from_args() -> Config:
    """Parse command line arguments and return Config."""
    parser = argparse.ArgumentParser(description="Signal-Aware Data Parallel Training")

    # Data args
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-train-tokens", type=int, default=None,
                        help="Limit training tokens (None = full dataset)")

    # Corruption args
    parser.add_argument("--non-iid-mode", type=str, default="iid",
                        choices=["iid", "clean_vs_corrupt", "dup_skew"])
    parser.add_argument("--corrupt-rank", type=int, default=1)
    parser.add_argument("--corrupt-fraction", type=float, default=0.5,
                        help="Fraction of ranks to corrupt (0.5 = half the cluster)")
    parser.add_argument("--corruption-type", type=str, default="token_swap",
                        choices=["token_swap", "span_drop", "random_insert", "duplicate"])
    parser.add_argument("--corruption-prob", type=float, default=0.3)

    # Model args
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training args
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)

    # DP args
    parser.add_argument("--agg-mode", type=str, default="uniform",
                        choices=["uniform", "signal_weighted"])
    parser.add_argument("--signal-type", type=str, default="loss_ema",
                        choices=["loss_ema", "grad_norm_stability", "cosine_to_prev_global"])
    parser.add_argument("--weight-temp", type=float, default=1.0)
    parser.add_argument("--weight-ema-beta", type=float, default=0.9)
    parser.add_argument("--signal-ema-beta", type=float, default=0.99)
    parser.add_argument("--w-min", type=float, default=0.1)
    parser.add_argument("--w-max", type=float, default=0.9)
    parser.add_argument("--weight-freeze-step", type=int, default=500)

    # Runtime args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="signal-aware-dp")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--compile-model", action="store_true")

    args = parser.parse_args()

    # Build config from args
    config = Config(
        data=DataConfig(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            tokenizer_name=args.tokenizer_name,
            seq_len=args.seq_len,
            max_train_tokens=args.max_train_tokens,
        ),
        corruption=CorruptionConfig(
            non_iid_mode=args.non_iid_mode,
            corrupt_rank=args.corrupt_rank,
            corrupt_fraction=args.corrupt_fraction,
            corruption_type=args.corruption_type,
            corruption_prob=args.corruption_prob,
        ),
        model=ModelConfig(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            eval_every=args.eval_every,
            clip_grad_norm=args.clip_grad_norm,
        ),
        dp=DPConfig(
            agg_mode=args.agg_mode,
            signal_type=args.signal_type,
            weight_temp=args.weight_temp,
            weight_ema_beta=args.weight_ema_beta,
            signal_ema_beta=args.signal_ema_beta,
            w_min=args.w_min,
            w_max=args.w_max,
            weight_freeze_step=args.weight_freeze_step,
        ),
        runtime=RuntimeConfig(
            seed=args.seed,
            log_dir=args.log_dir,
            ckpt_dir=args.ckpt_dir,
            log_every=args.log_every,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            compile_model=args.compile_model,
        ),
    )

    return config


def get_default_config() -> Config:
    """Return default configuration."""
    return Config()


# Preset configurations for common experiments
def get_baseline_uniform_config() -> Config:
    """Baseline uniform DP configuration."""
    cfg = Config()
    cfg.dp.agg_mode = "uniform"
    return cfg


def get_signal_weighted_config() -> Config:
    """Signal-weighted DP configuration."""
    cfg = Config()
    cfg.dp.agg_mode = "signal_weighted"
    cfg.dp.signal_type = "loss_ema"
    return cfg


def get_clean_vs_corrupt_config() -> Config:
    """Configuration for clean vs corrupt experiment."""
    cfg = Config()
    cfg.corruption.non_iid_mode = "clean_vs_corrupt"
    cfg.corruption.corrupt_rank = 1
    cfg.corruption.corruption_type = "token_swap"
    cfg.corruption.corruption_prob = 0.3
    cfg.dp.agg_mode = "signal_weighted"
    return cfg
