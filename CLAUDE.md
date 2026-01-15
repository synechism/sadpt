# Signal-Aware Data Parallel Training (SADPT)

## Project Summary

This project implements **signal-aware gradient aggregation** for distributed training. Instead of the standard uniform averaging of gradients across workers:

```
g_global = (1/N) * sum(g_i)  # Standard DP
```

We use weighted aggregation based on each worker's "signal" (gradient quality):

```
g_global = sum(w_i * g_i)  where sum(w_i) = 1  # Signal-aware DP
```

The key insight: workers with cleaner data produce better gradients. By measuring gradient quality online and upweighting trustworthy workers, we can improve convergence on non-IID data.

## Current State

**IMPLEMENTATION COMPLETE** - Ready for training experiments.

### Files Structure

```
sadpt/
├── train_shakespeare.py    # Main training script (TinyShakespeare)
├── modal_train.py          # Modal cloud training (2-GPU experiments)
├── train.py                # Full training script (WikiText, more complex)
├── config.py               # All configuration knobs
├── dist_utils.py           # Distributed training utilities
├── data/
│   ├── shakespeare.py      # TinyShakespeare dataset (~300k tokens)
│   ├── dataset.py          # WikiText dataset loading
│   ├── sharding.py         # Non-IID corruption (token_swap, etc.)
│   └── collate.py          # Batch collation
├── model/
│   ├── gpt.py              # GPT model (from nanoGPT)
│   └── loss.py             # Loss functions
├── dp/
│   ├── signals.py          # Signal computation (LossEMA, GradNormStability)
│   ├── weights.py          # Signal → Weight mapping
│   └── aggregator.py       # Custom gradient sync (the core innovation)
└── utils/
    ├── seed.py             # Reproducibility
    ├── logging.py          # Metrics logging (JSONL + W&B)
    └── checkpoint.py       # Save/resume
```

## How to Run Experiments

### Option 1: Modal (Recommended - 2 GPUs in cloud)

```bash
# Install Modal (one time)
pip install modal
modal setup

# Run baseline (uniform DP)
modal run modal_train.py

# Run signal-weighted with corruption
modal run modal_train.py --agg-mode signal_weighted --non-iid-mode clean_vs_corrupt

# Run full experiment (compares all 4 configurations)
modal run modal_train.py --experiment
```

### Option 2: Local with torchrun (if you have 2 GPUs)

```bash
# Baseline
torchrun --nproc_per_node=2 train_shakespeare.py --agg-mode uniform

# Signal-weighted with non-IID data
torchrun --nproc_per_node=2 train_shakespeare.py \
    --agg-mode signal_weighted \
    --non-iid-mode clean_vs_corrupt \
    --corruption-type token_swap \
    --corruption-prob 0.3
```

### Option 3: Single GPU (for debugging only)

```bash
python train_shakespeare.py --max-steps 500
```

## Key Experiment: Clean vs Corrupt

The core experiment tests whether signal-aware weighting helps when one worker has corrupted data:

**Setup:**
- Rank 0: Clean TinyShakespeare data
- Rank 1: Corrupted data (30% token swaps)

**Expected Results:**
- Uniform DP: Degraded performance (averaging in bad gradients)
- Signal-weighted DP: Better performance (downweights rank 1)

**Success Metric:** Weight for rank 0 should be > 0.5, weight for rank 1 should be < 0.5

## Configuration Quick Reference

Key settings in `train_shakespeare.py`:

```python
# DP mode
--agg-mode uniform          # Baseline (standard DP)
--agg-mode signal_weighted  # Our method

# Data corruption
--non-iid-mode iid                # All workers get same clean data
--non-iid-mode clean_vs_corrupt   # Rank 0 clean, rank 1 corrupted

# Corruption strength
--corruption-type token_swap      # Replace tokens randomly
--corruption-prob 0.3             # 30% of tokens corrupted
```

## Signal Types (dp/signals.py)

1. **loss_ema** (default): EMA of training loss. Lower loss = higher signal = more weight.
2. **grad_norm_stability**: mean/variance of gradient norms. Stable = trustworthy.
3. **cosine_to_prev_global**: Alignment with previous global gradient.

## What Success Looks Like

When running the experiment with `modal run modal_train.py --experiment`:

```
EXPERIMENT SUMMARY
============================================================
iid_uniform          | loss: 1.8234 | time: 120s | OK
iid_weighted         | loss: 1.8256 | time: 122s | OK    # Similar (no corruption)
noniid_uniform       | loss: 2.1456 | time: 118s | OK    # Worse (bad gradients)
noniid_weighted      | loss: 1.9123 | time: 125s | OK    # Better (downweighted rank 1)

Key result: Signal-weighted improves non-IID loss by 0.2333
SUCCESS: Signal-aware weighting helps with non-IID data!
```

## Next Steps

1. **Run the experiment** - `modal run modal_train.py --experiment`
2. **Verify weight adaptation** - Check logs to see if weights diverge (w0 > w1)
3. **Tune if needed** - Adjust `--corruption-prob` if effect is too small/large
4. **Try different signals** - Compare `loss_ema` vs `grad_norm_stability`

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
tiktoken>=0.5.0
requests>=2.28.0
modal  # for cloud training
```

## Notes

- TinyShakespeare has ~300k train tokens, ~36k val tokens
- Model is small (4 layers, 256 dim, ~2M params) for fast iteration
- Training takes ~2-3 minutes on 2x T4 GPUs
- Modal costs ~$0.50/hr for 2x T4, ~$6/hr for 2x A100
