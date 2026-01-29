"""
Modal cloud training for Signal-Aware Data Parallel Training.

Modal provides serverless GPU compute - pay per second, no instance management.

Setup (one time):
    pip install modal
    modal setup  # authenticates with Modal

Run training:
    # Baseline (uniform DP)
    modal run modal_train.py

    # Signal-weighted DP with corruption experiment
    modal run modal_train.py --agg-mode signal_weighted --non-iid-mode clean_vs_corrupt

    # Full experiment comparing both
    modal run modal_train.py::run_experiment

Costs:
    - 2x T4: ~$0.50/hr
    - 2x A10G: ~$2/hr
    - 2x A100-40GB: ~$6/hr
    - 2x H100: ~$8/hr
"""

import modal

# Create Modal app
app = modal.App("signal-aware-dp")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tiktoken>=0.5.0",
        "requests>=2.28.0",
        "tqdm>=4.65.0",
        "datasets>=2.14.0",
        "transformers>=4.30.0",
    )
    .add_local_dir(".", "/app")
)

# Volume to persist data/checkpoints between runs
volume = modal.Volume.from_name("sadpt-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4:2",  # 2x T4 GPUs - cheap for testing. Change to "A100:2" or "H100:2" for faster training
    timeout=3600,  # 1 hour max
    volumes={"/data": volume},
)
def train_distributed(
    agg_mode: str = "uniform",
    non_iid_mode: str = "iid",
    corruption_type: str = "token_swap",
    corruption_prob: float = 0.3,
    max_steps: int = 2000,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """
    Run distributed training on 2 GPUs via torchrun.

    Returns dict with training results.
    """
    import subprocess
    import os
    import json

    os.chdir("/app")

    # Build torchrun command
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--master_port=29500",
        "train_shakespeare.py",
        f"--agg-mode={agg_mode}",
        f"--non-iid-mode={non_iid_mode}",
        f"--corruption-type={corruption_type}",
        f"--corruption-prob={corruption_prob}",
        f"--max-steps={max_steps}",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
        f"--seed={seed}",
    ]

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    # Run training
    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
    )

    return {
        "return_code": result.returncode,
        "agg_mode": agg_mode,
        "non_iid_mode": non_iid_mode,
    }


@app.function(
    image=image,
    gpu="T4:2",
    timeout=7200,  # 2 hours for full experiment
    volumes={"/data": volume},
)
def run_experiment() -> dict:
    """
    Run the full experiment: compare uniform vs signal-weighted DP
    on clean vs corrupted data using TinyShakespeare.
    """
    import subprocess
    import os
    import time

    os.chdir("/app")
    results = {}

    # Only run the two non-IID configurations (the key comparison)
    configs = [
        {"name": "noniid_uniform", "agg_mode": "uniform", "non_iid_mode": "clean_vs_corrupt"},
        {"name": "noniid_weighted", "agg_mode": "signal_weighted", "non_iid_mode": "clean_vs_corrupt"},
    ]

    for cfg in configs:
        print("\n" + "=" * 60)
        print(f"Running: {cfg['name']}")
        print("=" * 60)

        cmd = [
            "torchrun",
            "--nproc_per_node=2",
            "--master_port=29500",
            "train_shakespeare.py",
            f"--agg-mode={cfg['agg_mode']}",
            f"--non-iid-mode={cfg['non_iid_mode']}",
            "--corruption-prob=0.5",  # Higher corruption for stronger signal
            "--max-steps=2000",
            "--batch-size=32",
        ]

        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start

        # Print output for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Parse final eval loss from output
        final_loss = None
        for line in result.stdout.split("\n"):
            if "Final eval loss:" in line:
                try:
                    final_loss = float(line.split(":")[-1].strip())
                except ValueError:
                    pass

        results[cfg["name"]] = {
            "final_loss": final_loss,
            "elapsed_seconds": elapsed,
            "success": result.returncode == 0,
        }

        print(f"Result: loss={final_loss}, time={elapsed:.1f}s, returncode={result.returncode}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        status = "OK" if res["success"] else "FAIL"
        loss = f"{res['final_loss']:.4f}" if res["final_loss"] else "N/A"
        print(f"{name:20s} | loss: {loss} | time: {res['elapsed_seconds']:.1f}s | {status}")

    # The key comparison
    if results.get("noniid_uniform", {}).get("final_loss") and results.get("noniid_weighted", {}).get("final_loss"):
        improvement = results["noniid_uniform"]["final_loss"] - results["noniid_weighted"]["final_loss"]
        print("\n" + "-" * 60)
        print(f"Key result: Signal-weighted improves non-IID loss by {improvement:.4f}")
        if improvement > 0:
            print("SUCCESS: Signal-aware weighting helps with non-IID data!")
        else:
            print("NOTE: No improvement - may need tuning or more corruption")

    return results


@app.function(
    image=image,
    gpu="A10G:2",  # Use A10G for faster training on larger dataset
    timeout=14400,  # 4 hours for WikiText experiment
    volumes={"/data": volume},
)
def run_wikitext_experiment() -> dict:
    """
    Run experiment on WikiText-103 (100M+ tokens) - much larger than TinyShakespeare.
    This should show clearer benefits of signal-aware weighting since overfitting is less dominant.
    """
    import subprocess
    import os
    import time

    os.chdir("/app")
    results = {}

    configs = [
        {"name": "noniid_uniform", "agg_mode": "uniform", "non_iid_mode": "clean_vs_corrupt"},
        {"name": "noniid_weighted", "agg_mode": "signal_weighted", "non_iid_mode": "clean_vs_corrupt"},
    ]

    for cfg in configs:
        print("\n" + "=" * 60)
        print(f"Running WikiText-103: {cfg['name']}")
        print("=" * 60)

        # Use separate checkpoint directory per config to avoid resume issues
        ckpt_dir = f"checkpoints_{cfg['name']}"

        # Clean up any existing checkpoints
        subprocess.run(["rm", "-rf", ckpt_dir], capture_output=True)

        cmd = [
            "torchrun",
            "--nproc_per_node=2",
            "--master_port=29500",
            "train.py",
            f"--agg-mode={cfg['agg_mode']}",
            f"--non-iid-mode={cfg['non_iid_mode']}",
            "--dataset-name=Salesforce/wikitext",
            "--dataset-config=wikitext-103-raw-v1",
            "--corruption-prob=0.3",
            "--max-steps=5000",
            "--batch-size=16",
            "--eval-every=500",
            "--log-every=100",
            "--weight-freeze-step=1000",
            f"--ckpt-dir={ckpt_dir}",
        ]

        start = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = time.time() - start

        results[cfg["name"]] = {
            "elapsed_seconds": elapsed,
            "success": result.returncode == 0,
        }

        print(f"Result: time={elapsed:.1f}s, returncode={result.returncode}")

    # Print summary
    print("\n" + "=" * 60)
    print("WIKITEXT EXPERIMENT SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        status = "OK" if res["success"] else "FAIL"
        print(f"{name:20s} | time: {res['elapsed_seconds']:.1f}s | {status}")

    return results


@app.local_entrypoint()
def main(
    agg_mode: str = "uniform",
    non_iid_mode: str = "iid",
    corruption_type: str = "token_swap",
    corruption_prob: float = 0.3,
    max_steps: int = 2000,
    experiment: bool = False,
    wikitext: bool = False,
):
    """
    Local entrypoint - run from command line.

    Examples:
        # Quick baseline test
        modal run modal_train.py

        # Signal-weighted with corruption
        modal run modal_train.py --agg-mode signal_weighted --non-iid-mode clean_vs_corrupt

        # Full experiment on TinyShakespeare
        modal run modal_train.py --experiment

        # Full experiment on WikiText-103 (larger dataset, less overfitting)
        modal run modal_train.py --wikitext
    """
    if wikitext:
        print("Running WikiText-103 experiment (larger dataset)...")
        result = run_wikitext_experiment.remote()
    elif experiment:
        print("Running full experiment (TinyShakespeare)...")
        result = run_experiment.remote()
    else:
        print(f"Running single training: agg_mode={agg_mode}, non_iid_mode={non_iid_mode}")
        result = train_distributed.remote(
            agg_mode=agg_mode,
            non_iid_mode=non_iid_mode,
            corruption_type=corruption_type,
            corruption_prob=corruption_prob,
            max_steps=max_steps,
        )

    print("\nResult:", result)
