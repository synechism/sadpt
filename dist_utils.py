"""
Distributed utilities for Signal-Aware Data Parallel Training.
Keeps all distributed boilerplate out of train.py.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist


# Global state for distributed info
_RANK: int = 0
_WORLD_SIZE: int = 1
_LOCAL_RANK: int = 0
_DEVICE: torch.device = torch.device("cpu")
_INITIALIZED: bool = False


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """
    Initialize distributed training from torchrun environment variables.

    Returns:
        (rank, world_size, local_rank, device)
    """
    global _RANK, _WORLD_SIZE, _LOCAL_RANK, _DEVICE, _INITIALIZED

    if _INITIALIZED:
        return _RANK, _WORLD_SIZE, _LOCAL_RANK, _DEVICE

    # Check if running with torchrun
    if all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        _RANK = int(os.environ["RANK"])
        _WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        _LOCAL_RANK = int(os.environ["LOCAL_RANK"])

        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(_LOCAL_RANK)
            _DEVICE = torch.device("cuda", _LOCAL_RANK)
        else:
            _DEVICE = torch.device("cpu")

        # Initialize process group
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

        if _RANK == 0:
            print(f"Initialized distributed training: world_size={_WORLD_SIZE}, backend={backend}")
    else:
        # Single process mode
        _RANK = 0
        _WORLD_SIZE = 1
        _LOCAL_RANK = 0
        if torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _DEVICE = torch.device("mps")
        else:
            _DEVICE = torch.device("cpu")

    _INITIALIZED = True
    return _RANK, _WORLD_SIZE, _LOCAL_RANK, _DEVICE


def cleanup_distributed():
    """Clean up distributed process group."""
    global _INITIALIZED
    if dist.is_initialized():
        dist.destroy_process_group()
    _INITIALIZED = False


def rank() -> int:
    """Get current process rank."""
    return _RANK


def world_size() -> int:
    """Get total number of processes."""
    return _WORLD_SIZE


def local_rank() -> int:
    """Get local rank (GPU index on this node)."""
    return _LOCAL_RANK


def device() -> torch.device:
    """Get device for current process."""
    return _DEVICE


def is_main() -> bool:
    """Check if this is the main process (rank 0)."""
    return _RANK == 0


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return _WORLD_SIZE > 1


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_gather_float(x: float, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Gather a scalar float from all processes.

    Args:
        x: Local scalar value
        device: Device to use (defaults to current device)

    Returns:
        Tensor of shape [world_size] with values from all ranks
    """
    if device is None:
        device = _DEVICE

    if not dist.is_initialized():
        return torch.tensor([x], device=device)

    # Convert to tensor
    local_tensor = torch.tensor([x], dtype=torch.float32, device=device)

    # Gather from all ranks
    gathered = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, local_tensor)

    return torch.cat(gathered)


def all_gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Gather a tensor from all processes.

    Args:
        tensor: Local tensor

    Returns:
        List of tensors from all ranks
    """
    if not dist.is_initialized():
        return [tensor]

    gathered = [torch.zeros_like(tensor) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, tensor)
    return gathered


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor with mean reduction.

    Args:
        tensor: Local tensor

    Returns:
        Averaged tensor across all ranks
    """
    if not dist.is_initialized():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(_WORLD_SIZE)
    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor with sum reduction.

    Args:
        tensor: Local tensor

    Returns:
        Summed tensor across all ranks
    """
    if not dist.is_initialized():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast a tensor from source rank to all ranks.

    Args:
        tensor: Tensor to broadcast (must exist on all ranks)
        src: Source rank

    Returns:
        Broadcasted tensor
    """
    if not dist.is_initialized():
        return tensor

    dist.broadcast(tensor, src=src)
    return tensor


def reduce_mean_scalar(x: float) -> float:
    """
    Reduce a scalar float with mean across all ranks.

    Args:
        x: Local scalar value

    Returns:
        Mean value across all ranks
    """
    if not dist.is_initialized():
        return x

    tensor = torch.tensor([x], dtype=torch.float32, device=_DEVICE)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / _WORLD_SIZE


def print0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main():
        print(*args, **kwargs)
