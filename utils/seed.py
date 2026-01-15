"""
Reproducibility utilities for Signal-Aware Data Parallel Training.
"""

import random
import numpy as np
import torch


def seed_all(seed: int, rank: int = 0, deterministic: bool = False):
    """
    Seed all random number generators for reproducibility.

    Uses seed + rank for per-rank randomness so corruption/sharding
    differs intentionally but reproducibly across workers.

    Args:
        seed: Base random seed
        rank: Process rank (added to seed for per-rank variation)
        deterministic: If True, enable deterministic algorithms (slower)
    """
    effective_seed = seed + rank

    # Python random
    random.seed(effective_seed)

    # NumPy
    np.random.seed(effective_seed)

    # PyTorch
    torch.manual_seed(effective_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)

    # Deterministic settings (can be slower)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        # Allow cuDNN to optimize (faster but non-deterministic)
        torch.backends.cudnn.benchmark = True


def get_generator(seed: int, rank: int = 0, device: torch.device = None) -> torch.Generator:
    """
    Create a PyTorch random generator with specific seed.

    Args:
        seed: Base random seed
        rank: Process rank
        device: Device for the generator

    Returns:
        Seeded torch.Generator
    """
    effective_seed = seed + rank
    g = torch.Generator(device=device)
    g.manual_seed(effective_seed)
    return g
