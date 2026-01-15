"""
Signal-Aware Data Parallel modules.

This is the heart of the project: measuring gradient quality and
dynamically reweighting workers during training.
"""

from dp.signals import LossEMASignal, GradNormStabilitySignal, create_signal
from dp.weights import compute_weights, WeightState
from dp.aggregator import sync_grads_uniform, sync_grads_weighted

__all__ = [
    "LossEMASignal",
    "GradNormStabilitySignal",
    "create_signal",
    "compute_weights",
    "WeightState",
    "sync_grads_uniform",
    "sync_grads_weighted",
]
