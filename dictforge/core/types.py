"""Shared dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SparseModel:
    """Learned dictionary model."""

    dictionary: object  # np.ndarray, shape (d, K) with unit-norm columns
    name: str = "model"
    sparsity: int = 0
    train_nrmse: float = 0.0
    history: tuple = field(default_factory=tuple)


@dataclass
class BenchmarkRow:
    """One (method, dataset, seed) benchmark observation."""

    method: str
    dataset: str
    seed: int
    nrmse: float
    sparsity: float
    train_nrmse: float = 0.0
