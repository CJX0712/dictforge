"""Benchmark metrics for dictionary-learning evaluation."""

from __future__ import annotations

import numpy as np


def reconstruction_nrmse(D: np.ndarray, X: np.ndarray, L: int) -> float:
    """Normalised reconstruction error at a matched sparsity budget ``L``.

    Scale-free: dividing by ``||X||_F`` makes the number comparable across
    datasets of different magnitude. Lower is better.
    """
    from ..dictlearn.omp import omp_batch

    Z = omp_batch(D, X, L)
    denom = float(np.linalg.norm(X))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(X - D @ Z) / denom)


def mean_sparsity(Z: np.ndarray) -> float:
    """Mean number of non-zero coefficients per signal."""
    return float(np.mean(np.count_nonzero(Z, axis=0)))
