"""Method of Optimal Directions (MOD, Engan et al. 1999).

Iterates: (1) sparse-code with OMP, (2) update the dictionary by the least
squares solution ``D = X Z^T (Z Z^T)^+`` then renormalise columns. Deterministic.
"""

from __future__ import annotations

import numpy as np

from .omp import omp_batch


def mod(
    D0: np.ndarray,
    X: np.ndarray,
    L: int,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run MOD. Returns ``(D, Z, errors)`` (errors monotonically non-increasing)."""
    D = D0.astype(float).copy()
    errors: list[float] = []
    Z = omp_batch(D, X, L)
    for _ in range(n_iter):
        ZZt = Z @ Z.T
        # least-squares dictionary update
        D = (X @ Z.T) @ np.linalg.pinv(ZZt)
        D /= np.linalg.norm(D, axis=0, keepdims=True)
        Z = omp_batch(D, X, L)
        errors.append(float(np.linalg.norm(X - D @ Z)))
    return D, Z, errors
