"""Orthogonal Matching Pursuit (OMP) sparse coding.

OMP greedily selects dictionary atoms and, after each selection, re-solves the
least-squares coefficients for the selected support. The residual is therefore
orthogonal to the span of the selected atoms -- a hard invariant we gate on.
"""

from __future__ import annotations

import numpy as np


def omp_single(D: np.ndarray, x: np.ndarray, L: int) -> tuple[np.ndarray, list[int]]:
    """Code a single signal ``x`` with at most ``L`` atoms of dictionary ``D``.

    Returns ``(z, support)`` where ``z`` is the sparse code (K,) and ``support``
    is the ordered list of selected atom indices.
    """
    K = D.shape[1]
    if L <= 0:
        return np.zeros(K), []
    r = x.astype(float).copy()
    support: list[int] = []
    for _ in range(L):
        corr = D.T @ r
        idx = int(np.argmax(np.abs(corr)))
        if idx in support:
            # residual already orthogonal to selected atoms; no further gain
            break
        support.append(idx)
        Ds = D[:, support]
        coef_s, *_ = np.linalg.lstsq(Ds, x, rcond=None)
        r = x - Ds @ coef_s
    z = np.zeros(K)
    if support:
        z[support] = coef_s
    return z, support


def omp_batch(D: np.ndarray, X: np.ndarray, L: int) -> np.ndarray:
    """Code every column of ``X`` (d x N) with OMP -> codes ``Z`` (K x N)."""
    K = D.shape[1]
    N = X.shape[1]
    Z = np.zeros((K, N))
    for i in range(N):
        z, _ = omp_single(D, X[:, i], L)
        Z[:, i] = z
    return Z
