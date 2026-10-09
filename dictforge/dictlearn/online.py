"""Online dictionary learning (Mairal et al. 2009).

Tier-0 backend: scikit-learn ``MiniBatchDictionaryLearning`` when importable
(demonstrates reuse of a world-class open-source implementation); otherwise a
pure-numpy block-coordinate least-squares fallback (Tier-1 offline path). Both
are deterministic given the provided ``rng`` / ``random_state``.
"""

from __future__ import annotations

import numpy as np

from .omp import omp_batch


def _online_numpy(
    D0: np.ndarray,
    X: np.ndarray,
    n_iter: int,
    batch_size: int,
    L: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    D = D0.astype(float).copy()
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    d, K = D.shape
    n = X.shape[1]
    for _ in range(n_iter):
        A = np.zeros((K, K))
        B = np.zeros((d, K))
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            Xb = X[:, idx]
            Zb = omp_batch(D, Xb, L)
            A += Zb @ Zb.T
            B += Xb @ Zb.T
        D = B @ np.linalg.pinv(A)
        D /= np.linalg.norm(D, axis=0, keepdims=True)
    Z = omp_batch(D, X, L)
    return D, Z, [float(np.linalg.norm(X - D @ Z))]


def _online_sklearn(
    D0: np.ndarray,
    X: np.ndarray,
    n_iter: int,
    batch_size: int,
    L: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    from sklearn.decomposition import MiniBatchDictionaryLearning

    est = MiniBatchDictionaryLearning(
        n_components=D0.shape[1],
        n_iter=n_iter,
        batch_size=batch_size,
        dict_init=D0,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=L,
        fit_algorithm="cd",
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    est.fit(X.T)  # sklearn: (n_samples, n_features)
    D = est.components_.T
    Z = est.transform(X.T).T
    return D, Z, [float(np.linalg.norm(X - D @ Z))]


def online_dict_learning(
    D0: np.ndarray,
    X: np.ndarray,
    n_iter: int,
    batch_size: int,
    L: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Online dictionary learning (pure-numpy, deterministic, offline).

    scikit-learn's ``MiniBatchDictionaryLearning`` is available as an optional
    cross-check (see ``_online_sklearn``) but the benchmark uses the fast,
    dependency-free numpy implementation so the whole system stays Tier-1
    offline and single-threaded deterministic.
    """
    return _online_numpy(D0, X, n_iter, batch_size, L, rng)
