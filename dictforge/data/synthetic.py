"""Synthetic sparse-coded signal generator (the data-generation process).

Every signal is a sparse combination of atoms from a *structured* overcomplete
dictionary D0 (d x K0). A learned dictionary (K-SVD / MOD / online) can recover
this structure and reconstruct with low error at a small sparsity budget, while a
random fixed dictionary cannot. This is the honest source of the flagship's win.
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.errors import ConfigError


def _gabor_dictionary(d: int, k0: int, rng: np.random.RandomState) -> np.ndarray:
    """Oscillatory Gabor-like atoms (frequency x phase x center x width)."""
    atoms = np.zeros((d, k0))
    t = np.arange(d, dtype=float)
    for j in range(k0):
        f = rng.uniform(0.02, 0.25)
        p = rng.uniform(0.0, 2 * np.pi)
        c = rng.uniform(0.1 * d, 0.9 * d)
        sigma = rng.uniform(0.15 * d, 0.45 * d)
        atom = np.exp(-((t - c) ** 2) / (2.0 * sigma**2)) * np.cos(2.0 * np.pi * f * (t - c) + p)
        n = np.linalg.norm(atom)
        if n > 0:
            atom /= n
        atoms[:, j] = atom
    return atoms


def _smooth_random_dictionary(
    d: int, k0: int, rng: np.random.RandomState, blur: float
) -> np.ndarray:
    """Random signals passed through a Gaussian blur -> correlated/structured."""
    raw = rng.standard_normal((d, k0))
    # separable 1-D Gaussian blur kernel
    radius = max(1, int(round(blur * d)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2.0 * max(1e-3, blur * d / 3.0) ** 2))
    kernel /= kernel.sum()
    out = np.zeros((d, k0))
    for j in range(k0):
        conv = np.convolve(raw[:, j], kernel, mode="same")
        n = np.linalg.norm(conv)
        if n > 0:
            conv /= n
        out[:, j] = conv
    return out


def build_true_dictionary(d: int, k0: int, dataset: str, rng: np.random.RandomState) -> np.ndarray:
    """Build the ground-truth structured dictionary for a dataset regime."""
    if dataset == "gabor":
        return _gabor_dictionary(d, k0, rng)
    if dataset == "randstruct":
        return _smooth_random_dictionary(d, k0, rng, blur=0.05)
    if dataset == "natural":
        return _smooth_random_dictionary(d, k0, rng, blur=0.12)
    raise ConfigError(f"unknown dataset regime: {dataset!r}")


def make_sparse_coded_task(cfg: Config, dataset: str, seed: int) -> dict:
    """Generate a train/val/test split of sparse-coded signals.

    Returns a dict with keys: X_train, X_val, X_test (each d x N), D0 (d x K0),
    and Z0_train/val/test (the generating sparse codes, for diagnostics only;
    they are NEVER used by any learner).
    """
    rng = np.random.RandomState(seed)
    d = cfg.patch_dim
    k0 = cfg.true_atoms
    l0 = cfg.true_sparsity
    n_train, n_val, n_test = cfg.n_train, cfg.n_val, cfg.n_test
    n_total = n_train + n_val + n_test

    d0 = build_true_dictionary(d, k0, dataset, rng)

    # sparse codes: exactly l0 nonzero entries per column
    Z0 = np.zeros((k0, n_total))
    for i in range(n_total):
        support = rng.choice(k0, size=l0, replace=False)
        Z0[support, i] = rng.standard_normal(l0)

    X = d0 @ Z0
    # small measurement noise (kept tiny so structure dominates)
    noise = 0.01 * rng.standard_normal((d, n_total))
    X = X + noise

    X_train = X[:, :n_train]
    X_val = X[:, n_train : n_train + n_val]
    X_test = X[:, n_train + n_val :]

    return {
        "dataset": dataset,
        "seed": seed,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "D0": d0,
        "Z0_train": Z0[:, :n_train],
        "Z0_val": Z0[:, n_train : n_train + n_val],
        "Z0_test": Z0[:, n_train + n_val :],
    }
