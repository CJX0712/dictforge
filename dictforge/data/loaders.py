"""Loaders: optional sklearn availability probe and npz task loading."""

from __future__ import annotations

import os

import numpy as np


def available_sklearn() -> bool:
    """Probe whether scikit-learn is importable (optional Tier-0 backend)."""
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:  # pragma: no cover - depends on environment
        return False


def split_train_val_test(
    X: np.ndarray, n_train: int, n_val: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic column-wise split into (train, val, test)."""
    n = X.shape[1]
    n_test = n - n_train - n_val
    if n_test < 0:
        raise ValueError("n_train + n_val exceeds number of samples")
    return X[:, :n_train], X[:, n_train : n_train + n_val], X[:, n_train + n_val :]


def load_task_npz(path: str) -> dict:
    """Load a previously generated task from an ``.npz`` archive."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
