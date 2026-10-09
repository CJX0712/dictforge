"""Baseline (non-learned) dictionaries: random fixed and DCT frame."""

from __future__ import annotations

import numpy as np


def init_dict(d: int, K: int, rng: np.random.RandomState) -> np.ndarray:
    """Random initial dictionary with unit-norm columns."""
    D = rng.standard_normal((d, K))
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    return D


def dct_frame(d: int, K: int) -> np.ndarray:
    """Deterministic overcomplete discrete-cosine-like frame.

    Fixed (no learning) structured dictionary -- a meaningful "strong fixed"
    baseline between random and learned dictionaries.
    """
    t = np.arange(d, dtype=float)
    D = np.zeros((d, K))
    for k in range(K):
        atom = np.cos(np.pi * (t + 0.5) * k / d)
        nrm = np.linalg.norm(atom)
        if nrm > 0:
            atom /= nrm
        D[:, k] = atom
    return D
