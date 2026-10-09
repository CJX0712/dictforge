"""Shared test utilities (importable by every test module)."""

from __future__ import annotations

from dictforge.core.config import Config
from dictforge.data.synthetic import make_sparse_coded_task


def make_task(
    dataset: str = "gabor", seed: int = 7, n_train: int = 200, n_test: int = 80, n_val: int = 60
):
    """Build a small synthetic task for fast, deterministic unit tests."""
    cfg = Config(
        patch_dim=16,
        n_atoms=24,
        sparsity=4,
        true_atoms=8,
        true_sparsity=3,
        n_train=n_train,
        n_test=n_test,
        n_val=n_val,
        n_iter=4,
        seed=seed,
    )
    return cfg, make_sparse_coded_task(cfg, dataset, seed)


def small_cfg(n_seeds: int = 2, datasets: tuple = ("gabor", "randstruct")) -> Config:
    """A lightweight benchmark config for fast gate tests."""
    cfg = Config(n_seeds=n_seeds, datasets=datasets)
    cfg.n_train = 200
    cfg.n_test = 80
    cfg.n_val = 60
    cfg.n_iter = 4
    return cfg
