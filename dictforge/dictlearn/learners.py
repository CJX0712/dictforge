"""Learner registry / dispatcher."""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.errors import ConfigError
from ..eval.metrics import reconstruction_nrmse
from .baselines import dct_frame, init_dict
from .fuse import ksvd_fuse
from .ksvd import ksvd
from .mod import mod
from .online import online_dict_learning

ENGINE_NAMES = ["random", "dct", "mod", "online", "ksvd", "ksvdfuse"]


def fit_dictionary(
    name: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: Config,
    seed_int: int,
) -> tuple[np.ndarray, dict]:
    """Fit a single dictionary-learning engine and return ``(D, info)``."""
    rng = np.random.RandomState(seed_int)
    K = cfg.n_atoms
    L = cfg.sparsity

    if name == "random":
        D = init_dict(cfg.patch_dim, K, rng)
        return D, {"method": "random", "train_nrmse": reconstruction_nrmse(D, X_train, L)}
    if name == "dct":
        D = dct_frame(cfg.patch_dim, K)
        return D, {"method": "dct", "train_nrmse": reconstruction_nrmse(D, X_train, L)}
    if name == "mod":
        D0 = init_dict(cfg.patch_dim, K, rng)
        D, _, _ = mod(D0, X_train, L, cfg.n_iter)
        return D, {"method": "mod", "train_nrmse": reconstruction_nrmse(D, X_train, L)}
    if name == "ksvd":
        D0 = init_dict(cfg.patch_dim, K, rng)
        D, _, _ = ksvd(D0, X_train, L, cfg.n_iter, cfg.tol)
        return D, {"method": "ksvd", "train_nrmse": reconstruction_nrmse(D, X_train, L)}
    if name == "online":
        D0 = init_dict(cfg.patch_dim, K, rng)
        D, _, _ = online_dict_learning(D0, X_train, cfg.n_iter, cfg.batch_size, L, rng)
        return D, {"method": "online", "train_nrmse": reconstruction_nrmse(D, X_train, L)}
    if name == "ksvdfuse":
        D, fuse_info = ksvd_fuse(X_train, X_val, cfg, rng)
        return D, {
            "method": "ksvdfuse",
            "train_nrmse": reconstruction_nrmse(D, X_train, L),
            "fuse_info": fuse_info,
        }
    raise ConfigError(f"unknown engine: {name}")
