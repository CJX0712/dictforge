"""Determinism: identical seed -> bit-identical results."""

import numpy as np
from util import make_task

from dictforge.core.config import Config
from dictforge.core.seed import set_all
from dictforge.dictlearn.baselines import init_dict
from dictforge.dictlearn.ksvd import ksvd
from dictforge.pipeline.benchmark import run_benchmark


def test_ksvd_deterministic():
    set_all(11)
    cfg, task = make_task()
    rng1 = np.random.RandomState(0)
    D01 = init_dict(cfg.patch_dim, cfg.n_atoms, rng1)
    D1, Z1, _ = ksvd(D01, task["X_train"], cfg.sparsity, cfg.n_iter, cfg.tol)

    set_all(11)
    rng2 = np.random.RandomState(0)
    D02 = init_dict(cfg.patch_dim, cfg.n_atoms, rng2)
    D2, Z2, _ = ksvd(D02, task["X_train"], cfg.sparsity, cfg.n_iter, cfg.tol)

    assert np.max(np.abs(D1 - D2)) <= 1e-9
    assert np.max(np.abs(Z1 - Z2)) <= 1e-9


def test_benchmark_deterministic():
    cfg = Config(n_seeds=2, datasets=("gabor",))
    r1 = run_benchmark(cfg)
    r2 = run_benchmark(cfg)
    for m in r1["per_method"]:
        d1 = r1["per_method"][m]
        d2 = r2["per_method"][m]
        assert abs(d1["mean"] - d2["mean"]) <= 1e-9
        assert abs(d1["std"] - d2["std"]) <= 1e-9
    assert abs(r1["headline"]["reduction_vs_naive"] - r2["headline"]["reduction_vs_naive"]) <= 1e-9
