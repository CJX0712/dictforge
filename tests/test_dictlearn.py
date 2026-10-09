"""Engine correctness, the flagship gate, and end-to-end pipeline gate."""

import numpy as np
from util import make_task, small_cfg

from dictforge.core.seed import set_all
from dictforge.dictlearn.learners import ENGINE_NAMES, fit_dictionary
from dictforge.dictlearn.omp import omp_batch
from dictforge.eval.invariants import assert_sparsity, assert_unit_norm
from dictforge.eval.metrics import reconstruction_nrmse
from dictforge.pipeline.benchmark import run_benchmark


def test_all_engines_finite_shape_and_invariants():
    set_all(0)
    cfg, task = make_task()
    for m in ENGINE_NAMES:
        D, info = fit_dictionary(m, task["X_train"], task["X_val"], cfg, 123)
        assert np.all(np.isfinite(D)), f"{m}: non-finite dictionary"
        assert D.shape == (cfg.patch_dim, cfg.n_atoms)
        assert_unit_norm(D)
        Z = omp_batch(D, task["X_test"], cfg.sparsity)
        assert_sparsity(Z, cfg.sparsity)


def test_learned_beats_naive_random():
    set_all(1)
    cfg, task = make_task()
    Drand, _ = fit_dictionary("random", task["X_train"], task["X_val"], cfg, 1)
    Dk, _ = fit_dictionary("ksvd", task["X_train"], task["X_val"], cfg, 2)
    err_rand = reconstruction_nrmse(Drand, task["X_test"], cfg.sparsity)
    err_ksvd = reconstruction_nrmse(Dk, task["X_test"], cfg.sparsity)
    assert err_ksvd < err_rand


def test_fuse_non_inferior_and_gate():
    cfg = small_cfg(n_seeds=2, datasets=("gabor", "randstruct"))
    res = run_benchmark(cfg)
    h = res["headline"]
    best_comp = min(res["per_method"][c]["mean"] for c in ("mod", "online", "ksvd"))
    assert h["flagship_ksvdfuse"]["mean"] <= best_comp * 1.02 + 1e-9
    assert h["reduction_vs_naive"] >= 0.30
    assert h["non_inferior"] is True
    assert h["significant"] is True
    assert h["gate_pass"] is True


def test_reduction_at_least_30_percent():
    cfg = small_cfg(n_seeds=2, datasets=("gabor", "randstruct", "natural"))
    res = run_benchmark(cfg)
    assert res["headline"]["reduction_vs_naive"] >= 0.30


def test_pipeline_leak_guard_passes():
    cfg = small_cfg(n_seeds=1, datasets=("gabor",))
    # the pipeline internally asserts train/test disjoint; a successful run is the proof
    res = run_benchmark(cfg)
    assert res["headline"]["gate_pass"] is True
