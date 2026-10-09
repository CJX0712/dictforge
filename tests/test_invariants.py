"""Mathematical invariant gates (mirrors the ``selftest`` hard gates)."""

import numpy as np
import pytest
from util import make_task

from dictforge.core.errors import NumericalError
from dictforge.core.seed import set_all
from dictforge.dictlearn.baselines import init_dict
from dictforge.dictlearn.ksvd import ksvd
from dictforge.dictlearn.omp import omp_batch, omp_single
from dictforge.eval.invariants import (
    assert_monotone,
    assert_no_leak,
    assert_omp_residual_orthogonal,
    assert_sparsity,
    assert_unit_norm,
)
from dictforge.eval.metrics import reconstruction_nrmse


def test_omp_residual_orthogonal():
    set_all(0)
    cfg, task = make_task()
    rng = np.random.RandomState(0)
    D = init_dict(cfg.patch_dim, cfg.n_atoms, rng)
    x = task["X_train"][:, 0]
    z, support = omp_single(D, x, cfg.sparsity)
    assert_omp_residual_orthogonal(D, x, z)  # raises only on violation


def test_unit_norm_holds():
    set_all(1)
    cfg, task = make_task()
    rng = np.random.RandomState(2)
    D0 = init_dict(cfg.patch_dim, cfg.n_atoms, rng)
    Dk, _, _ = ksvd(D0, task["X_train"], cfg.sparsity, cfg.n_iter, cfg.tol)
    assert_unit_norm(Dk)


def test_ksvd_monotone_non_increasing():
    set_all(3)
    cfg, task = make_task()
    rng = np.random.RandomState(4)
    D0 = init_dict(cfg.patch_dim, cfg.n_atoms, rng)
    _, _, errors = ksvd(D0, task["X_train"], cfg.sparsity, cfg.n_iter, cfg.tol)
    assert_monotone(errors)


def test_sparsity_budget_respected():
    set_all(5)
    cfg, task = make_task()
    rng = np.random.RandomState(6)
    D0 = init_dict(cfg.patch_dim, cfg.n_atoms, rng)
    Z = omp_batch(D0, task["X_test"], cfg.sparsity)
    assert_sparsity(Z, cfg.sparsity)


def test_nrmse_scale_invariant():
    set_all(7)
    cfg2, task2 = make_task()
    rng = np.random.RandomState(8)
    D0 = init_dict(cfg2.patch_dim, cfg2.n_atoms, rng)
    a = reconstruction_nrmse(D0, task2["X_test"], cfg2.sparsity)
    b = reconstruction_nrmse(D0, task2["X_test"] * 5.0, cfg2.sparsity)
    assert abs(a - b) < 1e-9


def test_assert_no_leak_raises_on_overlap():
    with pytest.raises(Exception):  # DataLeakError is a subclass
        assert_no_leak(np.array([0, 1, 2]), np.array([2, 3, 4]))


def test_assert_monotone_raises_on_increase():
    with pytest.raises(NumericalError):
        assert_monotone([1.0, 1.0, 2.0])  # increase -> violation
