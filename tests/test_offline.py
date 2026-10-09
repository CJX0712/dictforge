"""Offline fallback: pure-numpy paths must work without scikit-learn."""

import numpy as np
from util import make_task

from dictforge.data.loaders import available_sklearn
from dictforge.dictlearn.baselines import init_dict
from dictforge.dictlearn.online import _online_numpy
from dictforge.eval.metrics import reconstruction_nrmse


def test_online_numpy_runs_without_sklearn():
    # scikit-learn may or may not be importable; the numpy path must always work
    cfg, task = make_task()
    rng = np.random.RandomState(0)
    D0 = init_dict(cfg.patch_dim, cfg.n_atoms, rng)
    D, Z, errors = _online_numpy(D0, task["X_train"], cfg.n_iter, cfg.batch_size, cfg.sparsity, rng)
    assert np.all(np.isfinite(D))
    assert D.shape == (cfg.patch_dim, cfg.n_atoms)
    assert Z.shape == (cfg.n_atoms, task["X_train"].shape[1])
    assert errors[-1] > 0


def test_online_beats_random_fixed_dict():
    # learning must beat a non-learned random dictionary on held-out data
    cfg, task = make_task()
    rng = np.random.RandomState(1)
    D0 = init_dict(cfg.patch_dim, cfg.n_atoms, rng)
    Drand = init_dict(cfg.patch_dim, cfg.n_atoms, np.random.RandomState(99))
    Dlearn, _, _ = _online_numpy(D0, task["X_train"], cfg.n_iter, cfg.batch_size, cfg.sparsity, rng)
    err_rand = reconstruction_nrmse(Drand, task["X_test"], cfg.sparsity)
    err_learn = reconstruction_nrmse(Dlearn, task["X_test"], cfg.sparsity)
    assert err_learn < err_rand


def test_available_sklearn_is_bool():
    assert isinstance(available_sklearn(), bool)
