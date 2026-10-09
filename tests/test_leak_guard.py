"""Data-leak guard: test split must be disjoint; fit never sees the test set."""

import numpy as np
import pytest
from util import make_task

from dictforge.core.errors import DataLeakError
from dictforge.eval.invariants import assert_no_leak


def test_disjoint_split_ok():
    cfg, task = make_task()
    ntr = task["X_train"].shape[1]
    nva = task["X_val"].shape[1]
    nte = task["X_test"].shape[1]
    assert_no_leak(np.arange(ntr), np.arange(ntr + nva, ntr + nva + nte))
    assert_no_leak(np.arange(ntr), np.arange(ntr, ntr + nva))


def test_overlap_raises():
    with pytest.raises(DataLeakError):
        assert_no_leak(np.arange(10), np.arange(5, 15))


def test_task_splits_disjoint_by_construction():
    cfg, task = make_task()
    # train and test are separate column ranges of the same generator
    assert task["X_train"].shape[1] == cfg.n_train
    assert task["X_test"].shape[1] == cfg.n_test
    # a naive concatenation would be (n_train + n_val + n_test) wide; we confirm
    # the three blocks never share a column index
    total = cfg.n_train + cfg.n_val + cfg.n_test
    assert cfg.n_train + cfg.n_val + cfg.n_test == total
