"""Data generation and loading."""

from __future__ import annotations

from .loaders import available_sklearn, load_task_npz, split_train_val_test
from .synthetic import make_sparse_coded_task

__all__ = [
    "available_sklearn",
    "load_task_npz",
    "make_sparse_coded_task",
    "split_train_val_test",
]
