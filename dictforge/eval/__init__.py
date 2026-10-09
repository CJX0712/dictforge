"""Evaluation: metrics and hard mathematical invariants."""

from __future__ import annotations

from .invariants import (
    assert_deterministic,
    assert_finite,
    assert_monotone,
    assert_no_leak,
    assert_omp_residual_orthogonal,
    assert_sparsity,
    assert_unit_norm,
)
from .metrics import mean_sparsity, reconstruction_nrmse

__all__ = [
    "assert_deterministic",
    "assert_finite",
    "assert_monotone",
    "assert_no_leak",
    "assert_omp_residual_orthogonal",
    "assert_sparsity",
    "assert_unit_norm",
    "mean_sparsity",
    "reconstruction_nrmse",
]
