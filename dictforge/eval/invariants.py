"""Hard mathematical invariants -- the system's correctness gate.

Each returns ``None`` on success or raises ``NumericalError`` / ``DataLeakError``.
They are exercised by ``selftest`` and the test-suite as CI gates.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataLeakError, NumericalError


def assert_finite(arr: np.ndarray, name: str = "") -> None:
    if not np.all(np.isfinite(arr)):
        raise NumericalError(f"non-finite values in {name or 'array'}")


def assert_unit_norm(D: np.ndarray, tol: float = 1e-6) -> None:
    norms = np.linalg.norm(D, axis=0)
    dev = float(np.max(np.abs(norms - 1.0)))
    if dev > tol:
        raise NumericalError(f"dictionary atoms not unit norm (max dev {dev:.2e})")


def assert_sparsity(Z: np.ndarray, L: int, tol: int = 0) -> None:
    max_nz = int(np.max(np.count_nonzero(Z, axis=0))) if Z.size else 0
    if max_nz > L + tol:
        raise NumericalError(f"code sparsity {max_nz} exceeds budget L={L}")


def assert_omp_residual_orthogonal(
    D: np.ndarray, x: np.ndarray, z: np.ndarray, tol: float = 1e-6
) -> None:
    support = np.nonzero(z)[0]
    if support.size == 0:
        return
    r = x - D @ z
    proj = D[:, support].T @ r
    dev = float(np.max(np.abs(proj)))
    if dev > tol:
        raise NumericalError(f"OMP residual not orthogonal to selected atoms (max proj {dev:.2e})")


def assert_monotone(errors: list[float], tol: float = 1e-6) -> None:
    """Training error must be non-increasing across iterations (diffs <= 0)."""
    errs = np.asarray(errors, dtype=float)
    if errs.size < 2:
        return
    diffs = np.diff(errs)
    if float(np.max(diffs)) > tol * max(1.0, abs(errs[0])):
        raise NumericalError(
            f"training error increased across iterations (max diff {float(np.max(diffs)):.2e})"
        )


def assert_deterministic(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> None:
    dev = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    if dev > tol:
        raise NumericalError(f"determinism check failed (max diff {dev:.2e} > {tol})")


def assert_no_leak(train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    if {int(i) for i in train_idx} & {int(i) for i in test_idx}:
        raise DataLeakError("train/test sample indices overlap (data leak)")
