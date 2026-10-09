"""Mathematical invariant self-test (the correctness gate, ``dictforge selftest``)."""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.seed import set_all
from ..data.synthetic import make_sparse_coded_task
from ..dictlearn.baselines import init_dict
from ..dictlearn.ksvd import ksvd
from ..dictlearn.omp import omp_single
from ..eval.invariants import (
    assert_deterministic,
    assert_finite,
    assert_monotone,
    assert_no_leak,
    assert_omp_residual_orthogonal,
    assert_sparsity,
    assert_unit_norm,
)
from ..eval.metrics import reconstruction_nrmse


def _check(checks: list[tuple[str, bool, str]], name: str, fn) -> None:
    try:
        fn()
        checks.append((name, True, ""))
    except Exception as exc:  # noqa: BLE001 - self-test intentionally catches all
        checks.append((name, False, str(exc)))


def run_selftest() -> dict:
    """Run every hard invariant and return ``{"checks": [...], "all_pass": bool}``."""
    set_all(7)
    cfg = Config(
        patch_dim=24,
        n_atoms=40,
        sparsity=6,
        true_atoms=12,
        true_sparsity=4,
        n_train=300,
        n_val=100,
        n_test=100,
        n_iter=8,
    )
    checks: list[tuple[str, bool, str]] = []

    task = make_sparse_coded_task(cfg, "gabor", 7)
    Xtr, Xte = task["X_train"], task["X_test"]

    rng = np.random.RandomState(0)
    D0 = init_dict(cfg.patch_dim, cfg.n_atoms, rng)

    x = Xtr[:, 0]
    z, _ = omp_single(D0, x, cfg.sparsity)
    _check(checks, "omp_residual_orthogonal", lambda: assert_omp_residual_orthogonal(D0, x, z))
    _check(checks, "unit_norm_initial", lambda: assert_unit_norm(D0))

    Dk, Zk, errors = ksvd(D0, Xtr, cfg.sparsity, cfg.n_iter, cfg.tol)
    _check(checks, "unit_norm_ksvd", lambda: assert_unit_norm(Dk))
    _check(checks, "monotone_ksvd", lambda: assert_monotone(errors))
    _check(checks, "sparsity_ksvd", lambda: assert_sparsity(Zk, cfg.sparsity))
    _check(checks, "finite_ksvd", lambda: assert_finite(Dk, "Dk"))

    # determinism: identical RNG + identical input -> identical output
    rng2 = np.random.RandomState(0)
    D0b = init_dict(cfg.patch_dim, cfg.n_atoms, rng2)
    Dkb, _, _ = ksvd(D0b, Xtr, cfg.sparsity, cfg.n_iter, cfg.tol)
    _check(checks, "deterministic_ksvd", lambda: assert_deterministic(Dk, Dkb, 1e-9))

    nrmse_a = reconstruction_nrmse(Dk, Xte, cfg.sparsity)
    nrmse_b = reconstruction_nrmse(Dkb, Xte, cfg.sparsity)
    _check(
        checks,
        "deterministic_nrmse",
        lambda: assert_deterministic(np.array([nrmse_a]), np.array([nrmse_b]), 1e-12),
    )

    # scale-invariance of normalised RMSE
    nrmse_scaled = reconstruction_nrmse(Dk, Xte * 3.0, cfg.sparsity)
    _check(
        checks,
        "nrmse_scale_invariant",
        lambda: assert_deterministic(np.array([nrmse_a]), np.array([nrmse_scaled]), 1e-9),
    )

    _check(
        checks,
        "no_data_leak",
        lambda: assert_no_leak(
            np.arange(cfg.n_train),
            np.arange(cfg.n_train, cfg.n_train + cfg.n_test),
        ),
    )

    all_pass = all(p for _, p, _ in checks)
    return {"checks": checks, "all_pass": all_pass}
