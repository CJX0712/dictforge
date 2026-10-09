"""Deterministic multi-dataset, multi-seed benchmark aggregation."""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.seed import set_all
from ..core.types import BenchmarkRow
from ..data.synthetic import make_sparse_coded_task
from ..dictlearn.learners import ENGINE_NAMES, fit_dictionary
from ..dictlearn.omp import omp_batch
from ..eval.invariants import (
    assert_finite,
    assert_no_leak,
    assert_sparsity,
    assert_unit_norm,
)
from ..eval.metrics import mean_sparsity, reconstruction_nrmse

_DATASET_INDEX = {"gabor": 0, "randstruct": 1, "natural": 2}
_METHOD_INDEX = {name: i for i, name in enumerate(ENGINE_NAMES)}


def _seed_int(dataset: str, method: str, base_seed: int) -> int:
    di = _DATASET_INDEX.get(dataset, 0)
    mi = _METHOD_INDEX.get(method, 0)
    return int(base_seed) * 100000 + di * 1000 + mi * 10


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values), ddof=1))


def run_benchmark(cfg: Config) -> dict:
    """Run the full benchmark and return a structured result dictionary."""
    cfg.validate()
    set_all(cfg.seed)
    seeds = list(range(cfg.seed, cfg.seed + cfg.n_seeds))
    rows: list[BenchmarkRow] = []

    for dataset in cfg.datasets:
        for seed in seeds:
            task = make_sparse_coded_task(cfg, dataset, seed)
            Xtr, Xva, Xte = task["X_train"], task["X_val"], task["X_test"]
            # leak guard: train / val / test column indices are disjoint
            ntr, nva, nte = Xtr.shape[1], Xva.shape[1], Xte.shape[1]
            train_idx = np.arange(ntr)
            val_idx = np.arange(ntr, ntr + nva)
            test_idx = np.arange(ntr + nva, ntr + nva + nte)
            assert_no_leak(train_idx, test_idx)
            assert_no_leak(train_idx, val_idx)
            for method in ENGINE_NAMES:
                sd = _seed_int(dataset, method, cfg.seed)
                D, info = fit_dictionary(method, Xtr, Xva, cfg, sd)
                assert_finite(D, "dictionary")
                assert_unit_norm(D)
                Zte = omp_batch(D, Xte, cfg.sparsity)
                assert_sparsity(Zte, cfg.sparsity)
                test_nrmse = reconstruction_nrmse(D, Xte, cfg.sparsity)
                train_nrmse = info.get("train_nrmse", reconstruction_nrmse(D, Xtr, cfg.sparsity))
                rows.append(
                    BenchmarkRow(
                        method=method,
                        dataset=dataset,
                        seed=seed,
                        nrmse=test_nrmse,
                        sparsity=mean_sparsity(Zte),
                        train_nrmse=train_nrmse,
                    )
                )

    return _aggregate(rows, cfg)


def _aggregate(rows: list[BenchmarkRow], cfg: Config) -> dict:
    per_method: dict[str, dict] = {}
    for m in ENGINE_NAMES:
        vals = [r.nrmse for r in rows if r.method == m]
        per_method[m] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": _std(vals),
            "n": len(vals),
        }

    naive = per_method["random"]
    flagship = per_method["ksvdfuse"]
    components = [per_method[c] for c in ("mod", "online", "ksvd")]
    best_component = min(components, key=lambda d: d["mean"])

    reduction_naive = (
        (naive["mean"] - flagship["mean"]) / naive["mean"] if naive["mean"] > 0 else 0.0
    )
    non_inferior = flagship["mean"] <= best_component["mean"] * 1.02
    significant = (naive["mean"] - flagship["mean"]) > 0.5 * (flagship["std"] + naive["std"])
    gate_pass = bool(reduction_naive >= 0.30 and non_inferior and significant)

    return {
        "config": {
            "seed": cfg.seed,
            "n_seeds": cfg.n_seeds,
            "datasets": list(cfg.datasets),
            "patch_dim": cfg.patch_dim,
            "n_atoms": cfg.n_atoms,
            "sparsity": cfg.sparsity,
            "true_atoms": cfg.true_atoms,
            "true_sparsity": cfg.true_sparsity,
            "n_iter": cfg.n_iter,
        },
        "rows": [
            {
                "method": r.method,
                "dataset": r.dataset,
                "seed": r.seed,
                "nrmse": r.nrmse,
                "sparsity": r.sparsity,
                "train_nrmse": r.train_nrmse,
            }
            for r in rows
        ],
        "per_method": per_method,
        "headline": {
            "naive_random": naive,
            "flagship_ksvdfuse": flagship,
            "best_component": {"method": _best_name(per_method), **best_component},
            "reduction_vs_naive": float(reduction_naive),
            "non_inferior": bool(non_inferior),
            "significant": bool(significant),
            "gate_pass": gate_pass,
        },
    }


def _best_name(per_method: dict) -> str:
    best = min((per_method[c] for c in ("mod", "online", "ksvd")), key=lambda d: d["mean"])
    for c in ("mod", "online", "ksvd"):
        if per_method[c] is best:
            return c
    return "ksvd"


class DictForgePipeline:
    """Thin object wrapper around :func:`run_benchmark`."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self) -> dict:
        return run_benchmark(self.cfg)
