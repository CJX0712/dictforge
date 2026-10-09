"""End-to-end demo: run the benchmark, verify determinism, write benchmark.json.

Pin BLAS threads to 1 *before* importing numpy for deterministic summation.
"""

import os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_k] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root (dictforge/..)

from dictforge.core.config import Config  # noqa: E402
from dictforge.pipeline.benchmark import run_benchmark  # noqa: E402


def _core_scalars(result: dict) -> dict:
    out: dict = {}
    for m, d in result["per_method"].items():
        out[f"per_method.{m}.mean"] = d["mean"]
        out[f"per_method.{m}.std"] = d["std"]
    h = result["headline"]
    out["headline.reduction_vs_naive"] = h["reduction_vs_naive"]
    out["headline.non_inferior"] = float(h["non_inferior"])
    out["headline.significant"] = float(h["significant"])
    return out


def _deterministic(a: dict, b: dict) -> bool:
    sa, sb = _core_scalars(a), _core_scalars(b)
    keys = set(sa) & set(sb)
    max_diff = max(abs(sa[k] - sb[k]) for k in keys)
    return max_diff <= 1e-9


def main() -> int:
    cfg = Config()
    cfg.validate()

    t0 = time.time()
    result = run_benchmark(cfg)
    result["wall_clock_sec"] = round(time.time() - t0, 3)

    result2 = run_benchmark(cfg)
    result["deterministic"] = _deterministic(result, result2)

    out_path = os.path.join(os.path.dirname(_HERE), "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    h = result["headline"]
    print("DictForge demo complete.")
    print(f"  flagship ksvdfuse nRMSE = {h['flagship_ksvdfuse']['mean']:.4f}")
    print(f"  naive random dict nRMSE = {h['naive_random']['mean']:.4f}")
    print(f"  reduction vs naive      = {h['reduction_vs_naive'] * 100:.2f}% (gate >= 30%)")
    print(f"  non-inferior            = {h['non_inferior']}")
    print(f"  significant             = {h['significant']}")
    print(f"  GATE PASS               = {h['gate_pass']}")
    print(f"  deterministic           = {result['deterministic']}")
    print(f"  wall-clock              = {result['wall_clock_sec']}s")
    print(f"  written -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
