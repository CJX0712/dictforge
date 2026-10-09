"""DictForge command-line entry point.

BLAS/LAPACK threads are pinned to 1 at the very top, *before* numpy is
imported, so summation order is deterministic across runs.
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

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from .core.config import Config  # noqa: E402
from .pipeline.benchmark import run_benchmark  # noqa: E402
from .pipeline.selftest import run_selftest  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_benchmark_path() -> str:
    return os.path.join(_REPO_ROOT, "benchmark.json")


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


def _compare_deterministic(a: dict, b: dict) -> bool:
    sa, sb = _core_scalars(a), _core_scalars(b)
    keys = set(sa) & set(sb)
    max_diff = 0.0
    for k in keys:
        max_diff = max(max_diff, abs(sa[k] - sb[k]))
    return max_diff <= 1e-9


def _print_summary(result: dict) -> None:
    h = result["headline"]
    print("=" * 64)
    print("DictForge benchmark summary")
    print("=" * 64)
    print(f"  datasets           : {result['config']['datasets']}")
    print(f"  seeds              : {result['config']['n_seeds']}")
    print(f"  patch_dim / atoms  : {result['config']['patch_dim']} / {result['config']['n_atoms']}")
    print(f"  sparsity budget L  : {result['config']['sparsity']}")
    print("-" * 64)
    print("  method            mean nRMSE   std")
    for m in result["per_method"]:
        d = result["per_method"][m]
        print(f"  {m:16s}  {d['mean']:.4f}     {d['std']:.4f}")
    print("-" * 64)
    print(f"  reduction vs naive : {h['reduction_vs_naive'] * 100:6.2f}%  (gate >= 30%)")
    print(f"  non-inferior      : {h['non_inferior']}")
    print(f"  significant       : {h['significant']}")
    print(f"  GATE PASS         : {h['gate_pass']}")
    print(f"  deterministic      : {result.get('deterministic')}")
    print(f"  wall-clock        : {result.get('wall_clock_sec')}s")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dictforge", description="DictForge CLI")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    sub = parser.add_subparsers(dest="cmd")

    p_demo = sub.add_parser("demo", help="run full benchmark -> benchmark.json")
    p_demo.add_argument("--seeds", type=int, default=None)
    p_demo.add_argument("--out", type=str, default=None)

    p_run = sub.add_parser("run", help="alias for demo")
    p_run.add_argument("--seeds", type=int, default=None)
    p_run.add_argument("--out", type=str, default=None)

    sub.add_parser("selftest", help="run mathematical invariant self-test")

    args = parser.parse_args(argv)
    cmd = args.cmd or "demo"

    if cmd == "selftest":
        res = run_selftest()
        for name, ok, detail in res["checks"]:
            mark = "PASS" if ok else "FAIL"
            line = f"[{mark}] {name}"
            if detail and not ok:
                line += f" -- {detail}"
            print(line)
        print("ALL_PASS" if res["all_pass"] else "SOME_FAILED")
        return 0 if res["all_pass"] else 1

    cfg = Config.from_env()
    n_seeds = getattr(args, "seeds", None)
    if n_seeds:
        cfg.n_seeds = n_seeds
    cfg.validate()

    t0 = time.time()
    result = run_benchmark(cfg)
    result["wall_clock_sec"] = round(time.time() - t0, 3)

    result2 = run_benchmark(cfg)
    result["deterministic"] = _compare_deterministic(result, result2)

    out = args.out or _default_benchmark_path()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
