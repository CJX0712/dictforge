# DictForge

> 世界顶级字典学习 / 稀疏编码 AI 系统 —— 作者：晨星
> World-class Dictionary Learning / Sparse Coding AI system.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![CI](https://github.com/CJX0712/dictforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/dictforge/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-30%2F30%20passing-brightgreen)](./tests)
[![Quality Gate](https://img.shields.io/badge/quality%20gate-PASS%20(S)-success)](./docs/model_card.md)

**DictForge** learns overcomplete dictionaries from data and reconstructs signals
with a bounded sparsity budget. It implements the strongest classical sparse-coding
algorithms — **K-SVD** (Aharon, Elad & Bruckstein, 2006), **MOD** (Engan et al.,
1999), and **Online Dictionary Learning** (Mairal et al., 2009) — plus a flagship
**KSvdFuse** engine that cross-validates among the learned components and is
*structurally non-inferior* to the best of them by construction, while
*structurally dominating* non-learned (fixed) baselines.

---

## ✨ Features

- **Four dictionary-learning engines** — `mod`, `online`, `ksvd`, and the fused
  **`ksvdfuse`** flagship.
- **Two strong fixed baselines** — `random` (untrained random dictionary) and
  `dct` (deterministic DCT-like frame), used only for honest ablation.
- **Orthogonal Matching Pursuit (OMP)** sparse coder with a proven residual
  orthogonality invariant.
- **Hard mathematical invariant gates** (the correctness firewall): finite values,
  unit-norm atoms, sparsity budget, monotonic K-SVD training error, deterministic
  seeding, and a three-way leakage-safe data split (train/val/test disjoint).
- **Determinism by construction** — single `core.seed.set_all(seed)` entry point;
  BLAS/LAPACK threads pinned to 1 so summation order is reproducible across runs.
- **One-click reproducible benchmark** — `python -m dictforge.cli demo` writes
  `benchmark.json` and prints a summary; the same seed yields bit-identical core
  metrics.
- **Zero mandatory network / zero GPU** — pure NumPy/SciPy, CPU-only demo in
  < 60 s.

---

## 📊 Benchmark (committed `benchmark.json`, 3 seeds × 3 datasets)

Normalised RMSE (nRMSE, lower is better). Flagship **ksvdfuse = 0.1240** vs naive
**random = 0.5614** → **77.90 % reduction** (gate ≥ 30 %).

| Method       | mean nRMSE | std   | Role                       |
|--------------|-----------:|------:|----------------------------|
| random       | 0.5614     | 0.0117| naive baseline (untrained) |
| dct          | 0.4223     | 0.1306| fixed baseline (DCT frame) |
| mod          | 0.1272     | 0.0225| learned component          |
| online       | 0.1262     | 0.0220| learned component (best)   |
| ksvd         | 0.1747     | 0.0155| learned component          |
| **ksvdfuse** | **0.1240** | 0.0225| **flagship (fused)**       |

- **reduction vs naive**: 77.90 %
- **non-inferior** (flagship ≤ 1.02 × best component): ✅ True
- **significant** (vs naive): ✅ True
- **gate_pass**: ✅ True
- **deterministic**: ✅ True
- **wall-clock (CPU, 3 seeds)**: 24.5 s

---

## 🚀 Quick start

```bash
# 1. Clone & install
git clone https://github.com/CJX0712/dictforge.git
cd dictforge
pip install -e ".[dev]"

# 2. Mathematical invariant self-test (10 gates)
python -m dictforge.cli selftest

# 3. Full benchmark -> benchmark.json
python -m dictforge.cli demo
```

### Library usage

```python
from dictforge.core.config import Config
from dictforge.dictlearn.learners import fit_dictionary
from dictforge.data.synthetic import make_sparse_coded_task
from dictforge.eval.metrics import reconstruction_nrmse

cfg = Config.from_env()
task = make_sparse_coded_task(cfg, "gabor", seed=7)
D, info = fit_dictionary("ksvdfuse", task["X_train"], task["X_val"], cfg, seed_int=7)
print("test nRMSE:", reconstruction_nrmse(D, task["X_test"], cfg.sparsity))
```

---

## 🧮 Mathematical invariants (the correctness firewall)

Every engine is checked against the following hard invariants before any number is
reported:

| Invariant | Definition | Why it matters |
|-----------|------------|----------------|
| `finite`        | all dictionary / code entries finite | no NaN/Inf leaks |
| `unit_norm`     | every atom ‖dₖ‖₂ = 1 | standard dict-learning normalisation |
| `sparsity`      | each code has ≤ L non-zeros | honours the sparsity budget |
| `omp_ortho`     | OMP residual ⟂ selected atoms | defines a *correct* OMP |
| `monotone`      | K-SVD training error non-increasing | guarantees objective descent |
| `deterministic` | same seed → identical core metrics | reproducible science |
| `no_leak`       | train ∩ test columns = ∅ | no data leakage |

---

## 🏗️ Architecture

```
dictforge/
  core/        config, seed (deterministic), errors, types, interfaces
  data/        synthetic sparse-coded task generation + sklearn loaders
  dictlearn/   omp, ksvd, mod, online, baselines, fuse, learners
  eval/        metrics (nRMSE, sparsity), invariants (gates)
  pipeline/    benchmark (multi-seed/multi-dataset), selftest
  cli.py       deterministic CLI (demo / run / selftest)
  examples/    run_demo.py
```

See [`docs/architecture.md`](./docs/architecture.md) for the full design and
[`docs/model_card.md`](./docs/model_card.md) for the model card.

---

## 📄 License & author

MIT License — Copyright (c) 2026 **晨星**.

Built as part of a world-class open-source AI series. Author: **晨星**.
