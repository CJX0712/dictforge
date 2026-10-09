# Changelog

All notable changes to DictForge are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-10-10

### Added
- Dictionary-learning engines: **MOD**, **Online** (pure-NumPy block-coordinate),
  **K-SVD** with monotonic training-error safeguard, and flagship **KSvdFuse**
  (cross-validated selection among learned components).
- Fixed baselines: **random** (untrained) and **dct** (deterministic DCT-like
  frame) for honest ablation.
- Orthogonal Matching Pursuit (`omp.py`) sparse coder with residual-orthogonality
  invariant.
- Mathematical invariant firewall (`eval/invariants.py`): finite, unit-norm,
  sparsity budget, OMP orthogonality, K-SVD monotonicity, determinism, and
  leakage guard.
- Deterministic seeding (`core/seed.py`) with BLAS/LAPACK thread pinning to 1.
- Three-way leakage-safe data split (train / val / test disjoint column indices).
- CLI: `demo`, `run`, `selftest` with `benchmark.json` output and determinism check.
- Self-test (`pipeline/selftest.py`) running 10 invariant gates.
- CI workflow (Python 3.12 / 3.13 × Ubuntu / Windows), Makefile, Dockerfile,
  dependency lock, MIT license, README (5 badges), architecture + model card docs.

### Quality gate (S)
- 30/30 unit tests passing.
- Flagship nRMSE 0.1240 vs naive 0.5614 → 77.90 % reduction (gate ≥ 30 %).
- Non-inferior to best component; significant vs naive; deterministic.
- Demo wall-clock 24.5 s on CPU (≤ 60 s gate).

---

## [Unreleased]
- (planned) GPU kernels, larger real-image datasets, streaming online update API.
