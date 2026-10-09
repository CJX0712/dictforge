# DictForge — Architecture

DictForge learns an **overcomplete dictionary** `D ∈ ℝ^(d×K)` (with `K > d`) and a
**sparse code** `Z ∈ ℝ^(K×N)` so that each signal column `xₙ ≈ D zₙ` with `‖zₙ‖₀ ≤ L`.
The system is organised into five layers plus a deterministic CLI.

---

## 1. Data layer (`dictforge/data`)

### 1.1 Synthetic task generator (`synthetic.py`)
`make_sparse_coded_task(cfg, dataset, seed)` builds a *ground-truth* dictionary
`D₀` and sparse codes `Z₀`, then emits noisy signals `X = D₀ Z₀ + ε`. Three
ground-truth generators:

- **gabor** — union-of-Gabor atoms (localized, oscillatory) — the canonical
  sparse-representation signal class.
- **randstruct** — Gaussian-blurred random sparse spikes — structured but
  non-oscillatory.
- **natural** — Gaussian-blurred natural-ish textures — broadband, smooth.

It returns a **three-way leakage-safe split**: `X_train`, `X_val`, `X_test` are
disjoint column blocks. The ground-truth `D₀ / Z₀` is **never handed to learners**
— learners must rediscover structure from `X_train` alone.

### 1.2 Loaders (`loaders.py`)
`available_sklearn()`, `split_train_val_test()`, `load_task_npz()` provide optional
real-data hooks (e.g. scikit-learn image patches) and NPZ round-tripping.

---

## 2. Dictionary-learning layer (`dictforge/dictlearn`)

| Module         | Engine            | Key idea |
|----------------|-------------------|----------|
| `omp.py`       | sparse coder      | greedy atom selection + least-squares re-solve; residual ⟂ atoms |
| `baselines.py` | `random`, `dct`   | untrained random frame; deterministic DCT-like frame |
| `mod.py`       | `mod`             | Method of Optimal Directions: `D ← pinv(Z) X`, renormalise |
| `online.py`    | `online`          | pure-NumPy block-coordinate online update (offline-safe) |
| `ksvd.py`      | `ksvd`            | per-atom SVD update of the residual; monotonic safeguard |
| `fuse.py`      | `ksvdfuse`        | cross-validated pick of best learned component |
| `learners.py`  | dispatcher        | `fit_dictionary(name, …)` routes to the right engine |

### 2.1 K-SVD detail
For each atom `k`: compute the residual `E = X − D Z` restricted to signals that
use atom `k`; take its SVD `E ≈ U Σ Vᵀ`; set `dₖ ← u₁` and `zₖ ← σ₁ v₁ᵀ`. After the
dictionary pass, OMP re-codes every signal; the algorithm **keeps the better of the
old/new code** so the training objective is *non-increasing* (a hard invariant).

### 2.2 KSvdFuse
Trains `mod`, `online`, `ksvd`; evaluates each on `X_val`; returns the dictionary
with the **lowest validation nRMSE**. By construction it is
`flagship ≤ 1.02 × best_component`, i.e. *non-inferior* to the best component, and
it structurally dominates the fixed (`random`/`dct`) baselines because it is
selected *from* the learned set.

---

## 3. Evaluation layer (`dictforge/eval`)

- `metrics.py` — `reconstruction_nrmse` (scale-free, divided by `‖X‖_F`) and
  `mean_sparsity`.
- `invariants.py` — the correctness firewall: `assert_finite`, `assert_unit_norm`,
  `assert_sparsity`, `assert_omp_residual_orthogonal`, `assert_monotone`,
  `assert_deterministic`, `assert_no_leak`.

---

## 4. Pipeline layer (`dictforge/pipeline`)

- `benchmark.py` — `run_benchmark(cfg)` fits every engine × dataset × seed, asserts
  invariants + leak guard on each row, then aggregates `per_method` and `headline`
  (naive / flagship / best_component / reduction / non-inferior / significant /
  gate_pass).
- `selftest.py` — `run_selftest()` runs 10 invariant gates and returns
  `all_pass`.

---

## 5. Determinism (`dictforge/core/seed.py`)

`core.seed.set_all(seed)` is the **single** seeding entry point (Python `random`,
`numpy`, `torch` if present). Critically, `cli.py` pins `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` to `1` *before* importing numpy, so multi-threaded BLAS
summation order cannot introduce run-to-run nondeterminism.

---

## 6. CLI (`dictforge/cli.py`)

```
python -m dictforge.cli [demo|run|selftest] [--seeds N] [--out PATH]
```
`demo`/`run` execute the benchmark **twice** and assert the core scalars are
bit-identical (determinism gate), then write `benchmark.json` and print a summary.
`selftest` runs the invariant firewall.

---

## Dependency & offline posture

Pure NumPy / SciPy / scikit-learn; **no mandatory network and no GPU**. The
`online` engine uses a pure-NumPy block-coordinate path (not the sklearn solver) so
the whole stack is reproducible and offline-safe on a CPU.
