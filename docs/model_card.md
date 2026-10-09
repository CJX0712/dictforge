# Model Card — DictForge

**Model name:** DictForge (dictionary-learning / sparse-coding system)
**Version:** 0.1.0
**Author:** 晨星
**Date:** 2026-10-10
**License:** MIT

---

## 1. Model details

- **Type:** Classical representation-learning system (unsupervised dictionary
  learning + sparse coding). No deep network; pure NumPy/SciPy numerical core.
- **Engines:** `random`, `dct` (fixed baselines); `mod`, `online`, `ksvd`
  (learned components); `ksvdfuse` (flagship, cross-validated fusion).
- **Sparse coder:** Orthogonal Matching Pursuit (OMP) with sparsity budget `L`.
- **Objective:** minimise `‖X − D Z‖_F` subject to `‖zₙ‖₀ ≤ L`, with unit-norm
  atom constraint.

## 2. Intended use

- Learning compact, interpretable overcomplete representations of signals
  (image patches, audio, time-series, sensor data).
- Denoising, inpainting, compression, and feature extraction where an explicit
  dictionary is desirable.
- Teaching / research baseline for sparse coding (clean, invariant-guarded code).

## 3. Out-of-scope use

- Not a generative foundation model; no semantic/text understanding.
- Not for safety-critical control without independent validation.
- Not trained on personal or protected data.

## 4. Training / evaluation data

- **Data source:** synthetic sparse-coded signals generated in-repository
  (`data/synthetic.py`). Three ground-truth families: `gabor`, `randstruct`,
  `natural`.
- **Split:** train (420) / val (100) / test (180) columns, **disjoint** — leakage
  guard enforced in code.
- **Scale:** patch dimension `d = 24`, `K = 48` atoms, sparsity `L = 5`,
  `n_iter = 5`, `n_seeds = 3`.
- **No external data required; no PII.**

## 5. Evaluation metrics

- **Primary:** normalised RMSE (nRMSE) on the held-out test split, scale-free.
- **Reported:** per-method mean ± std nRMSE across 3 seeds × 3 datasets.

### Headline results (committed `benchmark.json`)

| Method       | mean nRMSE | std   |
|--------------|-----------:|------:|
| random       | 0.5614     | 0.0117|
| dct          | 0.4223     | 0.1306|
| mod          | 0.1272     | 0.0225|
| online       | 0.1262     | 0.0220|
| ksvd         | 0.1747     | 0.0155|
| **ksvdfuse** | **0.1240** | 0.0225|

- reduction vs naive: **77.90 %** (gate ≥ 30 %)
- non-inferior to best component: ✅
- significant vs naive: ✅
- deterministic: ✅
- demo wall-clock (CPU): 24.5 s

## 6. Limitations

- Synthetically evaluated; real-world signals may need larger `K`, more
  iterations, and real-image patch datasets.
- OMP sparsity budget `L` is fixed; adaptive `L` is future work.
- Pure-NumPy `online` engine trades some asymptotic speed for offline determinism.

## 7. Ethical considerations

- Fully synthetic, reproducible, and offline; no data-collection or surveillance
  implications.
- Open-source (MIT) for transparent, auditable science.

## 8. Citation

If you use DictForge, please cite:

```
@software{dictforge2026,
  title  = {DictForge: World-class Dictionary Learning / Sparse Coding AI},
  author = {晨星},
  year   = {2026},
  url    = {https://github.com/CJX0712/dictforge}
}
```
