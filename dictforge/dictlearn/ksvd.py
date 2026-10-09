"""K-SVD dictionary learning (Aharon, Elad, Bruckstein 2006).

Alternates OMP sparse coding with a per-atom dictionary update that minimises
the Frobenius reconstruction error for the fixed support via a rank-1 SVD. The
training reconstruction error is monotonically non-increasing -- a hard gate.
"""

from __future__ import annotations

import numpy as np

from .omp import omp_batch


def ksvd(
    D0: np.ndarray,
    X: np.ndarray,
    L: int,
    n_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run K-SVD.

    Returns ``(D, Z, errors)`` where ``errors`` is the per-iteration Frobenius
    reconstruction error on ``X`` (monotonically non-increasing).
    """
    D = D0.astype(float).copy()
    Z = omp_batch(D, X, L)
    errors: list[float] = [float(np.linalg.norm(X - D @ Z))]
    for _ in range(n_iter):
        for k in range(D.shape[1]):
            wk = np.nonzero(Z[k, :] != 0)[0]
            if wk.size < 2:
                # too few samples using this atom -> leave it (or re-init)
                continue
            Dnk = D.copy()
            Dnk[:, k] = 0.0
            Ek = X[:, wk] - Dnk @ Z[:, wk]
            U, S, Vt = np.linalg.svd(Ek, full_matrices=False)
            D[:, k] = U[:, 0]
            Z[k, wk] = S[0] * Vt[0, :]
        # Re-code with OMP, but never allow the training error to increase:
        # OMP is greedy and can occasionally pick a worse L-term support, so we
        # keep the better of the old and new codes. This guarantees the training
        # objective is monotonically non-increasing.
        err_prev = errors[-1]
        Z_new = omp_batch(D, X, L)
        err_new = float(np.linalg.norm(X - D @ Z_new))
        if err_new <= err_prev:
            Z = Z_new
            errors.append(err_new)
        else:
            errors.append(float(np.linalg.norm(X - D @ Z)))
        if len(errors) >= 2 and abs(errors[-2] - errors[-1]) < tol * errors[0]:
            break
    return D, Z, errors
