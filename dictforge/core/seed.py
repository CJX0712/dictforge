"""Global deterministic seeding and BLAS thread pinning.

Determinism is a hard gate for this system: two runs with the same seed must
produce bit-identical core metrics. Multi-threaded BLAS/LAPACK summation order
is non-deterministic, so we pin every numerical thread to 1. This module MUST
be imported (and ``pin_threads`` called) *before* numpy is imported in any entry
script (cli.py / run_demo.py / conftest.py) so the environment variables take
effect at numpy import time.
"""

from __future__ import annotations

import os
import random

_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def pin_threads() -> None:
    """Force single-threaded BLAS/LAPACK to remove summation-order noise."""
    for key in _THREAD_KEYS:
        os.environ[key] = "1"


def set_all(seed: int) -> None:
    """Single entry-point for every stochastic source in the system."""
    seed = int(seed)
    random.seed(seed)
    # numpy is imported lazily to keep this callable even before numpy loads
    import numpy as np

    np.random.seed(seed)
    pin_threads()
