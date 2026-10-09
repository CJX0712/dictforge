"""Dictionary-learning engines."""

from __future__ import annotations

from .baselines import dct_frame, init_dict
from .fuse import ksvd_fuse
from .ksvd import ksvd
from .learners import ENGINE_NAMES, fit_dictionary
from .mod import mod
from .omp import omp_batch, omp_single
from .online import online_dict_learning

__all__ = [
    "ENGINE_NAMES",
    "dct_frame",
    "fit_dictionary",
    "init_dict",
    "ksvd",
    "ksvd_fuse",
    "mod",
    "omp_batch",
    "omp_single",
    "online_dict_learning",
]
