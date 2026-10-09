"""Configuration with ENV_XXX_* overrides and schema validation.

Note: PEP 563 makes ``dataclass.fields(...).type`` a *string*, so we never
rely on ``f.type`` for casting. Instead we map each field name to its target
python type explicitly (see ``_TYPES``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from .errors import ConfigError

_TYPES = {
    "seed": int,
    "n_train": int,
    "n_test": int,
    "n_val": int,
    "patch_dim": int,
    "n_atoms": int,
    "sparsity": int,
    "true_atoms": int,
    "true_sparsity": int,
    "n_iter": int,
    "batch_size": int,
    "n_seeds": int,
    "tol": float,
}


@dataclass
class Config:
    """System-wide configuration.

    Every field can be overridden via ``ENV_<NAME>`` (e.g. ``ENV_SEED=11``).
    """

    seed: int = 7
    n_train: int = 420
    n_test: int = 180
    n_val: int = 100
    patch_dim: int = 24
    n_atoms: int = 48
    sparsity: int = 5
    true_atoms: int = 12
    true_sparsity: int = 3
    n_iter: int = 5
    batch_size: int = 120
    n_seeds: int = 3
    datasets: tuple = ("gabor", "randstruct", "natural")
    tol: float = 1e-4

    @classmethod
    def from_env(cls) -> Config:
        cfg = cls()
        for fld in fields(cls):
            env_key = "ENV_" + fld.name.upper()
            if env_key not in os.environ:
                continue
            target = _TYPES.get(fld.name)
            if target is None:
                continue
            raw = os.environ[env_key]
            try:
                setattr(cfg, fld.name, target(raw))
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"invalid ENV {env_key}={raw!r}: {exc}") from None
        return cfg

    def validate(self) -> None:
        if self.patch_dim <= 0:
            raise ConfigError("patch_dim must be > 0")
        if self.n_atoms <= self.patch_dim:
            raise ConfigError("n_atoms must be > patch_dim (overcomplete)")
        if self.sparsity <= 0 or self.sparsity >= self.n_atoms:
            raise ConfigError("sparsity must satisfy 0 < sparsity < n_atoms")
        if self.true_atoms <= 0 or self.true_atoms > self.n_atoms:
            raise ConfigError("true_atoms must satisfy 0 < true_atoms <= n_atoms")
        if self.true_sparsity <= 0 or self.true_sparsity >= self.true_atoms:
            raise ConfigError("true_sparsity must satisfy 0 < true_sparsity < true_atoms")
        if self.n_iter <= 0:
            raise ConfigError("n_iter must be > 0")
        if self.seed < 0:
            raise ConfigError("seed must be >= 0")
        if self.n_seeds <= 0:
            raise ConfigError("n_seeds must be > 0")
