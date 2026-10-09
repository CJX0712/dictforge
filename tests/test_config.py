"""Configuration: ENV overrides and validation."""

import pytest

from dictforge.core.config import Config
from dictforge.core.errors import ConfigError


def test_env_override_int(monkeypatch):
    monkeypatch.setenv("ENV_SEED", "42")
    monkeypatch.setenv("ENV_N_ATOMS", "100")
    cfg = Config.from_env()
    assert cfg.seed == 42
    assert cfg.n_atoms == 100


def test_env_override_float(monkeypatch):
    monkeypatch.setenv("ENV_TOL", "0.001")
    cfg = Config.from_env()
    assert abs(cfg.tol - 0.001) < 1e-12


def test_env_invalid_value_raises(monkeypatch):
    monkeypatch.setenv("ENV_SEED", "not-an-int")
    with pytest.raises(ConfigError):
        Config.from_env()


def test_validation_overcomplete_required():
    cfg = Config(n_atoms=10, patch_dim=20)  # not overcomplete
    with pytest.raises(ConfigError):
        cfg.validate()


def test_validation_bad_sparsity():
    cfg = Config(sparsity=0)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_validation_negative_seed():
    cfg = Config(seed=-1)
    with pytest.raises(ConfigError):
        cfg.validate()
