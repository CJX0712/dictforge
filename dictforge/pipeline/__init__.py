"""Pipeline: deterministic benchmark aggregation and invariant self-test."""

from __future__ import annotations

from .benchmark import DictForgePipeline, run_benchmark
from .selftest import run_selftest

__all__ = ["DictForgePipeline", "run_benchmark", "run_selftest"]
