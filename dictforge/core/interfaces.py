"""Stable interfaces (Protocol) for dictionary learners."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DictLearner(Protocol):
    """Any object that learns a dictionary and sparsely codes new data."""

    name: str

    def fit(self, X: Any) -> DictLearner:
        """Learn a dictionary from training samples ``X`` (d x N)."""
        ...

    def transform(self, X: Any) -> Any:
        """Return sparse codes ``Z`` (K x N) for samples ``X``."""
        ...
