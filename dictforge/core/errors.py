"""Error taxonomy with stable codes (E100..E500)."""

from __future__ import annotations


class DictForgeError(Exception):
    """Base error for DictForge with a stable code."""

    code = "E000"

    def __init__(self, message: str = "", code: str | None = None) -> None:
        if code is not None:
            self.code = code
        super().__init__(f"[{self.code}] {message}")


class ConfigError(DictForgeError):
    """Invalid configuration or environment override."""

    code = "E100"


class ShapeError(DictForgeError):
    """Array shape contract violated."""

    code = "E200"


class NumericalError(DictForgeError):
    """Non-finite or degenerate numerical result."""

    code = "E300"


class DataLeakError(DictForgeError):
    """Holdout / test data would leak into model fitting."""

    code = "E500"
