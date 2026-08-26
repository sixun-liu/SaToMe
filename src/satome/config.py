"""Validated configuration contracts for the public integrations."""

from __future__ import annotations

from dataclasses import dataclass


def _validate_odd_kernel(kernel_size: int) -> None:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")


@dataclass(frozen=True)
class T2IConfig:
    """Paper-default configuration for the LLaDA-o T2I path."""

    total_steps: int = 50
    window_start: int = 16
    window_end: int = 45
    cell_size: int = 2
    protect_ratio: float = 0.35
    alpha: float = 0.95
    salience_kernel_size: int = 11

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= self.window_start < self.window_end <= self.total_steps:
            raise ValueError(
                "merge window must satisfy "
                f"0 <= start < end <= total_steps, got [{self.window_start}, {self.window_end}]/{self.total_steps}"
            )
        if self.cell_size != 2:
            raise ValueError("the released T2I method currently supports 2x2 cells only")
        if not 0.0 <= self.protect_ratio <= 1.0:
            raise ValueError("protect_ratio must be in [0, 1]")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        _validate_odd_kernel(self.salience_kernel_size)

    @property
    def visual_retention(self) -> float:
        """Fraction of visual tokens retained inside the active window."""

        return 0.5 * (1.0 + self.protect_ratio)


@dataclass(frozen=True)
class I2TConfig:
    """Configuration shared by the LaViDa and LLaDA-o I2T paths."""

    retention: float = 0.5
    merge_layer: int = 1
    salience_kernel_size: int = 11

    def __post_init__(self) -> None:
        if not 0.0 < self.retention <= 1.0:
            raise ValueError("retention must be in (0, 1]")
        if self.merge_layer < 0:
            raise ValueError("merge_layer must be non-negative")
        _validate_odd_kernel(self.salience_kernel_size)

    @property
    def merge_ratio(self) -> float:
        return 1.0 - self.retention
