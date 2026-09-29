# -*- coding: utf-8 -*-
"""Small deterministic statistical helpers for benchmark summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MeanCI:
    mean: float
    low: float
    high: float


def bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    iterations: int = 2000,
    seed: int = 42,
) -> MeanCI:
    """Return a percentile bootstrap confidence interval for the arithmetic mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.isfinite(arr).all():
        raise ValueError("values must contain only finite numbers")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    mean = float(arr.mean())
    if arr.size == 1:
        return MeanCI(mean, mean, mean)

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        arr.size,
        size=(iterations, arr.size),
    )
    boot_means = arr[sample_indices].mean(axis=1)
    alpha = 1.0 - confidence
    low = float(np.quantile(boot_means, alpha / 2.0))
    high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return MeanCI(mean, low, high)
