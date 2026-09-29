# -*- coding: utf-8 -*-
"""Submodular objectives used by the query-selection experiments."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


def _validate_inputs(sim: np.ndarray, rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize objective inputs."""
    sim_arr = np.asarray(sim, dtype=np.float32)
    rel_arr = np.asarray(rel, dtype=np.float32).reshape(-1)

    if sim_arr.ndim != 2 or sim_arr.shape[0] != sim_arr.shape[1]:
        raise ValueError("sim must be a square (n, n) matrix")
    if rel_arr.shape[0] != sim_arr.shape[0]:
        raise ValueError("rel length must equal sim.shape[0]")
    if not np.isfinite(sim_arr).all() or not np.isfinite(rel_arr).all():
        raise ValueError("sim and rel must contain only finite values")

    return sim_arr, rel_arr


class BaseSubmodularObjective(ABC):
    """Common interface consumed by greedy selectors."""

    n: int

    @abstractmethod
    def reset_state(self) -> None:
        """Reset incremental state to the empty set."""

    @abstractmethod
    def initial_gains(self) -> np.ndarray:
        """Return marginal gains from the empty set."""

    @abstractmethod
    def marginal_gain(self, idx: int) -> float:
        """Return the gain of adding idx to the current set."""

    @abstractmethod
    def add_to_set(self, idx: int) -> None:
        """Add idx to the current set and update incremental state."""

    @abstractmethod
    def total_value(self, selected: Sequence[int]) -> float:
        """Recompute f(S) exactly for correctness checks."""


class FacilityLocationObjective(BaseSubmodularObjective):
    """Relevance-aware facility-location objective.

    The previous implementation used alpha * relevance as a selection-independent
    coverage floor. That made relevant candidates look pre-covered before they
    were selected. Here relevance is a modular reward and coverage is the
    facility-location submodular term:

        f(S) = alpha * sum_{i in S} rel[i]
             + (1 - alpha) * sum_j max_{i in S} sim[i, j]

    Cosine relevance and similarity are clipped to [0, 1] so the objective is
    monotone and lazy-greedy upper bounds remain valid.
    """

    def __init__(self, sim: np.ndarray, rel: np.ndarray, alpha: float = 0.5) -> None:
        sim_arr, rel_arr = _validate_inputs(sim, rel)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        self.sim = np.clip(sim_arr, 0.0, 1.0)
        self.rel = np.clip(rel_arr, 0.0, 1.0)
        self.alpha = float(alpha)
        self.coverage_weight = 1.0 - self.alpha
        self.n = sim_arr.shape[0]

        self.current_cov = np.zeros((self.n,), dtype=np.float32)
        self.selected_mask = np.zeros((self.n,), dtype=bool)
        self.reset_state()

    def reset_state(self) -> None:
        self.current_cov.fill(0.0)
        self.selected_mask[:] = False

    def initial_gains(self) -> np.ndarray:
        gains = (
            self.alpha * self.rel
            + self.coverage_weight * self.sim.sum(axis=1)
        ).astype(np.float32)
        gains[self.selected_mask] = -np.inf
        return gains

    def marginal_gain(self, idx: int) -> float:
        if self.selected_mask[idx]:
            return -math.inf
        coverage_gain = np.maximum(0.0, self.sim[idx] - self.current_cov).sum()
        return float(
            self.alpha * self.rel[idx]
            + self.coverage_weight * coverage_gain
        )

    def add_to_set(self, idx: int) -> None:
        if self.selected_mask[idx]:
            return
        self.current_cov = np.maximum(self.current_cov, self.sim[idx])
        self.selected_mask[idx] = True

    def total_value(self, selected: Sequence[int]) -> float:
        if not selected:
            return 0.0

        idx = np.asarray(selected, dtype=int)
        if np.any(idx < 0) or np.any(idx >= self.n):
            raise IndexError("selected index out of range")

        cover = self.sim[idx].max(axis=0)
        relevance = self.rel[idx].sum()
        return float(
            self.alpha * relevance
            + self.coverage_weight * cover.sum()
        )


class GraphCutObjective(BaseSubmodularObjective):
    """Graph-cut objective with explicit diversity.

    w(i, j) = 1 - sim(i, j)
    f(S) = alpha * sum_{i in S} rel[i]
         + lambda_div * sum_{i in S, j not in S} w(i, j)

    With non-negative edge weights this objective is submodular, but it is not
    necessarily monotone. Selectors therefore stop when no positive gain
    remains.
    """

    def __init__(
        self,
        sim: np.ndarray,
        rel: np.ndarray,
        alpha: float = 0.5,
        lambda_div: float = 0.5,
    ) -> None:
        sim_arr, rel_arr = _validate_inputs(sim, rel)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if lambda_div < 0.0:
            raise ValueError("lambda_div must be non-negative")

        self.sim = np.clip(sim_arr, -1.0, 1.0)
        self.rel = np.clip(rel_arr, 0.0, 1.0)
        self.alpha = float(alpha)
        self.lambda_div = float(lambda_div)
        self.n = sim_arr.shape[0]

        self.w = (1.0 - self.sim).astype(np.float32)
        np.fill_diagonal(self.w, 0.0)
        self.total_row = self.w.sum(axis=1)
        self.sum_to_S = np.zeros((self.n,), dtype=np.float32)
        self.selected_mask = np.zeros((self.n,), dtype=bool)
        self.reset_state()

    def reset_state(self) -> None:
        self.sum_to_S.fill(0.0)
        self.selected_mask[:] = False

    def initial_gains(self) -> np.ndarray:
        gains = self.alpha * self.rel + self.lambda_div * self.total_row
        gains = gains.astype(np.float32)
        gains[self.selected_mask] = -np.inf
        return gains

    def marginal_gain(self, idx: int) -> float:
        if self.selected_mask[idx]:
            return -math.inf
        delta_cut = float(self.total_row[idx] - 2.0 * self.sum_to_S[idx])
        delta_rel = float(self.rel[idx])
        return self.alpha * delta_rel + self.lambda_div * delta_cut

    def add_to_set(self, idx: int) -> None:
        if self.selected_mask[idx]:
            return
        self.sum_to_S += self.w[:, idx]
        self.selected_mask[idx] = True

    def total_value(self, selected: Sequence[int]) -> float:
        if not selected:
            return 0.0

        idx = np.asarray(selected, dtype=int)
        if np.any(idx < 0) or np.any(idx >= self.n):
            raise IndexError("selected index out of range")

        mask = np.zeros((self.n,), dtype=bool)
        mask[idx] = True
        cut_val = float(self.w[np.ix_(mask, ~mask)].sum())
        rel_val = float(self.rel[mask].sum())
        return self.alpha * rel_val + self.lambda_div * cut_val
