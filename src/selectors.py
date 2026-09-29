# -*- coding: utf-8 -*-
"""Greedy selectors: standard greedy and CELF-style lazy greedy."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.objectives import BaseSubmodularObjective
from src.utils import is_finite_and_positive


@dataclass
class GreedyResult:
    """Greedy-selection result and diagnostics."""

    selected_indices: list[int]
    objective_values: list[float]
    per_step_gains: list[float]
    runtime_sec: float
    algo_name: str


def _validate_budget(k: int, n: int) -> int:
    if k < 0:
        raise ValueError("k must be non-negative")
    return min(int(k), int(n))


class StandardGreedySelector:
    """Standard greedy: recompute every remaining marginal gain each round."""

    def __init__(self, objective: BaseSubmodularObjective) -> None:
        self.objective = objective

    def select(self, k: int) -> GreedyResult:
        start = time.perf_counter()
        self.objective.reset_state()
        n = int(self.objective.n)
        budget = _validate_budget(k, n)

        selected: list[int] = []
        gains_all: list[float] = []
        values: list[float] = []

        for _ in range(budget):
            gains = np.full((n,), -np.inf, dtype=np.float64)
            for idx in range(n):
                gains[idx] = self.objective.marginal_gain(idx)

            best_idx = int(np.argmax(gains))
            best_gain = float(gains[best_idx])
            if not is_finite_and_positive(best_gain):
                logger.debug("Standard greedy stopped: no positive finite gain remains")
                break

            self.objective.add_to_set(best_idx)
            selected.append(best_idx)
            gains_all.append(best_gain)
            values.append(self.objective.total_value(selected))

        runtime = time.perf_counter() - start
        logger.info("StandardGreedy done in {:.3f}s, selected={}", runtime, len(selected))
        return GreedyResult(selected, values, gains_all, runtime, "standard_greedy")


class LazyGreedySelector:
    """CELF-style lazy greedy for submodular objectives.

    Cached marginal gains are upper bounds after the selected set grows because
    submodularity implies diminishing returns. A candidate is accepted only
    after its gain has been recomputed for the current step and it remains at
    the top of the heap.
    """

    def __init__(self, objective: BaseSubmodularObjective) -> None:
        self.objective = objective

    def select(self, k: int) -> GreedyResult:
        start = time.perf_counter()
        self.objective.reset_state()
        n = int(self.objective.n)
        budget = _validate_budget(k, n)

        selected: list[int] = []
        gains_all: list[float] = []
        values: list[float] = []

        if budget == 0:
            return GreedyResult(selected, values, gains_all, 0.0, "lazy_greedy")

        gains0 = self.objective.initial_gains()
        if gains0.shape != (n,):
            raise ValueError("initial_gains must return shape (n,)")

        heap: list[tuple[float, int, int]] = [
            (-float(gain), -1, idx) for idx, gain in enumerate(gains0)
        ]
        heapq.heapify(heap)

        step = 0
        while step < budget and heap:
            neg_gain, evaluated_at_step, idx = heapq.heappop(heap)

            if evaluated_at_step != step:
                true_gain = float(self.objective.marginal_gain(idx))
                heapq.heappush(heap, (-true_gain, step, idx))
                continue

            current_gain = -neg_gain
            if not is_finite_and_positive(current_gain):
                logger.debug("Lazy greedy stopped: no positive finite gain remains")
                break

            self.objective.add_to_set(idx)
            selected.append(idx)
            gains_all.append(current_gain)
            values.append(self.objective.total_value(selected))
            step += 1

        runtime = time.perf_counter() - start
        logger.info("LazyGreedy done in {:.3f}s, selected={}", runtime, len(selected))
        return GreedyResult(selected, values, gains_all, runtime, "lazy_greedy")
