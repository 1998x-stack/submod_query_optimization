# -*- coding: utf-8 -*-
"""Greedy selectors: standard greedy and lazy greedy (heap-based)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from loguru import logger

from src.objectives import BaseSubmodularObjective
from src.utils import is_finite_and_positive


@dataclass
class GreedyResult:
    """贪心选择结果。"""

    selected_indices: list[int]
    objective_values: list[float]
    per_step_gains: list[float]
    runtime_sec: float
    algo_name: str


class StandardGreedySelector:
    """标准贪心：每轮重算所有候选的边际增益。"""

    def __init__(self, objective: BaseSubmodularObjective) -> None:
        self.objective = objective

    def select(self, k: int) -> GreedyResult:
        start = time.time()
        self.objective.reset_state()
        n = getattr(self.objective, "n")
        selected: list[int] = []
        gains_all: list[float] = []
        values: list[float] = []

        for _ in range(k):
            gains = np.full((n,), -np.inf, dtype=np.float32)
            for idx in range(n):
                gains[idx] = self.objective.marginal_gain(idx)
            best_idx = int(np.argmax(gains))
            best_gain = float(gains[best_idx])
            if not is_finite_and_positive(best_gain):
                logger.debug("⏹️ 标准贪心提前停止（剩余无正增益）")
                break
            self.objective.add_to_set(best_idx)
            selected.append(best_idx)
            gains_all.append(best_gain)
            values.append(self.objective.total_value(selected))

        runtime = time.time() - start
        logger.info("🧮 StandardGreedy done in {:.3f}s, selected={}", runtime, len(selected))
        return GreedyResult(selected, values, gains_all, runtime, "standard_greedy")


class LazyGreedySelector:
    """懒贪心：最大堆 + 按需“懒重算”避免无谓计算。"""

    def __init__(self, objective: BaseSubmodularObjective) -> None:
        self.objective = objective

    def select(self, k: int) -> GreedyResult:
        import heapq

        start = time.time()
        self.objective.reset_state()
        n = getattr(self.objective, "n")
        selected: list[int] = []
        gains_all: list[float] = []
        values: list[float] = []

        # 初始化堆（最大堆 -> 用负数）
        gains0 = self.objective.initial_gains()
        heap: list[Tuple[float, int, int]] = [(-float(g), -1, i) for i, g in enumerate(gains0)]
        heapq.heapify(heap)

        step = 0
        while step < k and heap:
            neg_gain, last_step, idx = heapq.heappop(heap)
            current_gain = -neg_gain
            if last_step != step:
                true_gain = self.objective.marginal_gain(idx)
                heapq.heappush(heap, (-true_gain, step, idx))
                continue

            if not is_finite_and_positive(current_gain):
                logger.debug("⏹️ 懒贪心提前停止（剩余无正增益）")
                break

            self.objective.add_to_set(idx)
            selected.append(idx)
            gains_all.append(float(current_gain))
            values.append(self.objective.total_value(selected))
            step += 1

        runtime = time.time() - start
        logger.info("⚡ LazyGreedy done in {:.3f}s, selected={}", runtime, len(selected))
        return GreedyResult(selected, values, gains_all, runtime, "lazy_greedy")
