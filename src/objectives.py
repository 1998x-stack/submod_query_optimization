# -*- coding: utf-8 -*-
"""Submodular objectives: Facility Location & Graph Cut.

提供两种子模目标：
1) 设施选址（隐式多样性）：最大覆盖
2) 图割（显式多样性）：相关性 + 跨割多样性

均实现统一接口，以供贪心选择器调用。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from src.utils import cosine_similarity_matrix


class BaseSubmodularObjective(ABC):
    """子模目标抽象基类。"""

    n: int

    @abstractmethod
    def reset_state(self) -> None:
        """重置内部状态。"""

    @abstractmethod
    def initial_gains(self) -> np.ndarray:
        """返回初始边际增益（用于懒贪心的堆初始化）。"""

    @abstractmethod
    def marginal_gain(self, idx: int) -> float:
        """计算将 idx 加入当前集合的边际增益。"""

    @abstractmethod
    def add_to_set(self, idx: int) -> None:
        """将 idx 加入集合并更新内部状态。"""

    @abstractmethod
    def total_value(self, selected: Sequence[int]) -> float:
        """重算 f(S) 的精确值（用于 correctness recheck）。"""


class FacilityLocationObjective(BaseSubmodularObjective):
    """设施选址法目标（隐式多样性）。

    定义：
        f(S) = sum_j max( alpha*rel[j], max_{i in S} sim[i,j] )

    维护：
        current_cov[j] = 当前对 j 的最优覆盖（已包含 alpha*rel[j]）
    """

    def __init__(self, sim: np.ndarray, rel: np.ndarray, alpha: float = 0.5) -> None:
        assert sim.shape[0] == sim.shape[1], "sim 必须为 (n,n)"
        assert rel.shape[0] == sim.shape[0], "rel 长度应等于 n"
        self.sim = sim.astype(np.float32)
        self.rel = rel.astype(np.float32)
        self.alpha = float(alpha)
        self.n = sim.shape[0]
        self.current_cov = np.zeros((self.n,), dtype=np.float32)
        self.selected_mask = np.zeros((self.n,), dtype=bool)
        self.reset_state()

    def reset_state(self) -> None:
        self.current_cov = (self.alpha * self.rel).copy()
        self.selected_mask[:] = False

    def initial_gains(self) -> np.ndarray:
        base = self.current_cov  # alpha*rel
        gains = np.maximum(0.0, self.sim - base[None, :]).sum(axis=1)
        gains[self.selected_mask] = -np.inf
        return gains.astype(np.float32)

    def marginal_gain(self, idx: int) -> float:
        if self.selected_mask[idx]:
            return -math.inf
        diff = self.sim[idx, :] - self.current_cov
        return float(np.maximum(0.0, diff).sum())

    def add_to_set(self, idx: int) -> None:
        if self.selected_mask[idx]:
            return
        self.current_cov = np.maximum(self.current_cov, self.sim[idx, :])
        self.selected_mask[idx] = True

    def total_value(self, selected: Sequence[int]) -> float:
        base = self.alpha * self.rel
        if not selected:
            return float(base.sum())
        cover = np.maximum.reduce([self.sim[i, :] for i in selected])
        cover = np.maximum(base, cover)
        return float(cover.sum())


class GraphCutObjective(BaseSubmodularObjective):
    """图割法目标（显式多样性）。

    我们用 w(i,j) = 1 - sim(i,j) ≥ 0 作为“不相似度”，
    定义：
        Cut(S) = sum_{i in S, j in V\\S} w(i,j)
        Rel(S) = sum_{i in S} rel[i]  # 其中 rel[i] = sim(q0, i)
        f(S)   = alpha * Rel(S) + lambda * Cut(S)

    增量式：
        ΔCut = total_row[i] - 2 * sum_to_S[i]
        ΔRel = rel[i]
        Δf   = alpha*ΔRel + lambda*ΔCut
    """

    def __init__(self, sim: np.ndarray, rel: np.ndarray, alpha: float = 0.5, lambda_div: float = 0.5) -> None:
        assert sim.shape[0] == sim.shape[1], "sim 必须为 (n,n)"
        assert rel.shape[0] == sim.shape[0], "rel 长度应等于 n"
        self.sim = sim.astype(np.float32)
        self.rel = rel.astype(np.float32)
        self.alpha = float(alpha)
        self.lambda_div = float(lambda_div)
        self.n = sim.shape[0]

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
        gains[self.selected_mask] = -np.inf
        return gains.astype(np.float32)

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
        S = np.zeros((self.n,), dtype=bool)
        S[selected] = True
        cut_val = float(self.w[np.ix_(S, ~S)].sum())
        rel_val = float(self.rel[S].sum())
        return self.alpha * rel_val + self.lambda_div * cut_val
