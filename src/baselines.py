# -*- coding: utf-8 -*-
"""Deterministic baseline selectors for benchmark comparisons."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Selection output shared by non-submodular benchmark baselines."""

    selected_indices: list[int]
    runtime_sec: float
    algo_name: str


def _validate_inputs(sim: np.ndarray, rel: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, int]:
    sim_arr = np.asarray(sim, dtype=np.float32)
    rel_arr = np.asarray(rel, dtype=np.float32).reshape(-1)

    if sim_arr.ndim != 2 or sim_arr.shape[0] != sim_arr.shape[1]:
        raise ValueError("sim must be a square (n, n) matrix")
    if rel_arr.shape[0] != sim_arr.shape[0]:
        raise ValueError("rel length must equal sim.shape[0]")
    if not np.isfinite(sim_arr).all() or not np.isfinite(rel_arr).all():
        raise ValueError("sim and rel must contain only finite values")
    if k < 0:
        raise ValueError("k must be non-negative")

    return sim_arr, rel_arr, min(int(k), int(sim_arr.shape[0]))


def top_relevance_select(sim: np.ndarray, rel: np.ndarray, k: int) -> BaselineResult:
    """Select the k candidates with highest topic relevance."""
    start = time.perf_counter()
    _, rel_arr, budget = _validate_inputs(sim, rel, k)
    order = np.argsort(-rel_arr, kind="stable")
    selected = [int(idx) for idx in order[:budget]]
    return BaselineResult(selected, time.perf_counter() - start, "top_relevance")


def mmr_select(
    sim: np.ndarray,
    rel: np.ndarray,
    k: int,
    lambda_relevance: float = 0.6,
) -> BaselineResult:
    """Select candidates using classic maximal marginal relevance.

    score(i) = lambda * relevance(i)
             - (1 - lambda) * max_similarity(i, selected)
    """
    start = time.perf_counter()
    sim_arr, rel_arr, budget = _validate_inputs(sim, rel, k)
    if not 0.0 <= lambda_relevance <= 1.0:
        raise ValueError("lambda_relevance must be in [0, 1]")
    if budget == 0:
        return BaselineResult([], time.perf_counter() - start, "mmr")

    sim_arr = np.clip(sim_arr, -1.0, 1.0)
    rel_arr = np.clip(rel_arr, -1.0, 1.0)
    selected: list[int] = []
    available = np.ones((sim_arr.shape[0],), dtype=bool)

    first = int(np.argmax(rel_arr))
    selected.append(first)
    available[first] = False

    while len(selected) < budget:
        candidates = np.flatnonzero(available)
        if candidates.size == 0:
            break

        max_similarity = sim_arr[np.ix_(candidates, selected)].max(axis=1)
        scores = (
            lambda_relevance * rel_arr[candidates]
            - (1.0 - lambda_relevance) * max_similarity
        )
        best_local = int(np.argmax(scores))
        best_idx = int(candidates[best_local])
        selected.append(best_idx)
        available[best_idx] = False

    return BaselineResult(selected, time.perf_counter() - start, "mmr")


def random_select(
    sim: np.ndarray,
    rel: np.ndarray,
    k: int,
    seed: int,
) -> BaselineResult:
    """Select without replacement using a deterministic numpy generator."""
    start = time.perf_counter()
    sim_arr, _, budget = _validate_inputs(sim, rel, k)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if budget == 0:
        return BaselineResult([], time.perf_counter() - start, "random")

    rng = np.random.default_rng(seed)
    selected = [int(idx) for idx in rng.choice(sim_arr.shape[0], size=budget, replace=False)]
    return BaselineResult(selected, time.perf_counter() - start, "random")


def derive_topic_seed(base_seed: int, topic: str) -> int:
    """Derive a stable per-topic 32-bit seed independent of Python hash randomization."""
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    digest = hashlib.sha256(f"{base_seed}:{topic}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)
