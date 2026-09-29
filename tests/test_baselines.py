from __future__ import annotations

import numpy as np
import pytest

from src.baselines import (
    derive_topic_seed,
    mmr_select,
    random_select,
    top_relevance_select,
)


SIM = np.array(
    [
        [1.0, 0.95, 0.10, 0.10],
        [0.95, 1.0, 0.15, 0.10],
        [0.10, 0.15, 1.0, 0.20],
        [0.10, 0.10, 0.20, 1.0],
    ],
    dtype=np.float32,
)
REL = np.array([0.95, 0.90, 0.70, 0.60], dtype=np.float32)


def test_top_relevance_is_stable_and_descending() -> None:
    result = top_relevance_select(SIM, REL, 3)
    assert result.selected_indices == [0, 1, 2]


def test_mmr_trades_redundancy_for_diversity() -> None:
    result = mmr_select(SIM, REL, 2, lambda_relevance=0.5)
    assert result.selected_indices[0] == 0
    assert result.selected_indices[1] in {2, 3}
    assert len(set(result.selected_indices)) == 2


def test_random_baseline_is_reproducible() -> None:
    first = random_select(SIM, REL, 3, seed=123)
    second = random_select(SIM, REL, 3, seed=123)
    assert first.selected_indices == second.selected_indices


def test_topic_seed_is_stable_and_topic_specific() -> None:
    assert derive_topic_seed(42, "alpha") == derive_topic_seed(42, "alpha")
    assert derive_topic_seed(42, "alpha") != derive_topic_seed(42, "beta")


def test_invalid_mmr_lambda_is_rejected() -> None:
    with pytest.raises(ValueError):
        mmr_select(SIM, REL, 2, lambda_relevance=1.1)
