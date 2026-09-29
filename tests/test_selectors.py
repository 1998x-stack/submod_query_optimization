from __future__ import annotations

import numpy as np
import pytest

from src.objectives import FacilityLocationObjective, GraphCutObjective
from src.selectors import LazyGreedySelector, StandardGreedySelector


SIM = np.array(
    [
        [1.0, 0.9, 0.2, 0.1, 0.0],
        [0.9, 1.0, 0.3, 0.1, 0.0],
        [0.2, 0.3, 1.0, 0.8, 0.2],
        [0.1, 0.1, 0.8, 1.0, 0.4],
        [0.0, 0.0, 0.2, 0.4, 1.0],
    ],
    dtype=np.float32,
)
REL = np.array([0.95, 0.85, 0.7, 0.55, 0.3], dtype=np.float32)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FacilityLocationObjective(SIM, REL, alpha=0.4),
        lambda: GraphCutObjective(SIM, REL, alpha=0.4, lambda_div=0.2),
    ],
)
def test_lazy_matches_standard_greedy_objective(factory) -> None:
    standard = StandardGreedySelector(factory()).select(3)
    lazy = LazyGreedySelector(factory()).select(3)

    assert lazy.selected_indices == standard.selected_indices
    assert lazy.objective_values == pytest.approx(standard.objective_values, abs=1e-6)
    assert lazy.per_step_gains == pytest.approx(standard.per_step_gains, abs=1e-6)


def test_empty_candidate_set_is_safe() -> None:
    objective = FacilityLocationObjective(
        np.empty((0, 0), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )
    assert StandardGreedySelector(objective).select(5).selected_indices == []
    assert LazyGreedySelector(objective).select(5).selected_indices == []


def test_negative_budget_is_rejected() -> None:
    objective = FacilityLocationObjective(SIM, REL)
    with pytest.raises(ValueError):
        StandardGreedySelector(objective).select(-1)
    with pytest.raises(ValueError):
        LazyGreedySelector(objective).select(-1)
