from __future__ import annotations

import numpy as np
import pytest

from src.objectives import FacilityLocationObjective, GraphCutObjective


SIM = np.array(
    [
        [1.0, 0.8, 0.1, 0.2],
        [0.8, 1.0, 0.3, 0.2],
        [0.1, 0.3, 1.0, 0.7],
        [0.2, 0.2, 0.7, 1.0],
    ],
    dtype=np.float32,
)
REL = np.array([0.95, 0.8, 0.55, 0.4], dtype=np.float32)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FacilityLocationObjective(SIM, REL, alpha=0.4),
        lambda: GraphCutObjective(SIM, REL, alpha=0.4, lambda_div=0.3),
    ],
)
def test_incremental_gain_matches_exact_recomputation(factory) -> None:
    obj = factory()
    selected: list[int] = []

    for idx in [0, 2, 1]:
        before = obj.total_value(selected)
        incremental = obj.marginal_gain(idx)
        exact = obj.total_value([*selected, idx]) - before
        assert incremental == pytest.approx(exact, abs=1e-6)

        obj.add_to_set(idx)
        selected.append(idx)
        assert obj.total_value(selected) == pytest.approx(before + incremental, abs=1e-6)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FacilityLocationObjective(SIM, REL, alpha=0.4),
        lambda: GraphCutObjective(SIM, REL, alpha=0.4, lambda_div=0.3),
    ],
)
def test_marginal_gains_obey_diminishing_returns(factory) -> None:
    obj = factory()
    initial = obj.marginal_gain(1)
    obj.add_to_set(0)
    after_growth = obj.marginal_gain(1)
    assert after_growth <= initial + 1e-6


def test_facility_location_empty_set_is_zero() -> None:
    obj = FacilityLocationObjective(SIM, REL, alpha=0.5)
    assert obj.total_value([]) == 0.0


def test_invalid_objective_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        FacilityLocationObjective(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError):
        GraphCutObjective(np.eye(2), np.ones(3))
    with pytest.raises(ValueError):
        FacilityLocationObjective(np.eye(2), np.ones(2), alpha=1.1)
