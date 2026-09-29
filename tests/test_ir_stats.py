from __future__ import annotations

from src.ir_stats import bootstrap_mean_ci


def test_bootstrap_is_deterministic_for_fixed_seed() -> None:
    first = bootstrap_mean_ci([0.0, 0.5, 1.0], iterations=500, seed=7)
    second = bootstrap_mean_ci([0.0, 0.5, 1.0], iterations=500, seed=7)
    assert first == second
    assert first.low <= first.mean <= first.high


def test_single_value_has_degenerate_interval() -> None:
    result = bootstrap_mean_ci([0.25], iterations=10, seed=1)
    assert result.mean == result.low == result.high == 0.25
