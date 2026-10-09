from __future__ import annotations

import pytest

from src.ir_metrics import evaluate_ranking


def test_perfect_ranking_has_unit_metrics() -> None:
    metrics = evaluate_ranking(
        ["d1", "d2", "d3"],
        {"d1": 3, "d2": 1, "d3": 0},
        cutoff=2,
    )
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.hit_rate == 1.0
    assert metrics.reciprocal_rank == 1.0
    assert metrics.ndcg == pytest.approx(1.0)


def test_partial_ranking_uses_binary_relevance_for_recall_and_graded_for_ndcg() -> None:
    metrics = evaluate_ranking(
        ["d3", "d2", "d1"],
        {"d1": 3, "d2": 1},
        cutoff=2,
    )
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.hit_rate == 1.0
    assert metrics.reciprocal_rank == pytest.approx(0.5)
    assert 0.0 < metrics.ndcg < 1.0


def test_no_hit_is_zero() -> None:
    metrics = evaluate_ranking(["x", "y"], {"d1": 1}, cutoff=2)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.hit_rate == 0.0
    assert metrics.reciprocal_rank == 0.0
    assert metrics.ndcg == 0.0


def test_duplicate_ranked_docs_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        evaluate_ranking(["d1", "d1"], {"d1": 1}, cutoff=2)
