from __future__ import annotations

import pytest

from src.config import ExperimentConfig


def test_experiment_config_normalizes_topics() -> None:
    cfg = ExperimentConfig(topics=["  alpha  ", "", "beta"])
    assert cfg.topics == ["alpha", "beta"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"topics": []}, "topics"),
        ({"topics": ["x"], "num_candidates": 0}, "num_candidates"),
        ({"topics": ["x"], "k": 0}, "k"),
        ({"topics": ["x"], "alpha": -0.1}, "alpha"),
        ({"topics": ["x"], "mmr_lambda": 1.1}, "mmr_lambda"),
        ({"topics": ["x"], "random_seed": -1}, "random_seed"),
        ({"topics": ["x"], "chunk_size": 100, "chunk_overlap": 100}, "chunk_overlap"),
        (
            {
                "topics": ["x"],
                "candidate_cache_dir": "",
                "use_candidate_cache": True,
            },
            "candidate_cache_dir",
        ),
    ],
)
def test_invalid_experiment_config_fails_fast(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(**kwargs)


def test_empty_cache_dir_is_allowed_when_cache_disabled() -> None:
    cfg = ExperimentConfig(
        topics=["x"],
        candidate_cache_dir="",
        use_candidate_cache=False,
    )
    assert cfg.use_candidate_cache is False
