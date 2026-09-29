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
        ({"topics": ["x"], "ir_cutoffs": (5, 0)}, "ir_cutoffs"),
        ({"topics": ["x"], "ir_cutoffs": (10, 10)}, "ir_cutoffs"),
        ({"topics": ["x"], "ir_k_grid": (2, 0)}, "ir_k_grid"),
        ({"topics": ["x"], "ir_k_grid": (4, 4)}, "ir_k_grid"),
        ({"topics": ["x"], "ir_alpha_grid": (1.1,)}, "ir_alpha_grid"),
        ({"topics": ["x"], "ir_lambda_grid": ()}, "ir_lambda_grid"),
        ({"topics": ["x"], "ir_mmr_grid": (-0.1,)}, "ir_mmr_grid"),
        ({"topics": ["x"], "ir_random_repeats": 0}, "ir_random_repeats"),
        ({"topics": ["x"], "ir_bootstrap_iterations": 0}, "ir_bootstrap_iterations"),
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


def test_ir_grid_configuration_is_preserved() -> None:
    cfg = ExperimentConfig(
        topics=["x"],
        ir_dataset_dir="bench",
        ir_cutoffs=(3, 7),
        ir_k_grid=(2, 4),
        ir_alpha_grid=(0.2, 0.8),
        ir_lambda_grid=(0.1, 0.5),
        ir_mmr_grid=(0.4,),
        ir_random_repeats=3,
    )
    assert cfg.ir_dataset_dir == "bench"
    assert cfg.ir_cutoffs == (3, 7)
    assert cfg.ir_k_grid == (2, 4)
    assert cfg.ir_alpha_grid == (0.2, 0.8)


def test_ir_k_grid_defaults_to_main_selection_budget() -> None:
    cfg = ExperimentConfig(topics=["x"], k=7)
    assert cfg.ir_k_grid == (7,)
