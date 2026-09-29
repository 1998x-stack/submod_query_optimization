# -*- coding: utf-8 -*-
"""Central configuration dataclasses and validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EmbeddingConfig:
    """Embedding model configuration."""

    model_path: str = "model"
    batch_size: int = 32
    device: str | None = None

    def __post_init__(self) -> None:
        if not self.model_path.strip():
            raise ValueError("model_path must not be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(slots=True)
class LLMConfig:
    """OpenAI client and generation configuration."""

    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 800
    request_timeout_sec: float = 60.0
    max_retries: int = 2
    system_prompt: str = (
        "你是一位严谨的研究助理，只以 JSON 数组形式返回 N 条短查询字符串，"
        "每条长度不超过 16 个汉字或 10 个英文词，不要编号、不要描述。"
    )

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.request_timeout_sec <= 0:
            raise ValueError("request_timeout_sec must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


@dataclass(slots=True)
class ExperimentConfig:
    """Experiment-level parameters."""

    topics: list[str]
    num_candidates: int = 20
    k: int = 6
    alpha: float = 0.5
    lambda_diversity: float = 0.5
    mmr_lambda: float = 0.6
    random_seed: int = 42
    chunk_size: int = 800
    chunk_overlap: int = 120
    m3e_path: str = "model"
    data_dir: str = "data"
    output_dir: str = "output"
    llm_model: str = "gpt-4o"
    candidate_cache_dir: str = ".cache/candidates"
    use_candidate_cache: bool = True

    def __post_init__(self) -> None:
        self.topics = [topic.strip() for topic in self.topics if topic.strip()]
        if not self.topics:
            raise ValueError("topics must contain at least one non-empty topic")
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= self.lambda_diversity <= 1.0:
            raise ValueError("lambda_diversity must be in [0, 1]")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        if not self.m3e_path.strip():
            raise ValueError("m3e_path must not be empty")
        if not self.output_dir.strip():
            raise ValueError("output_dir must not be empty")
        if not self.llm_model.strip():
            raise ValueError("llm_model must not be empty")
        if self.use_candidate_cache and not self.candidate_cache_dir.strip():
            raise ValueError("candidate_cache_dir must not be empty when cache is enabled")
