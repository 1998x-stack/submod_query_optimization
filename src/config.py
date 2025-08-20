# -*- coding: utf-8 -*-
"""Configuration dataclasses for the project.

本模块集中管理项目中的各类配置（Embedding/LLM/实验参数等），
以便在不同模块间统一传递，避免“魔法常量”与重复代码。

遵循：
- PEP 257：Docstring
- PEP 8：命名与风格
- 全量类型注解
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingConfig:
    """Embedding 配置。

    Attributes:
        model_path: 本地 m3e 模型目录（如: "model"）。
        batch_size: 向量化的批大小。
        device: 设备标识，如 "cuda" / "cpu" / None(自动)。
    """

    model_path: str = "model"
    batch_size: int = 32
    device: str | None = None


@dataclass
class LLMConfig:
    """LLM（OpenAI GPT-4o）配置。

    Attributes:
        model_name: 模型名，如 "gpt-4o"。
        temperature: 采样温度（建议 0.3~0.8）。
        max_tokens: 最大生成 token。
        system_prompt: 系统描述，强制只返回 JSON 数组。
    """

    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 800
    system_prompt: str = (
        "你是一位严谨的研究助理，只以 JSON 数组形式返回 N 条短查询字符串，"
        "每条长度不超过 16 个汉字或 10 个英文词，不要编号、不要描述。"
    )


@dataclass
class ExperimentConfig:
    """实验配置（消融维度等）。

    Attributes:
        topics: 要测试的一组主题。
        num_candidates: 每个主题初始候选查询数。
        k: 最终要选择的查询数。
        alpha: 相关性权重（0~1）。
        lambda_diversity: 图割法的多样性权重（0~1）。
        chunk_size: 文本分块大小（字符）。
        chunk_overlap: 分块重叠（字符）。
        m3e_path: 本地 m3e 模型目录。
        data_dir: 语料目录。
        output_dir: 输出目录。
        llm_model: OpenAI 模型名。
    """

    topics: List[str]
    num_candidates: int = 20
    k: int = 6
    alpha: float = 0.5
    lambda_diversity: float = 0.5
    chunk_size: int = 800
    chunk_overlap: int = 120
    m3e_path: str = "model"
    data_dir: str = "data"
    output_dir: str = "output"
    llm_model: str = "gpt-4o"
