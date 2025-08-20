# -*- coding: utf-8 -*-
"""General utilities: cosine similarity, safe JSON parse, slugify, etc."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2 归一化（防零安全）。

    Args:
        x: 向量矩阵 (n,d).

    Returns:
        归一化后的矩阵 (n,d).
    """
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / denom).astype(np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算两组向量的余弦相似度矩阵。

    Args:
        a: (n,d)
        b: (m,d)

    Returns:
        (n,m) 相似度矩阵.
    """
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return np.asarray(a_n @ b_n.T, dtype=np.float32)


def pairwise_cosine(a: np.ndarray) -> np.ndarray:
    """计算一组向量的两两余弦相似度矩阵。

    Args:
        a: (n,d)

    Returns:
        (n,n) 对称矩阵.
    """
    return cosine_similarity_matrix(a, a)


def safe_json_loads(maybe_json: str) -> Any | None:
    """更稳健地解析 LLM 返回文本为 JSON。

    - 自动剥离 ```json 代码围栏
    - 失败返回 None

    Args:
        maybe_json: 原始文本。

    Returns:
        解析出的对象或 None。
    """
    try:
        cleaned = maybe_json.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned[cleaned.find("\n") + 1 :]
        return json.loads(cleaned)
    except Exception:
        return None


def slugify(text: str) -> str:
    """将任意字符串转换为文件名友好的 slug。

    Args:
        text: 原字符串。

    Returns:
        slug 字符串。
    """
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    s = "".join(keep)
    while "--" in s:
        s = s.replace("--", "-")
    s = s.strip("-")
    return s or "topic"


def is_monotonic_non_decreasing(xs: list[float], eps: float = 1e-6) -> bool:
    """判断序列是否单调不降（容忍微小误差）。

    Args:
        xs: 值序列。
        eps: 容差。

    Returns:
        True/False.
    """
    return all(xs[i] <= xs[i + 1] + eps for i in range(len(xs) - 1))


def is_finite_and_positive(x: float) -> bool:
    """判断数值是否为有限正数。"""
    return math.isfinite(x) and x > 0.0
