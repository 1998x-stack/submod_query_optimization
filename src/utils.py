# -*- coding: utf-8 -*-
"""General numerical and parsing utilities."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def _as_2d_float32(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows without producing NaNs for zero vectors."""
    arr = _as_2d_float32(x, "x")
    if arr.shape[0] == 0:
        return arr.copy()
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return (arr / denom).astype(np.float32, copy=False)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the cosine-similarity matrix between row vectors in a and b."""
    a_arr = _as_2d_float32(a, "a")
    b_arr = _as_2d_float32(b, "b")
    if a_arr.shape[1] != b_arr.shape[1]:
        raise ValueError(
            f"embedding dimensions differ: {a_arr.shape[1]} != {b_arr.shape[1]}"
        )
    if a_arr.shape[0] == 0 or b_arr.shape[0] == 0:
        return np.empty((a_arr.shape[0], b_arr.shape[0]), dtype=np.float32)
    return np.asarray(l2_normalize(a_arr) @ l2_normalize(b_arr).T, dtype=np.float32)


def pairwise_cosine(a: np.ndarray) -> np.ndarray:
    """Return the pairwise cosine-similarity matrix for row vectors in a."""
    arr = _as_2d_float32(a, "a")
    return cosine_similarity_matrix(arr, arr)


def safe_json_loads(maybe_json: str) -> Any | None:
    """Parse common LLM JSON wrappers without silently accepting arbitrary text."""
    cleaned = maybe_json.strip()
    if not cleaned:
        return None

    candidates = [cleaned]

    if cleaned.startswith("```") and cleaned.endswith("```"):
        fenced = cleaned[3:-3].strip()
        if fenced.lower().startswith("json"):
            fenced = fenced[4:].lstrip()
        candidates.append(fenced)

    left_arr, right_arr = cleaned.find("["), cleaned.rfind("]")
    if left_arr >= 0 and right_arr > left_arr:
        candidates.append(cleaned[left_arr : right_arr + 1])

    left_obj, right_obj = cleaned.find("{"), cleaned.rfind("}")
    if left_obj >= 0 and right_obj > left_obj:
        candidates.append(cleaned[left_obj : right_obj + 1])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def slugify(text: str) -> str:
    """Convert arbitrary text to a deterministic file-name-friendly slug."""
    keep = [char.lower() if char.isalnum() else "-" for char in text]
    value = "".join(keep)
    while "--" in value:
        value = value.replace("--", "-")
    return value.strip("-") or "topic"


def is_monotonic_non_decreasing(xs: list[float], eps: float = 1e-6) -> bool:
    """Return whether xs is non-decreasing within a numeric tolerance."""
    return all(xs[i] <= xs[i + 1] + eps for i in range(len(xs) - 1))


def is_finite_and_positive(x: float) -> bool:
    """Return whether x is finite and strictly positive."""
    return math.isfinite(x) and x > 0.0
