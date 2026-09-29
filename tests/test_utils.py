from __future__ import annotations

import numpy as np
import pytest

from src.utils import cosine_similarity_matrix, l2_normalize, safe_json_loads


def test_safe_json_loads_handles_fenced_and_wrapped_arrays() -> None:
    fenced = chr(96) * 3 + 'json\n["a", "b"]\n' + chr(96) * 3
    assert safe_json_loads(fenced) == ["a", "b"]
    assert safe_json_loads('prefix ["x", "y"] suffix') == ["x", "y"]


def test_l2_normalize_keeps_zero_vectors_finite() -> None:
    arr = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    normalized = l2_normalize(arr)
    assert np.isfinite(normalized).all()
    assert normalized[0].tolist() == [0.0, 0.0]
    assert normalized[1].tolist() == pytest.approx([0.6, 0.8])


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        cosine_similarity_matrix(
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 4), dtype=np.float32),
        )
