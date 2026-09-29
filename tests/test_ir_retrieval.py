from __future__ import annotations

import numpy as np

from src.ir_retrieval import DenseMultiQueryRetriever


def test_max_fusion_recovers_documents_from_multiple_expansions() -> None:
    retriever = DenseMultiQueryRetriever(
        ["d1", "d2", "d3"],
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    result = retriever.rank(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        top_k=3,
    )
    assert result.doc_ids[:2] == ["d1", "d2"]
    assert result.scores[:2] == [1.0, 1.0]


def test_ties_are_stable_by_document_order() -> None:
    retriever = DenseMultiQueryRetriever(
        ["first", "second"],
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    result = retriever.rank(np.array([[1.0, 0.0]], dtype=np.float32), top_k=2)
    assert result.doc_ids == ["first", "second"]
