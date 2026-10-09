# -*- coding: utf-8 -*-
"""Dense retrieval and deterministic multi-query score fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils import cosine_similarity_matrix


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    doc_ids: list[str]
    scores: list[float]


class DenseMultiQueryRetriever:
    """Rank a fixed document collection using max-fused cosine similarity."""

    def __init__(self, document_ids: list[str], document_embeddings: np.ndarray) -> None:
        embeddings = np.asarray(document_embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("document_embeddings must be a 2-D matrix")
        if len(document_ids) != embeddings.shape[0]:
            raise ValueError("document_ids length must match document_embeddings rows")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_ids must be unique")
        if not document_ids:
            raise ValueError("document collection must not be empty")
        if not np.isfinite(embeddings).all():
            raise ValueError("document_embeddings must contain only finite values")

        self.document_ids = list(document_ids)
        self.document_embeddings = embeddings

    def rank(self, query_embeddings: np.ndarray, top_k: int) -> RetrievalResult:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[0] == 0:
            raise ValueError("query_embeddings must be a non-empty 2-D matrix")
        if queries.shape[1] != self.document_embeddings.shape[1]:
            raise ValueError("query/document embedding dimensions must match")

        # (num_docs, num_queries), then max fusion across original + expansions.
        similarities = cosine_similarity_matrix(
            self.document_embeddings,
            queries,
        )
        fused_scores = similarities.max(axis=1)

        limit = min(top_k, len(self.document_ids))
        # Stable sorting makes exact-score ties deterministic by document order.
        order = np.argsort(-fused_scores, kind="stable")[:limit]
        return RetrievalResult(
            doc_ids=[self.document_ids[int(idx)] for idx in order],
            scores=[float(fused_scores[int(idx)]) for idx in order],
        )
