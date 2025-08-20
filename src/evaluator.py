# -*- coding: utf-8 -*-
"""Evaluator for query sets (diversity, redundancy, relevance, coverage)."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence

from src.embeddings import LangChainM3EEmbedder
from src.utils import cosine_similarity_matrix, pairwise_cosine


@dataclass
class EvaluationMetrics:
    """评估指标容器。"""

    mean_intra_similarity: float
    median_intra_similarity: float
    duplicate_rate_08: float
    avg_relevance_to_topic: float
    coverage_score_vs_corpus: float


class Evaluator:
    """对查询集合进行多维评估。"""

    def __init__(self, embedder: LangChainM3EEmbedder, corpus_embeddings: np.ndarray | None) -> None:
        self.embedder = embedder
        self.corpus_embeddings = corpus_embeddings

    @staticmethod
    def _dup_rate(sim_mat: np.ndarray, threshold: float = 0.8) -> float:
        n = sim_mat.shape[0]
        if n <= 1:
            return 0.0
        mask = np.triu(np.ones_like(sim_mat, dtype=bool), k=1)
        pairs = sim_mat[mask]
        dup = float((pairs >= threshold).sum())
        total = float(pairs.size)
        return dup / max(total, 1.0)

    def evaluate(
        self,
        queries: Sequence[str],
        query_embeddings: np.ndarray | None,
        topic_embedding: np.ndarray | None,
    ) -> EvaluationMetrics:
        """主评估函数。

        Args:
            queries: 查询列表。
            query_embeddings: 查询向量，如 None 则自动编码。
            topic_embedding: 主题向量，如 None 则 avg_relevance=0。

        Returns:
            EvaluationMetrics
        """
        if len(queries) == 0:
            return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        if query_embeddings is None:
            query_embeddings = self.embedder.encode(list(queries))

        sim_mat = pairwise_cosine(query_embeddings)
        n = sim_mat.shape[0]
        tri = sim_mat[np.triu_indices(n, k=1)]
        mean_intra = float(tri.mean()) if tri.size > 0 else 0.0
        median_intra = float(np.median(tri)) if tri.size > 0 else 0.0
        dup_rate = self._dup_rate(sim_mat, threshold=0.8)

        if topic_embedding is None:
            avg_rel = 0.0
        else:
            rel_vec = cosine_similarity_matrix(query_embeddings, topic_embedding[None, :]).reshape(-1)
            avg_rel = float(rel_vec.mean())

        coverage = 0.0
        if self.corpus_embeddings is not None and self.corpus_embeddings.shape[0] > 0:
            doc2q = cosine_similarity_matrix(self.corpus_embeddings, query_embeddings)  # (m,n)
            max_per_doc = doc2q.max(axis=1)
            coverage = float(max_per_doc.mean())

        return EvaluationMetrics(mean_intra, median_intra, dup_rate, avg_rel, coverage)
