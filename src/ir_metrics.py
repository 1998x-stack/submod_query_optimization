# -*- coding: utf-8 -*-
"""Standard information-retrieval metrics for graded qrels."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IRMetrics:
    cutoff: int
    precision: float
    recall: float
    hit_rate: float
    reciprocal_rank: float
    ndcg: float


def _dcg(relevances: list[int]) -> float:
    return sum(
        (2.0 ** rel - 1.0) / math.log2(rank + 2.0)
        for rank, rel in enumerate(relevances)
    )


def evaluate_ranking(
    ranked_doc_ids: list[str],
    qrels: dict[str, int],
    cutoff: int,
) -> IRMetrics:
    """Evaluate one ranked list against one query's graded relevance judgments.

    Unjudged documents are treated as non-relevant. Precision@k uses k as the
    denominator, matching the standard fixed-cutoff definition.
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if len(ranked_doc_ids) != len(set(ranked_doc_ids)):
        raise ValueError("ranked_doc_ids must not contain duplicates")

    top = ranked_doc_ids[:cutoff]
    binary = [1 if qrels.get(doc_id, 0) > 0 else 0 for doc_id in top]
    graded = [max(0, int(qrels.get(doc_id, 0))) for doc_id in top]

    relevant_total = sum(1 for rel in qrels.values() if rel > 0)
    retrieved_relevant = sum(binary)

    precision = retrieved_relevant / cutoff
    recall = retrieved_relevant / relevant_total if relevant_total > 0 else 0.0
    hit_rate = 1.0 if retrieved_relevant > 0 else 0.0

    reciprocal_rank = 0.0
    for rank, is_relevant in enumerate(binary, start=1):
        if is_relevant:
            reciprocal_rank = 1.0 / rank
            break

    ideal_relevances = sorted(
        (max(0, int(rel)) for rel in qrels.values()),
        reverse=True,
    )[:cutoff]
    idcg = _dcg(ideal_relevances)
    ndcg = _dcg(graded) / idcg if idcg > 0.0 else 0.0

    return IRMetrics(
        cutoff=cutoff,
        precision=float(precision),
        recall=float(recall),
        hit_rate=float(hit_rate),
        reciprocal_rank=float(reciprocal_rank),
        ndcg=float(ndcg),
    )
