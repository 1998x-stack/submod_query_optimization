# -*- coding: utf-8 -*-
"""Qrels-backed offline benchmark for query expansion/selection methods."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

from src.baselines import derive_topic_seed, mmr_select, random_select, top_relevance_select
from src.config import ExperimentConfig
from src.embeddings import LangChainM3EEmbedder
from src.ir_dataset import IRDataset, IRQuery
from src.ir_metrics import IRMetrics, evaluate_ranking
from src.ir_retrieval import DenseMultiQueryRetriever, RetrievalResult
from src.ir_stats import bootstrap_mean_ci
from src.objectives import FacilityLocationObjective, GraphCutObjective
from src.selectors import LazyGreedySelector
from src.utils import cosine_similarity_matrix, pairwise_cosine


CandidateProvider = Callable[[str, str], list[str]]


@dataclass(frozen=True, slots=True)
class SelectionRun:
    variant: str
    run_index: int
    selected_indices: list[int]
    runtime_sec: float


class IRBenchmarkRunner:
    """Evaluate expansion-query selection against document-level ground truth."""

    METRIC_NAMES = ("precision", "recall", "hit_rate", "reciprocal_rank", "ndcg")
    SUMMARY_NAMES = {
        "precision": "precision",
        "recall": "recall",
        "hit_rate": "hit_rate",
        "reciprocal_rank": "mrr",
        "ndcg": "ndcg",
    }

    def __init__(
        self,
        cfg: ExperimentConfig,
        embedder: LangChainM3EEmbedder,
        candidate_provider: CandidateProvider,
    ) -> None:
        if not cfg.ir_dataset_dir:
            raise ValueError("ir_dataset_dir is required for the IR benchmark")

        self.cfg = cfg
        self.embedder = embedder
        self.candidate_provider = candidate_provider
        self.dataset = IRDataset.load(cfg.ir_dataset_dir)
        self.output_dir = Path(cfg.output_dir) / "ir"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        document_ids = [doc.doc_id for doc in self.dataset.documents]
        document_texts = [doc.text for doc in self.dataset.documents]
        logger.info("IR: embedding {} benchmark documents", len(document_texts))
        document_embeddings = self.embedder.encode(document_texts)
        self.retriever = DenseMultiQueryRetriever(document_ids, document_embeddings)

    def run(self) -> None:
        detail_rows: list[dict[str, object]] = []
        max_cutoff = max(self.cfg.ir_cutoffs)

        for query in self.dataset.queries:
            detail_rows.extend(self._run_query(query, max_cutoff))

        self._write_csv(self.output_dir / "ir_benchmark_detail.csv", detail_rows)
        summary_rows = self._summarize(detail_rows)
        self._write_csv(self.output_dir / "ir_benchmark_summary.csv", summary_rows)
        self._write_manifest()

        logger.info(
            "IR benchmark complete: queries={}, documents={}, variants={}, output={}",
            len(self.dataset.queries),
            len(self.dataset.documents),
            len({str(row["variant"]) for row in detail_rows}),
            self.output_dir,
        )

    def _run_query(self, query: IRQuery, max_cutoff: int) -> list[dict[str, object]]:
        original_embedding = self.embedder.encode([query.text])
        candidates = self.candidate_provider(query.text, "simple")
        if not candidates:
            raise RuntimeError(
                f"IR query {query.query_id!r} produced no expansion candidates"
            )

        candidate_embeddings = self.embedder.encode(candidates)
        similarity = pairwise_cosine(candidate_embeddings)
        relevance = cosine_similarity_matrix(
            candidate_embeddings,
            original_embedding,
        ).reshape(-1)

        selections = self._build_selections(
            query=query,
            similarity=similarity,
            relevance=relevance,
        )
        qrels = self.dataset.qrels[query.query_id]
        rows: list[dict[str, object]] = []

        for selection in selections:
            selected_embeddings = (
                candidate_embeddings[selection.selected_indices]
                if selection.selected_indices
                else np.empty((0, candidate_embeddings.shape[1]), dtype=np.float32)
            )

            # The original query is always preserved. Selection methods only
            # decide which expansion queries are added.
            retrieval_embeddings = np.concatenate(
                [original_embedding, selected_embeddings],
                axis=0,
            )
            ranking = self.retriever.rank(
                retrieval_embeddings,
                top_k=max_cutoff,
            )

            for cutoff in self.cfg.ir_cutoffs:
                metrics = evaluate_ranking(ranking.doc_ids, qrels, cutoff)
                rows.append(
                    self._detail_row(
                        query=query,
                        candidates=candidates,
                        selection=selection,
                        candidate_count=len(candidates),
                        ranking=ranking,
                        metrics=metrics,
                    )
                )

        return rows

    def _build_selections(
        self,
        *,
        query: IRQuery,
        similarity: np.ndarray,
        relevance: np.ndarray,
    ) -> list[SelectionRun]:
        selections = [
            SelectionRun("OriginalQuery", 0, [], 0.0),
        ]

        for selection_k in self.cfg.ir_k_grid or (self.cfg.k,):
            top = top_relevance_select(similarity, relevance, selection_k)
            selections.append(
                SelectionRun(
                    f"TopRelevance(k={selection_k})",
                    0,
                    top.selected_indices,
                    top.runtime_sec,
                )
            )

            for mmr_lambda in self.cfg.ir_mmr_grid:
                result = mmr_select(
                    similarity,
                    relevance,
                    selection_k,
                    lambda_relevance=mmr_lambda,
                )
                selections.append(
                    SelectionRun(
                        f"MMR(k={selection_k},lambda={mmr_lambda:g})",
                        0,
                        result.selected_indices,
                        result.runtime_sec,
                    )
                )

            for alpha in self.cfg.ir_alpha_grid:
                result = LazyGreedySelector(
                    FacilityLocationObjective(
                        similarity,
                        relevance,
                        alpha=alpha,
                    )
                ).select(selection_k)
                selections.append(
                    SelectionRun(
                        f"FacilityLocation(k={selection_k},alpha={alpha:g})",
                        0,
                        result.selected_indices,
                        result.runtime_sec,
                    )
                )

            for alpha in self.cfg.ir_alpha_grid:
                for lambda_div in self.cfg.ir_lambda_grid:
                    result = LazyGreedySelector(
                        GraphCutObjective(
                            similarity,
                            relevance,
                            alpha=alpha,
                            lambda_div=lambda_div,
                        )
                    ).select(selection_k)
                    selections.append(
                        SelectionRun(
                            (
                                f"GraphCut(k={selection_k},alpha={alpha:g},"
                                f"lambda={lambda_div:g})"
                            ),
                            0,
                            result.selected_indices,
                            result.runtime_sec,
                        )
                    )

            base_seed = derive_topic_seed(
                self.cfg.random_seed,
                f"{query.query_id}:k={selection_k}",
            )
            for repeat in range(self.cfg.ir_random_repeats):
                result = random_select(
                    similarity,
                    relevance,
                    selection_k,
                    seed=base_seed + repeat,
                )
                selections.append(
                    SelectionRun(
                        f"Random(k={selection_k})",
                        repeat,
                        result.selected_indices,
                        result.runtime_sec,
                    )
                )

        return selections

    @staticmethod
    def _detail_row(
        *,
        query: IRQuery,
        candidates: list[str],
        selection: SelectionRun,
        candidate_count: int,
        ranking: RetrievalResult,
        metrics: IRMetrics,
    ) -> dict[str, object]:
        cutoff = metrics.cutoff
        selected_queries = [
            candidates[index] for index in selection.selected_indices
        ]
        return {
            "query_id": query.query_id,
            "query_text": query.text,
            "variant": selection.variant,
            "run_index": selection.run_index,
            "candidate_count": candidate_count,
            "selected_count": len(selection.selected_indices),
            "selection_runtime_sec": selection.runtime_sec,
            "selected_indices_json": json.dumps(selection.selected_indices),
            "selected_queries_json": json.dumps(selected_queries, ensure_ascii=False),
            "cutoff": cutoff,
            "retrieved_doc_ids_json": json.dumps(ranking.doc_ids[:cutoff]),
            "retrieved_scores_json": json.dumps(ranking.scores[:cutoff]),
            "precision": metrics.precision,
            "recall": metrics.recall,
            "hit_rate": metrics.hit_rate,
            "reciprocal_rank": metrics.reciprocal_rank,
            "ndcg": metrics.ndcg,
        }

    def _summarize(
        self,
        detail_rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        variants = sorted({str(row["variant"]) for row in detail_rows})
        summary: list[dict[str, object]] = []

        for variant in variants:
            for cutoff in self.cfg.ir_cutoffs:
                matching = [
                    row
                    for row in detail_rows
                    if row["variant"] == variant and row["cutoff"] == cutoff
                ]

                # Random repeats are averaged within each query before macro
                # averaging/bootstrap so the statistical unit remains a query.
                per_query: dict[str, dict[str, list[float]]] = {}
                for row in matching:
                    query_metrics = per_query.setdefault(
                        str(row["query_id"]),
                        {name: [] for name in self.METRIC_NAMES},
                    )
                    for metric_name in self.METRIC_NAMES:
                        query_metrics[metric_name].append(float(row[metric_name]))

                output: dict[str, object] = {
                    "variant": variant,
                    "cutoff": cutoff,
                    "query_count": len(per_query),
                }
                for metric_name in self.METRIC_NAMES:
                    query_values = [
                        float(np.mean(values[metric_name]))
                        for values in per_query.values()
                    ]
                    ci = bootstrap_mean_ci(
                        query_values,
                        iterations=self.cfg.ir_bootstrap_iterations,
                        seed=self.cfg.random_seed,
                    )
                    output_name = self.SUMMARY_NAMES[metric_name]
                    output[f"{output_name}_mean"] = ci.mean
                    output[f"{output_name}_ci_low"] = ci.low
                    output[f"{output_name}_ci_high"] = ci.high
                summary.append(output)

        return summary

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            raise ValueError("cannot write an empty benchmark CSV")
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)

    def _write_manifest(self) -> None:
        qrel_count = sum(len(items) for items in self.dataset.qrels.values())
        positive_qrels = sum(
            1
            for items in self.dataset.qrels.values()
            for relevance in items.values()
            if relevance > 0
        )
        payload = {
            "dataset_dir": self.cfg.ir_dataset_dir,
            "document_count": len(self.dataset.documents),
            "query_count": len(self.dataset.queries),
            "qrel_count": qrel_count,
            "positive_qrel_count": positive_qrels,
            "cutoffs": list(self.cfg.ir_cutoffs),
            "selection_k_grid": list(self.cfg.ir_k_grid or (self.cfg.k,)),
            "default_k": self.cfg.k,
            "alpha_grid": list(self.cfg.ir_alpha_grid),
            "lambda_grid": list(self.cfg.ir_lambda_grid),
            "mmr_grid": list(self.cfg.ir_mmr_grid),
            "random_repeats": self.cfg.ir_random_repeats,
            "bootstrap_iterations": self.cfg.ir_bootstrap_iterations,
            "metric_semantics": {
                "binary_relevant": "qrel > 0",
                "unjudged_documents": "treated as non-relevant",
                "precision": "Precision@k with denominator k",
                "recall": "Recall@k over all positive qrels",
                "hit_rate": "1 if any positive qrel appears in top-k",
                "mrr": "mean reciprocal rank of first positive qrel within top-k",
                "ndcg": "graded nDCG@k using gain=2^rel-1",
                "aggregation": "macro average across queries",
                "confidence_interval": "percentile bootstrap across query-level values",
            },
            "retrieval": {
                "similarity": "cosine",
                "fusion": "max over original + selected expansion queries",
                "query_set": "original query is always retained",
            },
        }
        path = self.output_dir / "ir_benchmark_manifest.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
