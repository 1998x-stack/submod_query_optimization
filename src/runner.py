# -*- coding: utf-8 -*-
"""Experiment runner for ablations and benchmark comparisons."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from tabulate import tabulate

from src.baselines import (
    derive_topic_seed,
    mmr_select,
    random_select,
    top_relevance_select,
)
from src.cache import CandidateCache
from src.config import EmbeddingConfig, ExperimentConfig, LLMConfig
from src.corpus import CorpusLoader
from src.embeddings import LangChainM3EEmbedder
from src.evaluator import EvaluationMetrics, Evaluator
from src.llm import OpenAILLM
from src.objectives import FacilityLocationObjective, GraphCutObjective
from src.selectors import GreedyResult, LazyGreedySelector, StandardGreedySelector
from src.utils import (
    cosine_similarity_matrix,
    is_monotonic_non_decreasing,
    pairwise_cosine,
    slugify,
)


class AblationRunner:
    """Coordinate candidate generation, selection, evaluation, and artifacts."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._candidate_cache: dict[tuple[str, str], list[str]] = {}
        self._disk_cache = (
            CandidateCache(cfg.candidate_cache_dir)
            if cfg.use_candidate_cache
            else None
        )

        self.embedder = LangChainM3EEmbedder(EmbeddingConfig(cfg.m3e_path))
        self.llm = OpenAILLM(LLMConfig(model_name=cfg.llm_model))

        corpus_loader = CorpusLoader(
            cfg.data_dir,
            self.embedder,
            cfg.chunk_size,
            cfg.chunk_overlap,
        )
        self.corpus_texts, self.corpus_emb = corpus_loader.load()
        self.evaluator = Evaluator(self.embedder, self.corpus_emb)

        self._write_json(
            "experiment_config.json",
            {
                "config": dataclasses.asdict(cfg),
                "corpus_chunks": len(self.corpus_texts),
            },
        )
        logger.info(
            "Runner initialized: topics={}, num_candidates={}, k={}, corpus_chunks={}, disk_cache={}",
            len(cfg.topics),
            cfg.num_candidates,
            cfg.k,
            len(self.corpus_texts),
            bool(self._disk_cache),
        )

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        """Write JSON through a temporary file to avoid partial artifacts."""
        target = self.output_dir / filename
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(target)

    def _write_csv(
        self,
        filename: str,
        fieldnames: list[str],
        rows: list[dict[str, Any]],
    ) -> None:
        """Atomically write a UTF-8 CSV artifact."""
        target = self.output_dir / filename
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(target)

    def _candidate_cache_metadata(self, topic: str, mode: str) -> dict[str, Any]:
        """Build a cache identity from every generation input that affects output."""
        if mode == "simple":
            user_prompt = self.llm.build_prompt_simple(topic, self.cfg.num_candidates)
        elif mode == "structured":
            user_prompt = self.llm.build_prompt_structured(topic, self.cfg.num_candidates)
        else:
            raise ValueError("unsupported candidate generation mode")

        return {
            "topic": topic,
            "mode": mode,
            "num_candidates": self.cfg.num_candidates,
            "model_name": self.llm.cfg.model_name,
            "temperature": self.llm.cfg.temperature,
            "max_tokens": self.llm.cfg.max_tokens,
            "system_prompt": self.llm.cfg.system_prompt,
            "user_prompt": user_prompt,
        }

    def get_candidates(self, topic: str, mode: str) -> list[str]:
        """Public candidate-provider interface reused by the IR benchmark."""
        return self._get_candidates(topic, mode)

    def _get_candidates(self, topic: str, mode: str) -> list[str]:
        """Reuse candidate sets in memory and, optionally, across process runs."""
        key = (topic, mode)
        if key in self._candidate_cache:
            return list(self._candidate_cache[key])

        metadata = self._candidate_cache_metadata(topic, mode)
        if self._disk_cache is not None:
            cached = self._disk_cache.load(metadata)
            if cached is not None:
                logger.info(
                    "Candidate cache hit: topic={!r}, mode={}, count={}",
                    topic,
                    mode,
                    len(cached),
                )
                self._candidate_cache[key] = cached
                return list(cached)

        generated = self.llm.generate_candidates(
            topic,
            self.cfg.num_candidates,
            mode,
        )
        self._candidate_cache[key] = generated

        if self._disk_cache is not None and generated:
            path = self._disk_cache.store(metadata, generated)
            logger.info("Candidate cache write: {}", path)

        return list(generated)

    def _evaluate_indices(
        self,
        candidates: list[str],
        candidate_embeddings: np.ndarray,
        topic_embedding: np.ndarray,
        selected_indices: list[int],
    ) -> EvaluationMetrics:
        selected_queries = [candidates[index] for index in selected_indices]
        selected_embeddings = (
            candidate_embeddings[selected_indices]
            if selected_indices
            else None
        )
        return self.evaluator.evaluate(
            selected_queries,
            selected_embeddings,
            topic_embedding,
        )

    def run_prompt_ablation(self) -> None:
        """Compare simple and structured prompts on the same evaluation pipeline."""
        logger.info("[ABLT] Prompt Ablation: simple vs structured")
        rows: list[list[Any]] = []

        for topic in self.cfg.topics:
            topic_emb = self.embedder.encode([topic])[0]
            cand_simple = self._get_candidates(topic, "simple")
            cand_struct = self._get_candidates(topic, "structured")

            emb_simple = self.embedder.encode(cand_simple)
            emb_struct = self.embedder.encode(cand_struct)

            m_simple = self.evaluator.evaluate(cand_simple, emb_simple, topic_emb)
            m_struct = self.evaluator.evaluate(cand_struct, emb_struct, topic_emb)

            rows.append([
                topic,
                len(cand_simple),
                f"{m_simple.mean_intra_similarity:.3f}",
                f"{m_simple.duplicate_rate_08:.2%}",
                f"{m_simple.avg_relevance_to_topic:.3f}",
                f"{m_simple.coverage_score_vs_corpus:.3f}",
                len(cand_struct),
                f"{m_struct.mean_intra_similarity:.3f}",
                f"{m_struct.duplicate_rate_08:.2%}",
                f"{m_struct.avg_relevance_to_topic:.3f}",
                f"{m_struct.coverage_score_vs_corpus:.3f}",
            ])

            self._write_json(
                f"prompt_ablation_{slugify(topic)}.json",
                {
                    "topic": topic,
                    "candidates_simple": cand_simple,
                    "candidates_structured": cand_struct,
                    "metrics_simple": dataclasses.asdict(m_simple),
                    "metrics_structured": dataclasses.asdict(m_struct),
                },
            )

        headers = [
            "Topic",
            "#Simple",
            "Simple-MeanIntra",
            "Simple-Dup@0.8",
            "Simple-AvgRel",
            "Simple-Coverage",
            "#Struct",
            "Struct-MeanIntra",
            "Struct-Dup@0.8",
            "Struct-AvgRel",
            "Struct-Coverage",
        ]
        logger.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))

    def run_submodular_ablation(self) -> None:
        """Compare objectives and greedy implementations using identical candidates."""
        logger.info("[ABLT] Submodular Objectives x Greedy Variants")
        rows: list[list[Any]] = []

        for topic in self.cfg.topics:
            topic_emb = self.embedder.encode([topic])[0]
            candidates = self._get_candidates(topic, "simple")

            if not candidates:
                logger.error("Skipping topic {!r}: no valid candidates", topic)
                self._write_json(
                    f"submod_ablation_{slugify(topic)}.json",
                    {
                        "topic": topic,
                        "status": "skipped",
                        "reason": "no valid candidates",
                    },
                )
                continue

            if len(candidates) < self.cfg.k:
                logger.warning(
                    "Topic {!r} has only {} candidates for k={}; selectors will cap the budget",
                    topic,
                    len(candidates),
                    self.cfg.k,
                )

            cand_emb = self.embedder.encode(candidates)
            sim = pairwise_cosine(cand_emb)
            rel = cosine_similarity_matrix(
                cand_emb,
                topic_emb[None, :],
            ).reshape(-1)

            obj_fl = FacilityLocationObjective(
                sim,
                rel,
                alpha=self.cfg.alpha,
            )
            obj_gc = GraphCutObjective(
                sim,
                rel,
                alpha=self.cfg.alpha,
                lambda_div=self.cfg.lambda_diversity,
            )

            results = {
                "FL_Std": StandardGreedySelector(obj_fl).select(self.cfg.k),
                "FL_Lazy": LazyGreedySelector(obj_fl).select(self.cfg.k),
                "GC_Std": StandardGreedySelector(obj_gc).select(self.cfg.k),
                "GC_Lazy": LazyGreedySelector(obj_gc).select(self.cfg.k),
            }
            metrics = {
                name: self._evaluate_indices(
                    candidates,
                    cand_emb,
                    topic_emb,
                    result.selected_indices,
                )
                for name, result in results.items()
            }

            for name, result in results.items():
                metric = metrics[name]
                rows.append([
                    topic,
                    name.replace("_", "-"),
                    len(result.selected_indices),
                    f"{result.runtime_sec:.3f}s",
                    f"{metric.mean_intra_similarity:.3f}",
                    f"{metric.duplicate_rate_08:.2%}",
                    f"{metric.avg_relevance_to_topic:.3f}",
                    f"{metric.coverage_score_vs_corpus:.3f}",
                ])

            self._write_json(
                f"submod_ablation_{slugify(topic)}.json",
                {
                    "topic": topic,
                    "candidates": candidates,
                    "selected": {
                        name: result.selected_indices
                        for name, result in results.items()
                    },
                    "selected_queries": {
                        name: [candidates[index] for index in result.selected_indices]
                        for name, result in results.items()
                    },
                    "objective_values": {
                        name: result.objective_values
                        for name, result in results.items()
                    },
                    "per_step_gains": {
                        name: result.per_step_gains
                        for name, result in results.items()
                    },
                    "runtimes": {
                        name: result.runtime_sec
                        for name, result in results.items()
                    },
                    "metrics": {
                        name: dataclasses.asdict(metric)
                        for name, metric in metrics.items()
                    },
                },
            )

            self._recheck_monotonicity(
                topic,
                {
                    name: result.objective_values
                    for name, result in results.items()
                },
            )
            self._recheck_solution_consistency(
                topic,
                results["FL_Std"],
                results["FL_Lazy"],
                "FacilityLocation",
            )
            self._recheck_solution_consistency(
                topic,
                results["GC_Std"],
                results["GC_Lazy"],
                "GraphCut",
            )

        headers = [
            "Topic",
            "Method",
            "#Selected",
            "Runtime",
            "MeanIntraSim",
            "Dup@0.8",
            "AvgRel",
            "Coverage",
        ]
        logger.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))

    def run_baseline_benchmark(self) -> None:
        """Compare submodular selectors against deterministic baseline methods."""
        logger.info("[BENCH] Top-Relevance / MMR / Random / FL / GraphCut")
        table_rows: list[list[Any]] = []
        csv_rows: list[dict[str, Any]] = []

        for topic in self.cfg.topics:
            topic_emb = self.embedder.encode([topic])[0]
            candidates = self._get_candidates(topic, "simple")
            if not candidates:
                logger.error("Skipping benchmark topic {!r}: no valid candidates", topic)
                continue

            cand_emb = self.embedder.encode(candidates)
            sim = pairwise_cosine(cand_emb)
            rel = cosine_similarity_matrix(
                cand_emb,
                topic_emb[None, :],
            ).reshape(-1)

            topic_seed = derive_topic_seed(self.cfg.random_seed, topic)
            selections = {
                "Top-Relevance": top_relevance_select(sim, rel, self.cfg.k),
                "MMR": mmr_select(
                    sim,
                    rel,
                    self.cfg.k,
                    lambda_relevance=self.cfg.mmr_lambda,
                ),
                "Random": random_select(
                    sim,
                    rel,
                    self.cfg.k,
                    seed=topic_seed,
                ),
                "FacilityLocation": LazyGreedySelector(
                    FacilityLocationObjective(
                        sim,
                        rel,
                        alpha=self.cfg.alpha,
                    )
                ).select(self.cfg.k),
                "GraphCut": LazyGreedySelector(
                    GraphCutObjective(
                        sim,
                        rel,
                        alpha=self.cfg.alpha,
                        lambda_div=self.cfg.lambda_diversity,
                    )
                ).select(self.cfg.k),
            }

            metrics = {
                name: self._evaluate_indices(
                    candidates,
                    cand_emb,
                    topic_emb,
                    result.selected_indices,
                )
                for name, result in selections.items()
            }

            topic_payload: dict[str, Any] = {
                "topic": topic,
                "candidate_count": len(candidates),
                "benchmark_parameters": {
                    "k": self.cfg.k,
                    "alpha": self.cfg.alpha,
                    "lambda_diversity": self.cfg.lambda_diversity,
                    "mmr_lambda": self.cfg.mmr_lambda,
                    "base_random_seed": self.cfg.random_seed,
                    "topic_random_seed": topic_seed,
                },
                "methods": {},
            }

            for name, result in selections.items():
                metric = metrics[name]
                selected_indices = list(result.selected_indices)
                runtime_sec = float(result.runtime_sec)

                topic_payload["methods"][name] = {
                    "selected_indices": selected_indices,
                    "selected_queries": [
                        candidates[index] for index in selected_indices
                    ],
                    "runtime_sec": runtime_sec,
                    "metrics": dataclasses.asdict(metric),
                }

                row = {
                    "topic": topic,
                    "method": name,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected_indices),
                    "runtime_sec": runtime_sec,
                    "mean_intra_similarity": metric.mean_intra_similarity,
                    "median_intra_similarity": metric.median_intra_similarity,
                    "duplicate_rate_08": metric.duplicate_rate_08,
                    "avg_relevance_to_topic": metric.avg_relevance_to_topic,
                    "coverage_score_vs_corpus": metric.coverage_score_vs_corpus,
                }
                csv_rows.append(row)
                table_rows.append([
                    topic,
                    name,
                    len(selected_indices),
                    f"{runtime_sec:.4f}s",
                    f"{metric.mean_intra_similarity:.3f}",
                    f"{metric.duplicate_rate_08:.2%}",
                    f"{metric.avg_relevance_to_topic:.3f}",
                    f"{metric.coverage_score_vs_corpus:.3f}",
                ])

            self._write_json(
                f"benchmark_{slugify(topic)}.json",
                topic_payload,
            )

        fieldnames = [
            "topic",
            "method",
            "candidate_count",
            "selected_count",
            "runtime_sec",
            "mean_intra_similarity",
            "median_intra_similarity",
            "duplicate_rate_08",
            "avg_relevance_to_topic",
            "coverage_score_vs_corpus",
        ]
        self._write_csv(
            "benchmark_summary.csv",
            fieldnames,
            csv_rows,
        )

        headers = [
            "Topic",
            "Method",
            "#Selected",
            "Runtime",
            "MeanIntraSim",
            "Dup@0.8",
            "AvgRel",
            "Coverage",
        ]
        logger.info("\n" + tabulate(table_rows, headers=headers, tablefmt="github"))
        logger.info(
            "Benchmark summary written to {}",
            self.output_dir / "benchmark_summary.csv",
        )

    @staticmethod
    def _recheck_monotonicity(
        topic: str,
        curves: dict[str, list[float]],
    ) -> None:
        for key, values in curves.items():
            if values and not is_monotonic_non_decreasing(values):
                logger.warning(
                    "{} - {} objective values decreased despite accepting positive gains",
                    topic,
                    key,
                )

    @staticmethod
    def _recheck_solution_consistency(
        topic: str,
        std_res: GreedyResult,
        lazy_res: GreedyResult,
        tag: str,
    ) -> None:
        std_final = std_res.objective_values[-1] if std_res.objective_values else 0.0
        lazy_final = lazy_res.objective_values[-1] if lazy_res.objective_values else 0.0

        if not np.isclose(std_final, lazy_final, rtol=1e-5, atol=1e-6):
            logger.warning(
                "{} - {}: standard/lazy objective mismatch: {:.8f} vs {:.8f}",
                topic,
                tag,
                std_final,
                lazy_final,
            )
        elif std_res.selected_indices != lazy_res.selected_indices:
            logger.info(
                "{} - {}: standard/lazy selected different but objective-equivalent sets",
                topic,
                tag,
            )
