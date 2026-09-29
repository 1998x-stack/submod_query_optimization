# -*- coding: utf-8 -*-
"""Experiment runner for prompt and submodular-selection ablations."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from tabulate import tabulate

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
            {"config": dataclasses.asdict(cfg), "corpus_chunks": len(self.corpus_texts)},
        )
        logger.info(
            "Runner initialized: topics={}, num_candidates={}, k={}, corpus_chunks={}",
            len(cfg.topics),
            cfg.num_candidates,
            cfg.k,
            len(self.corpus_texts),
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

    def _get_candidates(self, topic: str, mode: str) -> list[str]:
        """Generate each topic/mode candidate set once and reuse it across ablations."""
        key = (topic, mode)
        if key not in self._candidate_cache:
            self._candidate_cache[key] = self.llm.generate_candidates(
                topic,
                self.cfg.num_candidates,
                mode,
            )
        return list(self._candidate_cache[key])

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
                    {"topic": topic, "status": "skipped", "reason": "no valid candidates"},
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
            rel = cosine_similarity_matrix(cand_emb, topic_emb[None, :]).reshape(-1)

            obj_fl = FacilityLocationObjective(sim, rel, alpha=self.cfg.alpha)
            obj_gc = GraphCutObjective(
                sim,
                rel,
                alpha=self.cfg.alpha,
                lambda_div=self.cfg.lambda_diversity,
            )

            res_std_fl = StandardGreedySelector(obj_fl).select(self.cfg.k)
            res_lazy_fl = LazyGreedySelector(obj_fl).select(self.cfg.k)
            res_std_gc = StandardGreedySelector(obj_gc).select(self.cfg.k)
            res_lazy_gc = LazyGreedySelector(obj_gc).select(self.cfg.k)

            def eval_sel(res: GreedyResult) -> EvaluationMetrics:
                selected_queries = [candidates[i] for i in res.selected_indices]
                selected_emb = (
                    cand_emb[res.selected_indices]
                    if res.selected_indices
                    else None
                )
                return self.evaluator.evaluate(
                    selected_queries,
                    selected_emb,
                    topic_emb,
                )

            metrics = {
                "FL_Std": eval_sel(res_std_fl),
                "FL_Lazy": eval_sel(res_lazy_fl),
                "GC_Std": eval_sel(res_std_gc),
                "GC_Lazy": eval_sel(res_lazy_gc),
            }
            results = {
                "FL_Std": res_std_fl,
                "FL_Lazy": res_lazy_fl,
                "GC_Std": res_std_gc,
                "GC_Lazy": res_lazy_gc,
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
                        name: [candidates[i] for i in result.selected_indices]
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
                {name: result.objective_values for name, result in results.items()},
            )
            self._recheck_solution_consistency(
                topic, res_std_fl, res_lazy_fl, "FacilityLocation"
            )
            self._recheck_solution_consistency(
                topic, res_std_gc, res_lazy_gc, "GraphCut"
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
