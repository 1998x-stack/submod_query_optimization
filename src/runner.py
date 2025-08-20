# -*- coding: utf-8 -*-
"""Experiment runner for ablation studies and final checks."""

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
from src.utils import cosine_similarity_matrix, pairwise_cosine, slugify, is_monotonic_non_decreasing


class AblationRunner:
    """组织并运行消融实验：Prompt / Objective / GreedyVariant。"""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化嵌入 & LLM
        self.embedder = LangChainM3EEmbedder(EmbeddingConfig(cfg.m3e_path))
        self.llm = OpenAILLM(LLMConfig(model_name=cfg.llm_model))

        # 语料分块与向量（LangChain）
        corpus_loader = CorpusLoader(cfg.data_dir, self.embedder, cfg.chunk_size, cfg.chunk_overlap)
        self.corpus_texts, self.corpus_emb = corpus_loader.load()

        logger.info("Runner initialized. topics={}, num_candidates={}, k={}",
                    len(cfg.topics), cfg.num_candidates, cfg.k)

    # -------------------------- Prompt Ablation --------------------------

    def run_prompt_ablation(self) -> None:
        """比较简单提示词 vs 结构化提示词（验证 LLM 主流偏好 & 冷门缺失）。"""
        logger.info("[ABLT] Prompt Ablation — 简单提示词 vs 结构化提示词")
        rows: list[list[Any]] = []

        for topic in self.cfg.topics:
            topic_emb = self.embedder.encode([topic])[0]

            cand_simple = self.llm.generate_candidates(topic, self.cfg.num_candidates, "simple")
            cand_struct = self.llm.generate_candidates(topic, self.cfg.num_candidates, "structured")

            emb_simple = self.embedder.encode(cand_simple)
            emb_struct = self.embedder.encode(cand_struct)

            evaluator = Evaluator(self.embedder, self.corpus_emb)
            m_simple = evaluator.evaluate(cand_simple, emb_simple, topic_emb)
            m_struct = evaluator.evaluate(cand_struct, emb_struct, topic_emb)

            rows.append([
                topic, len(cand_simple),
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

            # 保存 JSON，方便人工复核“冷门角度缺失”
            out = {
                "topic": topic,
                "candidates_simple": cand_simple,
                "candidates_structured": cand_struct,
                "metrics_simple": dataclasses.asdict(m_simple),
                "metrics_structured": dataclasses.asdict(m_struct),
            }
            (self.output_dir / f"prompt_ablation_{slugify(topic)}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        headers = [
            "Topic", "#Simple", "Simple-MeanIntra", "Simple-Dup@0.8",
            "Simple-AvgRel", "Simple-Coverage",
            "#Struct", "Struct-MeanIntra", "Struct-Dup@0.8",
            "Struct-AvgRel", "Struct-Coverage",
        ]
        logger.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))
        logger.info("说明：MeanIntra越低越多样；Dup@0.8越低越少复述；AvgRel越高越贴题；Coverage越高覆盖越广。")

    # ------------------ Submodular × Greedy Ablation ------------------

    def run_submodular_ablation(self) -> None:
        """对比 设施选址 vs 图割 × 标准贪心 vs 懒贪心。"""
        logger.info("[ABLT] Submodular Objectives × Greedy Variants")
        rows: list[list[Any]] = []

        for topic in self.cfg.topics:
            topic_emb = self.embedder.encode([topic])[0]

            # 用相同的候选集合，公平比较不同目标/算法
            candidates = self.llm.generate_candidates(topic, self.cfg.num_candidates, "simple")
            cand_emb = self.embedder.encode(candidates)

            sim = pairwise_cosine(cand_emb)
            rel = cosine_similarity_matrix(cand_emb, topic_emb[None, :]).reshape(-1)

            obj_fl = FacilityLocationObjective(sim, rel, alpha=self.cfg.alpha)
            obj_gc = GraphCutObjective(sim, rel, alpha=self.cfg.alpha, lambda_div=self.cfg.lambda_diversity)

            sel_std_fl = StandardGreedySelector(obj_fl)
            sel_lazy_fl = LazyGreedySelector(obj_fl)
            sel_std_gc = StandardGreedySelector(obj_gc)
            sel_lazy_gc = LazyGreedySelector(obj_gc)

            res_std_fl = sel_std_fl.select(self.cfg.k)
            res_lazy_fl = sel_lazy_fl.select(self.cfg.k)
            res_std_gc = sel_std_gc.select(self.cfg.k)
            res_lazy_gc = sel_lazy_gc.select(self.cfg.k)

            evaluator = Evaluator(self.embedder, self.corpus_emb)

            def eval_sel(res: GreedyResult) -> EvaluationMetrics:
                qs = [candidates[i] for i in res.selected_indices]
                q_emb = cand_emb[res.selected_indices] if res.selected_indices else None
                return evaluator.evaluate(qs, q_emb, topic_emb)

            m_std_fl = eval_sel(res_std_fl)
            m_lazy_fl = eval_sel(res_lazy_fl)
            m_std_gc = eval_sel(res_std_gc)
            m_lazy_gc = eval_sel(res_lazy_gc)

            def summarize_row(name: str, res: GreedyResult, m: EvaluationMetrics) -> list[Any]:
                return [
                    topic, name, len(res.selected_indices), f"{res.runtime_sec:.3f}s",
                    f"{m.mean_intra_similarity:.3f}", f"{m.duplicate_rate_08:.2%}",
                    f"{m.avg_relevance_to_topic:.3f}", f"{m.coverage_score_vs_corpus:.3f}",
                ]

            rows.append(summarize_row("FL-Std", res_std_fl, m_std_fl))
            rows.append(summarize_row("FL-Lazy", res_lazy_fl, m_lazy_fl))
            rows.append(summarize_row("GC-Std", res_std_gc, m_std_gc))
            rows.append(summarize_row("GC-Lazy", res_lazy_gc, m_lazy_gc))

            # 保存细节（增益曲线、选择集与指标）
            out = {
                "topic": topic,
                "candidates": candidates,
                "selected": {
                    "FL_Std": res_std_fl.selected_indices,
                    "FL_Lazy": res_lazy_fl.selected_indices,
                    "GC_Std": res_std_gc.selected_indices,
                    "GC_Lazy": res_lazy_gc.selected_indices,
                },
                "objective_values": {
                    "FL_Std": res_std_fl.objective_values,
                    "FL_Lazy": res_lazy_fl.objective_values,
                    "GC_Std": res_std_gc.objective_values,
                    "GC_Lazy": res_lazy_gc.objective_values,
                },
                "runtimes": {
                    "FL_Std": res_std_fl.runtime_sec,
                    "FL_Lazy": res_lazy_fl.runtime_sec,
                    "GC_Std": res_std_gc.runtime_sec,
                    "GC_Lazy": res_lazy_gc.runtime_sec,
                },
                "metrics": {
                    "FL_Std": dataclasses.asdict(m_std_fl),
                    "FL_Lazy": dataclasses.asdict(m_lazy_fl),
                    "GC_Std": dataclasses.asdict(m_std_gc),
                    "GC_Lazy": dataclasses.asdict(m_lazy_gc),
                },
            }
            (self.output_dir / f"submod_ablation_{slugify(topic)}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # ---- 关键正确性复核（Correctness Recheck） ----
            self._recheck_monotonicity(topic, {
                "FL_Std": res_std_fl.objective_values,
                "FL_Lazy": res_lazy_fl.objective_values,
                "GC_Std": res_std_gc.objective_values,
                "GC_Lazy": res_lazy_gc.objective_values,
            })
            self._recheck_solution_consistency(topic, res_std_fl, res_lazy_fl, "FL")
            self._recheck_solution_consistency(topic, res_std_gc, res_lazy_gc, "GC")

        headers = ["Topic", "Method", "#Selected", "Runtime",
                   "MeanIntraSim", "Dup@0.8", "AvgRel", "Coverage"]
        logger.info("\n" + tabulate(rows, headers=headers, tablefmt="github"))
        logger.info("注：Runtime 体现懒贪心的效率；MeanIntra/Dup@0.8 越低越好；AvgRel/Coverage 越高越好。")

    # -------------------------- Recheck Helpers --------------------------

    @staticmethod
    def _recheck_monotonicity(topic: str, curves: dict[str, list[float]]) -> None:
        """检查目标函数曲线单调不降。"""
        for key, vals in curves.items():
            if not vals:
                continue
            if not is_monotonic_non_decreasing(vals):
                logger.warning("[WARN] {} - {} 目标值非单调，请检查实现/数据。", topic, key)

    @staticmethod
    def _recheck_solution_consistency(topic: str, std_res: GreedyResult, lazy_res: GreedyResult, tag: str) -> None:
        """检查标准与懒贪心在同一目标函数下是否得到一致或等价解。"""
        if std_res.selected_indices != lazy_res.selected_indices:
            logger.info("[INFO] {} - {}: 懒贪心与标准贪心解不同（可能同分多解属正常）。", topic, tag)
