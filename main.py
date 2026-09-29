# -*- coding: utf-8 -*-
"""CLI entry for running ablations and benchmark comparisons."""

from __future__ import annotations

import argparse
import dataclasses

from loguru import logger

from src.config import ExperimentConfig
from src.logging_utils import setup_logging
from src.runner import AblationRunner


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Submodular Query Optimization — 少而多样的高质量查询"
    )
    parser.add_argument(
        "--topics",
        nargs="*",
        default=["embeddings and rerankers"],
        help="主题列表",
    )
    parser.add_argument("--num-candidates", type=int, default=20, help="候选查询数量")
    parser.add_argument("--k", type=int, default=6, help="最终选择的查询数")
    parser.add_argument("--alpha", type=float, default=0.5, help="Facility Location 相关性权重")
    parser.add_argument(
        "--lambda-diversity",
        type=float,
        default=0.5,
        help="Graph Cut 多样性权重",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.6,
        help="MMR 中 relevance 权重（0~1）",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random baseline 基础随机种子",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-4o", help="OpenAI 模型名")
    parser.add_argument("--m3e-path", type=str, default="model", help="本地 embedding 模型目录")
    parser.add_argument("--data-dir", type=str, default="data", help="语料目录")
    parser.add_argument("--output-dir", type=str, default="output", help="输出目录")
    parser.add_argument(
        "--candidate-cache-dir",
        type=str,
        default=".cache/candidates",
        help="LLM 候选查询磁盘缓存目录",
    )
    parser.add_argument(
        "--no-candidate-cache",
        action="store_true",
        help="关闭跨运行候选查询缓存",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="文本分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="文本分块重叠")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="日志等级：DEBUG/INFO/WARNING",
    )
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    setup_logging(args.output_dir, level=args.log_level)

    cfg = ExperimentConfig(
        topics=list(args.topics),
        num_candidates=int(args.num_candidates),
        k=int(args.k),
        alpha=float(args.alpha),
        lambda_diversity=float(args.lambda_diversity),
        mmr_lambda=float(args.mmr_lambda),
        random_seed=int(args.random_seed),
        chunk_size=int(args.chunk_size),
        chunk_overlap=int(args.chunk_overlap),
        m3e_path=str(args.m3e_path),
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        llm_model=str(args.llm_model),
        candidate_cache_dir=str(args.candidate_cache_dir),
        use_candidate_cache=not bool(args.no_candidate_cache),
    )

    if cfg.k > cfg.num_candidates:
        logger.warning("k 大于候选数，将按实际候选数量截断。")

    logger.info("Experiment Config:\n{}", dataclasses.asdict(cfg))
    runner = AblationRunner(cfg)

    runner.run_prompt_ablation()
    runner.run_submodular_ablation()
    runner.run_baseline_benchmark()

    logger.success("全部实验完成。输出与日志位于：{}", args.output_dir)


if __name__ == "__main__":
    main()
