# -*- coding: utf-8 -*-
"""CLI entry for running all experiments.

用法示例：
    export OPENAI_API_KEY=sk-...
    python main.py \
        --topics "embeddings and rerankers" "generative ai" \
        --num-candidates 20 \
        --k 6 \
        --alpha 0.5 \
        --lambda-diversity 0.5 \
        --llm-model gpt-4o \
        --m3e-path model \
        --data-dir data \
        --output-dir output \
        --chunk-size 800 \
        --chunk-overlap 120 \
        --log-level INFO
"""

from __future__ import annotations

import argparse
import dataclasses
from loguru import logger

from src.config import ExperimentConfig
from src.logging_utils import setup_logging
from src.runner import AblationRunner


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Submodular Query Optimization — 少而多样的高质量查询")
    p.add_argument("--topics", nargs="*", default=["embeddings and rerankers"], help="主题列表")
    p.add_argument("--num-candidates", type=int, default=20, help="候选查询数量")
    p.add_argument("--k", type=int, default=6, help="最终选择的查询数")
    p.add_argument("--alpha", type=float, default=0.5, help="相关性权重（0~1）")
    p.add_argument("--lambda-diversity", type=float, default=0.5, help="图割多样性权重（0~1）")
    p.add_argument("--llm-model", type=str, default="gpt-4o", help="OpenAI 模型名")
    p.add_argument("--m3e-path", type=str, default="model", help="本地 m3e 模型目录")
    p.add_argument("--data-dir", type=str, default="data", help="语料目录")
    p.add_argument("--output-dir", type=str, default="output", help="输出目录")
    p.add_argument("--chunk-size", type=int, default=800, help="LangChain 文本分块大小")
    p.add_argument("--chunk-overlap", type=int, default=120, help="LangChain 分块重叠")
    p.add_argument("--log-level", type=str, default="INFO", help="日志等级：DEBUG/INFO/WARNING")
    return p.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    setup_logging(args.output_dir, level=args.log_level)

    if args.num_candidates <= 0 or args.k <= 0:
        raise ValueError("num-candidates 与 k 必须为正整数。")
    if args.k > args.num_candidates:
        logger.warning("k 大于候选数，将在候选不足时提前停止。")

    cfg = ExperimentConfig(
        topics=list(args.topics),
        num_candidates=int(args.num_candidates),
        k=int(args.k),
        alpha=float(args.alpha),
        lambda_diversity=float(args.lambda_diversity),
        chunk_size=int(args.chunk_size),
        chunk_overlap=int(args.chunk_overlap),
        m3e_path=str(args.m3e_path),
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        llm_model=str(args.llm_model),
    )

    logger.info("🔧 Experiment Config:\n{}", dataclasses.asdict(cfg))
    runner = AblationRunner(cfg)

    # 1) Prompt 消融
    runner.run_prompt_ablation()
    # 2) 子模目标 × 贪心变体 消融
    runner.run_submodular_ablation()

    # —— 关键收尾：再次打印提示产物目录，提醒查看 recheck 警告 —— #
    logger.success("✅ 全部实验完成。输出与日志位于：{}", args.output_dir)
    logger.info("若见 [WARN]/[INFO] Recheck 提示，请依据日志检查实现与数据。")


if __name__ == "__main__":
    main()
