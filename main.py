# -*- coding: utf-8 -*-
"""CLI entry for ablations, proxy benchmarks, and qrels-backed IR benchmarks."""

from __future__ import annotations

import argparse
import dataclasses

from loguru import logger

from src.config import ExperimentConfig
from src.ir_benchmark import IRBenchmarkRunner
from src.logging_utils import setup_logging
from src.runner import AblationRunner


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc
    if not values:
        raise argparse.ArgumentTypeError("grid must not be empty")
    return values


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submodular Query Optimization — reproducible query-selection benchmarks"
    )
    parser.add_argument("--topics", nargs="*", default=["embeddings and rerankers"])
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda-diversity", type=float, default=0.5)
    parser.add_argument("--mmr-lambda", type=float, default=0.6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--llm-model", type=str, default="gpt-4o")
    parser.add_argument("--m3e-path", type=str, default="model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--candidate-cache-dir", type=str, default=".cache/candidates")
    parser.add_argument("--no-candidate-cache", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--log-level", type=str, default="INFO")

    parser.add_argument(
        "--ir-dataset-dir",
        type=str,
        default=None,
        help="directory containing documents.jsonl, queries.jsonl, qrels.tsv",
    )
    parser.add_argument(
        "--ir-only",
        action="store_true",
        help="run only the qrels-backed IR benchmark",
    )
    parser.add_argument("--ir-cutoffs", type=_csv_ints, default=(5, 10, 20))
    parser.add_argument("--ir-alpha-grid", type=_csv_floats, default=(0.5,))
    parser.add_argument("--ir-lambda-grid", type=_csv_floats, default=(0.5,))
    parser.add_argument("--ir-mmr-grid", type=_csv_floats, default=(0.6,))
    parser.add_argument("--ir-random-repeats", type=int, default=5)
    parser.add_argument("--ir-bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.output_dir, level=args.log_level)

    if args.ir_only and not args.ir_dataset_dir:
        raise ValueError("--ir-only requires --ir-dataset-dir")

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
        ir_dataset_dir=args.ir_dataset_dir,
        ir_cutoffs=tuple(args.ir_cutoffs),
        ir_alpha_grid=tuple(args.ir_alpha_grid),
        ir_lambda_grid=tuple(args.ir_lambda_grid),
        ir_mmr_grid=tuple(args.ir_mmr_grid),
        ir_random_repeats=int(args.ir_random_repeats),
        ir_bootstrap_iterations=int(args.ir_bootstrap_iterations),
    )

    if cfg.k > cfg.num_candidates:
        logger.warning("k exceeds num_candidates; selection budgets will be capped")

    logger.info("Experiment Config:\n{}", dataclasses.asdict(cfg))
    runner = AblationRunner(cfg)

    if not args.ir_only:
        runner.run_prompt_ablation()
        runner.run_submodular_ablation()
        runner.run_baseline_benchmark()

    if cfg.ir_dataset_dir:
        IRBenchmarkRunner(
            cfg=cfg,
            embedder=runner.embedder,
            candidate_provider=runner.get_candidates,
        ).run()

    logger.success("experiments complete; outputs are under {}", args.output_dir)


if __name__ == "__main__":
    main()
