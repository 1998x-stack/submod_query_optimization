# -*- coding: utf-8 -*-
"""Logging utilities using loguru.

本模块统一初始化 loguru，支持控制台 + 文件双路输出。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger


def setup_logging(output_dir: str, level: str = "INFO") -> None:
    """Initialize loguru logging.

    Args:
        output_dir: 日志文件输出目录。
        level: 日志等级，如 "INFO"/"DEBUG".
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()  # 移除默认 handler，避免重复输出
    # 控制台
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )
    # 文件
    logger.add(
        str(Path(output_dir) / "run.log"),
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        level=level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        encoding="utf-8",
    )
    logger.info("🚀 Logger initialized (level={})", level)
