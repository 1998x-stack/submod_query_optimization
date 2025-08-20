# -*- coding: utf-8 -*-
"""Corpus loader: read txt files, split with LangChain, and embed."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from loguru import logger

# Text splitters（v0.2 独立包），同时兼容旧路径
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from src.embeddings import LangChainM3EEmbedder


class CorpusLoader:
    """加载 data/ 目录中的 txt 文件，做 LangChain 分块，并生成向量。"""

    def __init__(self, data_dir: str, embedder: LangChainM3EEmbedder, chunk_size: int, chunk_overlap: int) -> None:
        self.data_dir = Path(data_dir)
        self.embedder = embedder
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def _split_text(self, text: str) -> list[str]:
        """用递归字符分割器切块，兼顾中英文与符号边界。"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "，", ",", " "],
        )
        return splitter.split_text(text)

    def load(self, max_files: int | None = None, max_chars_per_file: int = 40000) -> tuple[list[str], np.ndarray]:
        """读取 corpus 并编码。

        Args:
            max_files: 最多读取的文件数，None 表示全部。
            max_chars_per_file: 单文件最大截断长度，避免超大文件拖慢进程。

        Returns:
            (chunks_texts, embeddings)；若无文件则返回空列表与空矩阵。
        """
        if not self.data_dir.exists():
            logger.warning("⚠️ 语料目录不存在：{}", self.data_dir)
            return [], np.zeros((0, 768), dtype=np.float32)

        files = sorted([p for p in self.data_dir.glob("**/*.txt") if p.is_file()])
        if max_files is not None:
            files = files[:max_files]

        all_chunks: list[str] = []
        for p in files:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                logger.exception("读取文件失败: {}", p)
                continue
            content = content.strip()[:max_chars_per_file]
            if not content:
                continue
            chunks = self._split_text(content)
            all_chunks.extend(chunks)

        if not all_chunks:
            return [], np.zeros((0, 768), dtype=np.float32)

        logger.info("📚 已分块：{} chunks, 正在向量化…", len(all_chunks))
        emb = self.embedder.encode(all_chunks)
        logger.info("✅ 语料向量完成：shape={}", emb.shape)
        return all_chunks, emb
