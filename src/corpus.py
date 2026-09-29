# -*- coding: utf-8 -*-
"""Corpus loading, chunking, and embedding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from src.embeddings import LangChainM3EEmbedder


class CorpusLoader:
    """Load text files, split them into chunks, and compute embeddings."""

    def __init__(
        self,
        data_dir: str,
        embedder: LangChainM3EEmbedder,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

        self.data_dir = Path(data_dir)
        self.embedder = embedder
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def _split_text(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "，", ",", " "],
        )
        return [chunk for chunk in splitter.split_text(text) if chunk.strip()]

    def load(
        self,
        max_files: int | None = None,
        max_chars_per_file: int = 40000,
    ) -> tuple[list[str], np.ndarray]:
        """Read corpus files and return chunk texts plus their embeddings."""
        if max_files is not None and max_files < 0:
            raise ValueError("max_files must be non-negative or None")
        if max_chars_per_file <= 0:
            raise ValueError("max_chars_per_file must be positive")

        if not self.data_dir.exists():
            logger.warning("Corpus directory does not exist: {}", self.data_dir)
            return [], np.empty((0, 0), dtype=np.float32)
        if not self.data_dir.is_dir():
            raise NotADirectoryError(str(self.data_dir))

        files = sorted(path for path in self.data_dir.rglob("*.txt") if path.is_file())
        if max_files is not None:
            files = files[:max_files]

        all_chunks: list[str] = []
        for path in files:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                logger.exception("Failed to read corpus file: {}", path)
                continue

            content = content.strip()[:max_chars_per_file]
            if content:
                all_chunks.extend(self._split_text(content))

        if not all_chunks:
            logger.warning("No usable corpus chunks found in {}", self.data_dir)
            return [], np.empty((0, 0), dtype=np.float32)

        logger.info("Embedding {} corpus chunks", len(all_chunks))
        embeddings = self.embedder.encode(all_chunks)
        logger.info("Corpus embeddings ready: shape={}", embeddings.shape)
        return all_chunks, embeddings
