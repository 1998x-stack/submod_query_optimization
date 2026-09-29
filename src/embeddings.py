# -*- coding: utf-8 -*-
"""LangChain-based embedding wrapper for a local sentence-transformer model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except ImportError as exc:  # pragma: no cover - dependency boundary
    raise RuntimeError(
        "langchain-community is required; install project dependencies first"
    ) from exc

from src.config import EmbeddingConfig


@dataclass
class LangChainM3EEmbedder:
    """Expose a small numpy-oriented embedding interface."""

    cfg: EmbeddingConfig
    _dimension: int | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        logger.info("Loading local embedding model from: {}", self.cfg.model_path)
        self._emb = SentenceTransformerEmbeddings(
            model_name=self.cfg.model_path,
            cache_folder=None,
            model_kwargs={"device": self.cfg.device} if self.cfg.device else {},
            encode_kwargs={
                "batch_size": self.cfg.batch_size,
                "normalize_embeddings": False,
            },
        )
        logger.info(
            "Embedding model ready (batch_size={}, device={})",
            self.cfg.batch_size,
            self.cfg.device,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts as a float32 (n, d) matrix."""
        if not texts:
            dim = self._dimension or 0
            return np.empty((0, dim), dtype=np.float32)

        vecs = self._emb.embed_documents(texts)
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise RuntimeError(
                f"embedding backend returned invalid shape {arr.shape} for {len(texts)} texts"
            )
        if not np.isfinite(arr).all():
            raise RuntimeError("embedding backend returned non-finite values")

        if self._dimension is None:
            self._dimension = int(arr.shape[1])
        elif arr.shape[1] != self._dimension:
            raise RuntimeError(
                f"embedding dimension changed from {self._dimension} to {arr.shape[1]}"
            )
        return arr
