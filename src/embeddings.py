# -*- coding: utf-8 -*-
"""LangChain-based embeddings wrapper for local m3e.

本模块使用 LangChain 的 `SentenceTransformerEmbeddings`，
从本地目录加载 m3e 模型，并提供统一的 encode 接口。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

# LangChain Embeddings（0.2 之后位于 community 包）
try:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "未找到 langchain_community.embeddings，请安装：pip install langchain-community"
    ) from exc

from src.config import EmbeddingConfig


@dataclass
class LangChainM3EEmbedder:
    """基于 LangChain 的 m3e 向量封装。"""

    cfg: EmbeddingConfig

    def __post_init__(self) -> None:
        # LangChain 的 SentenceTransformerEmbeddings 支持本地路径
        logger.info("🔧 Loading local m3e model from: {}", self.cfg.model_path)
        self._emb = SentenceTransformerEmbeddings(
            model_name=self.cfg.model_path,
            cache_folder=None,
            model_kwargs={"device": self.cfg.device} if self.cfg.device else {},
            encode_kwargs={"batch_size": self.cfg.batch_size, "normalize_embeddings": False},
        )
        logger.info("✅ m3e is ready (batch_size={}, device={})", self.cfg.batch_size, self.cfg.device)

    def encode(self, texts: list[str]) -> np.ndarray:
        """编码一批文本为向量（float32）。

        Args:
            texts: 文本列表。

        Returns:
            (n,d) numpy 数组。
        """
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        vecs = self._emb.embed_documents(texts)  # List[List[float]]
        arr = np.asarray(vecs, dtype=np.float32)
        return arr
