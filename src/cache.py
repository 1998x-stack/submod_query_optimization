# -*- coding: utf-8 -*-
"""Disk-backed cache for expensive LLM candidate generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger


CACHE_SCHEMA_VERSION = 1


class CandidateCache:
    """Content-addressed JSON cache keyed by generation inputs."""

    def __init__(self, root_dir: str) -> None:
        if not root_dir.strip():
            raise ValueError("root_dir must not be empty")
        self.root_dir = Path(root_dir)

    @staticmethod
    def _canonical_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            **metadata,
        }

    def _cache_path(self, metadata: dict[str, Any]) -> Path:
        canonical = self._canonical_metadata(metadata)
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return self.root_dir / f"{digest}.json"

    def load(self, metadata: dict[str, Any]) -> list[str] | None:
        """Return cached candidates when metadata and payload are valid."""
        path = self._cache_path(metadata)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable candidate cache entry: {}", path)
            return None

        expected = self._canonical_metadata(metadata)
        if payload.get("metadata") != expected:
            logger.warning("Ignoring candidate cache entry with mismatched metadata: {}", path)
            return None

        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not all(
            isinstance(item, str) and item.strip() for item in candidates
        ):
            logger.warning("Ignoring invalid candidate cache payload: {}", path)
            return None

        return [item.strip() for item in candidates]

    def store(self, metadata: dict[str, Any], candidates: list[str]) -> Path:
        """Atomically persist candidates and return the cache path."""
        if not all(isinstance(item, str) and item.strip() for item in candidates):
            raise ValueError("candidates must contain only non-empty strings")

        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(metadata)
        tmp = path.with_suffix(".json.tmp")
        payload = {
            "metadata": self._canonical_metadata(metadata),
            "candidates": [item.strip() for item in candidates],
        }
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)
        return path
