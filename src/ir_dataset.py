# -*- coding: utf-8 -*-
"""Offline IR benchmark dataset loader with explicit qrels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class IRDocument:
    doc_id: str
    text: str


@dataclass(frozen=True, slots=True)
class IRQuery:
    query_id: str
    text: str


@dataclass(frozen=True, slots=True)
class IRDataset:
    """Documents, benchmark queries, and graded relevance judgments."""

    documents: tuple[IRDocument, ...]
    queries: tuple[IRQuery, ...]
    qrels: dict[str, dict[str, int]]

    @classmethod
    def load(cls, root_dir: str) -> "IRDataset":
        root = Path(root_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"IR dataset directory not found: {root}")

        documents = cls._load_jsonl(
            root / "documents.jsonl", id_key="doc_id", record_type=IRDocument
        )
        queries = cls._load_jsonl(
            root / "queries.jsonl", id_key="query_id", record_type=IRQuery
        )
        qrels = cls._load_qrels(root / "qrels.tsv")

        doc_ids = {doc.doc_id for doc in documents}
        query_ids = {query.query_id for query in queries}

        unknown_queries = set(qrels) - query_ids
        if unknown_queries:
            raise ValueError(f"qrels reference unknown query ids: {sorted(unknown_queries)}")

        unknown_docs = {
            doc_id
            for judgments in qrels.values()
            for doc_id in judgments
            if doc_id not in doc_ids
        }
        if unknown_docs:
            raise ValueError(f"qrels reference unknown doc ids: {sorted(unknown_docs)}")

        missing_positive = [
            query.query_id
            for query in queries
            if not any(rel > 0 for rel in qrels.get(query.query_id, {}).values())
        ]
        if missing_positive:
            raise ValueError(
                "every benchmark query must have at least one positive qrel; "
                f"missing={missing_positive}"
            )

        return cls(tuple(documents), tuple(queries), qrels)

    @staticmethod
    def _load_jsonl(path: Path, id_key: str, record_type):
        if not path.is_file():
            raise FileNotFoundError(f"required dataset file not found: {path}")

        records = []
        seen: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

                identifier = str(obj.get(id_key, "")).strip()
                text = str(obj.get("text", "")).strip()
                if not identifier or not text:
                    raise ValueError(
                        f"{path}:{line_number}: {id_key} and text must be non-empty"
                    )
                if identifier in seen:
                    raise ValueError(f"{path}:{line_number}: duplicate {id_key}={identifier!r}")
                seen.add(identifier)
                records.append(record_type(identifier, text))

        if not records:
            raise ValueError(f"{path} contains no records")
        return records

    @staticmethod
    def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
        if not path.is_file():
            raise FileNotFoundError(f"required dataset file not found: {path}")

        qrels: dict[str, dict[str, int]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                parts = raw.split("\t")
                if parts == ["query_id", "doc_id", "relevance"]:
                    continue
                if len(parts) != 3:
                    raise ValueError(
                        f"{path}:{line_number}: expected query_id<TAB>doc_id<TAB>relevance"
                    )
                query_id, doc_id, relevance_raw = (part.strip() for part in parts)
                if not query_id or not doc_id:
                    raise ValueError(f"{path}:{line_number}: ids must be non-empty")
                try:
                    relevance = int(relevance_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: relevance must be an integer"
                    ) from exc
                if relevance < 0:
                    raise ValueError(f"{path}:{line_number}: relevance must be non-negative")
                per_query = qrels.setdefault(query_id, {})
                if doc_id in per_query:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate qrel ({query_id}, {doc_id})"
                    )
                per_query[doc_id] = relevance

        if not qrels:
            raise ValueError(f"{path} contains no qrels")
        return qrels
