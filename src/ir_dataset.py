# -*- coding: utf-8 -*-
"""Offline IR benchmark dataset loader with explicit qrels.

Supported layouts:
1) Native:
   documents.jsonl, queries.jsonl, qrels.tsv
2) BEIR-style:
   corpus.jsonl, queries.jsonl, qrels/test.tsv

JSONL ids may use doc_id/query_id or _id. Corpus text may include title + text.
Qrels accept 3-column (qid, docid, relevance) or 4-column TREC format.
"""

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
    documents: tuple[IRDocument, ...]
    queries: tuple[IRQuery, ...]
    qrels: dict[str, dict[str, int]]

    @classmethod
    def load(cls, root_dir: str) -> "IRDataset":
        root = Path(root_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"IR dataset directory not found: {root}")

        document_path = cls._first_existing(
            root / "documents.jsonl",
            root / "corpus.jsonl",
        )
        query_path = cls._first_existing(root / "queries.jsonl")
        qrels_path = cls._first_existing(
            root / "qrels.tsv",
            root / "qrels" / "test.tsv",
        )

        documents = cls._load_documents(document_path)
        queries = cls._load_queries(query_path)
        qrels = cls._load_qrels(qrels_path)

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
    def _first_existing(*paths: Path) -> Path:
        for path in paths:
            if path.is_file():
                return path
        raise FileNotFoundError(
            "none of the required dataset files exist: "
            + ", ".join(str(path) for path in paths)
        )

    @staticmethod
    def _load_documents(path: Path) -> list[IRDocument]:
        records: list[IRDocument] = []
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

                doc_id = str(obj.get("doc_id") or obj.get("_id") or "").strip()
                title = str(obj.get("title") or "").strip()
                body = str(obj.get("text") or "").strip()
                text = "\n".join(part for part in (title, body) if part)

                if not doc_id or not text:
                    raise ValueError(
                        f"{path}:{line_number}: doc_id/_id and text must be non-empty"
                    )
                if doc_id in seen:
                    raise ValueError(f"{path}:{line_number}: duplicate doc id={doc_id!r}")
                seen.add(doc_id)
                records.append(IRDocument(doc_id, text))

        if not records:
            raise ValueError(f"{path} contains no documents")
        return records

    @staticmethod
    def _load_queries(path: Path) -> list[IRQuery]:
        records: list[IRQuery] = []
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

                query_id = str(obj.get("query_id") or obj.get("_id") or "").strip()
                text = str(obj.get("text") or "").strip()
                if not query_id or not text:
                    raise ValueError(
                        f"{path}:{line_number}: query_id/_id and text must be non-empty"
                    )
                if query_id in seen:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate query id={query_id!r}"
                    )
                seen.add(query_id)
                records.append(IRQuery(query_id, text))

        if not records:
            raise ValueError(f"{path} contains no queries")
        return records

    @staticmethod
    def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
        qrels: dict[str, dict[str, int]] = {}

        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue

                parts = raw.split("\t") if "\t" in raw else raw.split()
                normalized_header = [part.strip().lower() for part in parts]
                if normalized_header in (
                    ["query_id", "doc_id", "relevance"],
                    ["query-id", "corpus-id", "score"],
                ):
                    continue

                if len(parts) == 3:
                    query_id, doc_id, relevance_raw = parts
                elif len(parts) == 4:
                    query_id, _iteration, doc_id, relevance_raw = parts
                else:
                    raise ValueError(
                        f"{path}:{line_number}: expected 3-column qrels or 4-column TREC qrels"
                    )

                query_id = query_id.strip()
                doc_id = doc_id.strip()
                if not query_id or not doc_id:
                    raise ValueError(f"{path}:{line_number}: ids must be non-empty")
                try:
                    relevance = int(relevance_raw.strip())
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
