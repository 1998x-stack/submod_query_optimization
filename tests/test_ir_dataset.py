from __future__ import annotations

import json

import pytest

from src.ir_dataset import IRDataset


def _write_dataset(tmp_path, qrels: str = "q1\td1\t2\n") -> str:
    root = tmp_path / "ir"
    root.mkdir()
    (root / "documents.jsonl").write_text(
        json.dumps({"doc_id": "d1", "text": "alpha"}) + "\n"
        + json.dumps({"doc_id": "d2", "text": "beta"}) + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        json.dumps({"query_id": "q1", "text": "alpha query"}) + "\n",
        encoding="utf-8",
    )
    (root / "qrels.tsv").write_text(qrels, encoding="utf-8")
    return str(root)


def test_dataset_loads_documents_queries_and_qrels(tmp_path) -> None:
    dataset = IRDataset.load(_write_dataset(tmp_path))
    assert [doc.doc_id for doc in dataset.documents] == ["d1", "d2"]
    assert dataset.queries[0].query_id == "q1"
    assert dataset.qrels["q1"]["d1"] == 2


def test_beir_style_layout_is_supported(tmp_path) -> None:
    root = tmp_path / "beir"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "Title", "text": "Body"}) + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "query"}) + "\n",
        encoding="utf-8",
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
        encoding="utf-8",
    )

    dataset = IRDataset.load(str(root))
    assert dataset.documents[0].doc_id == "d1"
    assert dataset.documents[0].text == "Title\nBody"
    assert dataset.qrels["q1"]["d1"] == 1


def test_trec_four_column_qrels_are_supported(tmp_path) -> None:
    dataset = IRDataset.load(_write_dataset(tmp_path, "q1 0 d1 2\n"))
    assert dataset.qrels["q1"]["d1"] == 2


def test_unknown_document_in_qrels_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown doc"):
        IRDataset.load(_write_dataset(tmp_path, "q1\tmissing\t1\n"))


def test_query_without_positive_qrel_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive qrel"):
        IRDataset.load(_write_dataset(tmp_path, "q1\td1\t0\n"))
