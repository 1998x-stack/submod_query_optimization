from __future__ import annotations

import json

from src.cache import CandidateCache


def test_candidate_cache_round_trip(tmp_path) -> None:
    cache = CandidateCache(str(tmp_path / "cache"))
    metadata = {
        "topic": "rag",
        "mode": "simple",
        "model_name": "model-a",
        "user_prompt": "prompt-v1",
    }

    assert cache.load(metadata) is None
    path = cache.store(metadata, ["query one", "query two"])

    assert path.exists()
    assert cache.load(metadata) == ["query one", "query two"]


def test_candidate_cache_invalidates_when_generation_inputs_change(tmp_path) -> None:
    cache = CandidateCache(str(tmp_path / "cache"))
    metadata = {
        "topic": "rag",
        "mode": "simple",
        "model_name": "model-a",
        "user_prompt": "prompt-v1",
    }
    cache.store(metadata, ["query one"])

    changed = dict(metadata)
    changed["user_prompt"] = "prompt-v2"
    assert cache.load(changed) is None


def test_candidate_cache_ignores_corrupt_payload(tmp_path) -> None:
    cache = CandidateCache(str(tmp_path / "cache"))
    metadata = {
        "topic": "rag",
        "mode": "simple",
        "model_name": "model-a",
        "user_prompt": "prompt-v1",
    }
    path = cache.store(metadata, ["query one"])
    path.write_text("{not-json", encoding="utf-8")

    assert cache.load(metadata) is None


def test_candidate_cache_rejects_empty_strings(tmp_path) -> None:
    cache = CandidateCache(str(tmp_path / "cache"))

    try:
        cache.store({"topic": "x"}, [""])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
