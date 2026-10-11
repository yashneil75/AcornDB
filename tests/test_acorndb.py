"""Tests that run without downloading the ~500MB GGUF (fake embedder)."""
from __future__ import annotations

import math


class FakeEmbedder:
    dimension = 8

    def embed(self, text: str) -> list[float]:
        v = [float((hash(text) >> (8 * i)) & 0xFF) for i in range(8)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_many(self, texts) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def make_db():  # type: ignore[no-untyped-def]
    from acorndb import AcornDB, QdrantConfig

    return AcornDB(qdrant=QdrantConfig(in_memory=True), embedder=FakeEmbedder())  # type: ignore[arg-type]


def test_databases_crud():
    db = make_db()
    assert db.list_databases() == []
    info = db.create_database("t1")
    assert info.name == "t1" and info.vector_size == 8
    assert db.database_exists("t1")
    try:
        db.create_database("t1")
        raise AssertionError("should have raised")
    except ValueError:
        pass
    assert len(db.list_databases()) == 1
    assert db.delete_database("t1") is True
    assert db.delete_database("t1") is False


def test_embed_and_search():
    db = make_db()
    vec = db.embed("hello")
    assert len(vec) == 8
    assert abs(sum(x * x for x in vec) - 1.0) < 1e-6
    db.add("notes", ["hello world", "totally unrelated xyz"], ids=[1, 2])
    hits = db.search("notes", "hello world", limit=2)
    assert len(hits) == 2
    assert hits[0].text == "hello world"
    assert hits[0].id == 1
    assert hits[0].score >= hits[1].score


def test_config_validation():
    from acorndb import EmbedConfig

    try:
        EmbedConfig(truncate_dim=100)
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_backend_param_validation():
    from acorndb import EmbedConfig, EmbeddingGemmaEmbedder

    try:
        EmbeddingGemmaEmbedder(EmbedConfig(), backend="bogus")
        raise AssertionError("should have raised")
    except ValueError:
        pass
    e = EmbeddingGemmaEmbedder(EmbedConfig(), backend="auto")
    assert e.backend_used is None
    e.close()  # no server started -> no-op


def test_parse_embedding_response_shapes():
    from acorndb import parse_embedding_response

    old = {"embedding": [0.1, 0.2, 0.3]}
    assert parse_embedding_response(old) == [0.1, 0.2, 0.3]
    new = [{"index": 0, "embedding": [[0.1, 0.2, 0.3]]}]
    assert parse_embedding_response(new) == [0.1, 0.2, 0.3]
    try:
        parse_embedding_response({"nope": 1})
        raise AssertionError("should have raised")
    except ValueError:
        pass
