"""Runnable demo: databases -> embed -> search."""
from __future__ import annotations

from easydb import EasyDB, EmbedConfig, QdrantConfig


def main() -> None:
    db = EasyDB(
        qdrant=QdrantConfig(in_memory=True),
        embed=EmbedConfig(truncate_dim=768),
    )
    print("databases before:", db.list_databases())

    db.create_database("demo", overwrite=True)
    print("databases after create:", db.list_databases())

    # Fake embedder path is exercised in tests; here we use the real model.
    # First call downloads embeddinggemma-2-UD-Q4_K_XL.gguf (~500MB).
    ids = db.add("demo", ["hello world", "qdrant is a vector database"])
    print("inserted:", ids)
    print("single embed dim:", len(db.embed("hello")))

    for hit in db.search("demo", "hello", limit=2):
        print(f"{hit.score:.4f} id={hit.id} text={hit.text!r}")

    db.delete_database("demo")
    print("databases at end:", db.list_databases())


if __name__ == "__main__":
    main()
