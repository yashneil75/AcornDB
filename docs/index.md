# AcornDB Documentation

Typed Python library combining **Qdrant** (vector DB) with
**EmbeddingGemma-2 GGUF** (`unsloth/embeddinggemma-2-GGUF` /
`embeddinggemma-2-UD-Q4_K_XL.gguf`) via `llama-cpp-python`.

Three typed surfaces:

1. **Databases** — create / delete / list collections
2. **Embed** — text → L2-normalized vectors (dim 768, MRL-truncatable)
3. **Search** — text or vector → ranked `SearchHit` list

## Contents

| Page | Covers |
|---|---|
| [Quickstart](quickstart.md) | Install, Qdrant setup, 10-line runnable demo |
| [AcornDB (`store.py`)](api-acorndb.md) | `AcornDB` — databases, `add`, `add_vectors`, `search`, `embed` |
| [Config (`config.py`)](api-config.md) | `EmbedConfig`, `QdrantConfig`, `AcornDBConfig`, defaults |
| [Models (`models.py`)](api-models.md) | `DatabaseInfo`, `Document`, `SearchHit`, `Distance`, `PointId` |
| [Embedder (`embedder.py`)](api-embedder.md) | `EmbeddingGemmaEmbedder` — backends, truncation, normalization |
| [Server (`server_embedder.py`)](api-server.md) | `ManagedLlamaServer`, `parse_embedding_response` |
| [Binaries (`llama_bin.py`)](api-llama-bin.md) | `ensure_server_binary`, `cache_dir`, `LLAMA_BUILD` |
| [Guides](guides.md) | Backends, swapping models, Qdrant modes, error handling |

## 30-second example

```python
from acorndb import AcornDB, QdrantConfig

db = AcornDB(qdrant=QdrantConfig(in_memory=True))
db.create_database("notes", overwrite=True)
db.add("notes", ["hello world", "qdrant is a vector database"])
hits = db.search("notes", "hello", limit=2)
for h in hits:
    print(h.score, h.text, h.payload)
```

Source of truth is `acorndb/` (`__init__.py`, `config.py`, `models.py`,
`store.py`, `embedder.py`, `server_embedder.py`, `llama_bin.py`).
Runnable demo: `examples/quickstart.py`. Offline tests: `tests/test_acorndb.py`.
