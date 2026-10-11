# AcornDB

Typed Python library combining **Qdrant** (vector DB) with
**EmbeddingGemma-2 GGUF** (`unsloth/embeddinggemma-2-GGUF` /
`embeddinggemma-2-UD-Q4_K_XL.gguf`) via `llama-cpp-python`.

Three typed surfaces:

1. **Databases**: `create_database` / `delete_database` / `list_databases`
2. **Embed**: `embed(text)` / `embed_many(texts)`
3. **Search**: `search(database, query, limit=...)`

## Install

```bash
pip install -e ".[all]"
# CPU-only llama-cpp-python is the default wheel.
# For CUDA: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Start Qdrant (or skip, `AcornDB()` defaults to in-memory):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Quickstart

```python
from acorndb import AcornDB, QdrantConfig

db = AcornDB(qdrant=QdrantConfig(url="http://localhost:6333"))
# or: db = AcornDB(qdrant=QdrantConfig(in_memory=True))

db.create_database("notes", overwrite=True)   # typed -> DatabaseInfo
db.add("notes", ["hello world", "qdrant is a vector database"])
print(db.list_databases())                    # typed -> list[DatabaseInfo]

vec = db.embed("hello")                       # typed -> list[float], len 768
hits = db.search("notes", "hello", limit=2)   # typed -> list[SearchHit]
for h in hits:
    print(h.score, h.text, h.payload)
```

Model auto-downloads from Hugging Face on first `embed()` call into
`~/.cache/acorndb/models/embeddinggemma-2-UD-Q4_K_XL.gguf`.

## Embedding backends (no server ops needed)

`EmbeddingGemmaEmbedder(backend=...)` resolves automatically (default `"auto"`):

- `"cpp"` — in-process `llama-cpp-python`. Fastest, but requires a build
  whose bundled llama.cpp knows the model arch since upstream support did not yet land.
- `"server"` — library-managed localhost `llama-server`. The prebuilt binary
  (first build with embeddinggemma2 support, currently b11457) is
  auto-downloaded to `~/.cache/acorndb/bin/`, started on 127.0.0.1 with an
  ephemeral port, and stopped on `embedder.close()`. No compiler, no setup.
- `"auto"` — tries `"cpp"`, falls back to `"server"` when the installed
  wheel can't load the model. Check `embedder.backend_used` to see which won.

```python
from acorndb import EmbedConfig, EmbeddingGemmaEmbedder

emb = EmbeddingGemmaEmbedder(EmbedConfig(), backend="auto")
print(emb.embed("hello world")[:3], emb.backend_used)
emb.close()
```

## Model details

- Repo: `unsloth/embeddinggemma-2-GGUF`, file `embeddinggemma-2-UD-Q4_K_XL.gguf`
- Native dim **768**, MRL-truncatable to 512/256/128 via `EmbedConfig(truncate_dim=256)`
- Context 8192 tokens. Vectors are L2-normalized, distance defaults to Cosine.

## API

| Method | Signature | Notes |
|---|---|---|
| `create_database` | `(name, *, vector_size=None, distance="Cosine", overwrite=False) -> DatabaseInfo` | auto-sizes to embedder dim |
| `delete_database` | `(name) -> bool` | True if it existed |
| `list_databases` | `() -> list[DatabaseInfo]` | |
| `database_exists` | `(name) -> bool` | |
| `embed` | `(text: str) -> list[float]` | |
| `embed_many` | `(texts) -> list[list[float]]` | |
| `add` | `(database, texts: list[str] \| list[Document], *, payloads, ids) -> list[id]` | embeds + upserts, auto-creates DB |
| `add_vectors` | `(database, vectors, texts, *, payloads, ids)` | skip embedding |
| `search` | `(database, query: str \| vector, *, limit, filter, with_payload) -> list[SearchHit]` | text auto-embedded |
| `delete_points` | `(database, ids)` | |

See `examples/quickstart.py` for a runnable demo.
