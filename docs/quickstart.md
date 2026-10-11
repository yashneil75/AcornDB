# Quickstart

## Install

```bash
pip install -e ".[all]"
# CPU-only llama-cpp-python is the default wheel.
# For CUDA:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Optional extras (see `pyproject.toml`):

| Extra | Installs | Needed for |
|---|---|---|
| `embed` | `llama-cpp-python>=0.2` | in-process `"cpp"` backend |
| `hub` | `huggingface-hub>=0.20` | auto-download of the GGUF |
| `all` | both | default recommended path |

## Start Qdrant (optional)

`AcornDB()` defaults to in-memory, so no infra is required. For persistence
or sharing, run Qdrant and point at it:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

```python
from acorndb import AcornDB, QdrantConfig

# Zero-infra (default):
db = AcornDB(qdrant=QdrantConfig(in_memory=True))

# Server mode:
db = AcornDB(qdrant=QdrantConfig(url="http://localhost:6333"))

# Local on-disk mode:
db = AcornDB(qdrant=QdrantConfig(path="./qdrant_data"))
```

See [Qdrant modes](guides.md#qdrant-connection-modes) for the full matrix.

## Minimal end-to-end

```python
from acorndb import AcornDB, QdrantConfig

db = AcornDB(qdrant=QdrantConfig(in_memory=True))

db.create_database("demo", overwrite=True)          # -> DatabaseInfo
ids = db.add("demo", ["hello world", "qdrant is a vector database"])
print("inserted:", ids)

print("embed dim:", len(db.embed("hello")))         # 768 by default

for hit in db.search("demo", "hello", limit=2):
    print(f"{hit.score:.4f} id={hit.id} text={hit.text!r}")

db.delete_database("demo")
```

This is `examples/quickstart.py`. First `embed()` / `add()` call
auto-downloads `embeddinggemma-2-UD-Q4_K_XL.gguf` (~500 MB) into
`~/.cache/acorndb/models/` (see [Embedder](api-embedder.md)).

## Embedder backends (no server ops needed)

```python
from acorndb import EmbedConfig, EmbeddingGemmaEmbedder

emb = EmbeddingGemmaEmbedder(EmbedConfig(), backend="auto")
print(emb.embed("hello world")[:3], emb.backend_used)
emb.close()
```

| `backend=` | Behaviour |
|---|---|
| `"cpp"` | In-process `llama-cpp-python`. Fastest, but requires a wheel whose bundled llama.cpp knows the model arch. |
| `"server"` | Library-managed localhost `llama-server`. Prebuilt binary auto-downloaded to `~/.cache/acorndb/bin/`, ephemeral port on 127.0.0.1, stopped on `close()`. |
| `"auto"` (default) | Tries `"cpp"`, falls back to `"server"` on `ImportError` / `ValueError`. Inspect `embedder.backend_used` to see which won. |

Details: [Embedder](api-embedder.md), [Server](api-server.md),
[Binaries](api-llama-bin.md), [Guides — backends](guides.md#embedding-backends).

## Next steps

- Full method reference: [AcornDB](api-acorndb.md)
- Config knobs: [Config](api-config.md)
- Typed returns: [Models](api-models.md)
- Operational notes (swapping models, errors): [Guides](guides.md)
