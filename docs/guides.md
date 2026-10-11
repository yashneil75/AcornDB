# Guides

## Embedding backends

`EmbeddingGemmaEmbedder(backend=...)` (default `"auto"`) resolves on first
`embed()` — see [Embedder](api-embedder.md). Check the winner:

```python
emb = EmbeddingGemmaEmbedder(EmbedConfig(), backend="auto")
emb.embed("probe")
print(emb.backend_used)  # "cpp" or "server"
```

| Symptom | Likely cause | Fix |
|---|---|---|
| `backend_used == "server"` unexpectedly | Installed `llama-cpp-python` wheel bundles a llama.cpp older than embeddinggemma2 support | Expected on `"auto"`; either accept the server path or upgrade/rebuild the wheel (CUDA: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`). |
| `llama-server exited during startup` | Bad model path, port conflict, missing binary | Read the path in the error: `~/.cache/acorndb/bin/llama-server.log`. Re-run with `EmbedConfig(verbose=True)` context; verify the GGUF exists at `EmbedConfig.model_path`. |
| `TimeoutError: llama-server did not start` | Slow first boot (model load) or hung process | Raise `ManagedLlamaServer(startup_timeout=...)` via a hand-built server, or retry; default 180 s is generous but cold disks can exceed it. |
| `no prebuilt llama.cpp ... binary` | Unsupported OS/arch | Install llama.cpp manually (out of scope for auto-download). |
| Vulkan GPU ignored | `use_vulkan=False` or non-Windows-x64 | Pass `EmbeddingGemmaEmbedder(config, backend="server", use_vulkan=True)` on Windows x64. |

Always `emb.close()` (or use `ManagedLlamaServer` as a context manager) to
stop the subprocess. The log appends across runs at
`~/.cache/acorndb/bin/llama-server.log`.

## Swapping models

The default GGUF is just `EmbedConfig` defaults. Point anywhere compatible:

```python
from acorndb import AcornDB, EmbedConfig, QdrantConfig

embed = EmbedConfig(
    repo_id="unsloth/embeddinggemma-2-GGUF",
    filename="embeddinggemma-2-UD-Q4_K_XL.gguf",
    # model_dir="~/.cache/acorndb/models",  # override cache location
    # truncate_dim=256,  # one of 128/256/512/768
)
db = AcornDB(qdrant=QdrantConfig(in_memory=True), embed=embed)
```

Rules (enforced or implied by `store.py`):

1. `create_database` auto-sizes to `embedder.dimension` (`truncate_dim`)
   unless `vector_size=` is explicit. A dim change therefore requires
   `create_database(name, overwrite=True)` or a new collection name —
   otherwise upserts fail on size mismatch.
2. Vectors from different models are **not comparable**. Re-embed and re-`add`
   after switching; old points score meaninglessly.
3. First `embed()` auto-downloads the new file into `model_dir`.
4. `truncate_dim` outside 128/256/512/768 raises `ValueError` at config
   construction — before any download.

MRL note: 768 is native; 512/256/128 are prefixes of the same vector, so
truncating a single model is safe, but switching model files is not.

## Qdrant connection modes

Configured via [QdrantConfig](api-config.md); client created lazily.

| Mode | Config | Durability | Use for |
|---|---|---|---|
| In-memory (default) | `QdrantConfig(in_memory=True)` or `QdrantConfig()` | None — dies with the process | Tests, demos, `examples/quickstart.py` |
| Server | `QdrantConfig(url="http://localhost:6333")` (+ `api_key=`, `prefer_grpc=`, `timeout=60.0`) | Persistent (server-side) | Docker / self-hosted / cloud |
| On-disk | `QdrantConfig(path="./qdrant_data")` | Persistent (local dir) | Single-node persistence without Docker |

Exactly one of `url` / `path` / `in_memory` — more raises `ValueError`.
`AcornDB()` with no args behaves as in-memory.

Filtered search passes `filter=` straight to
`client.query_points(query_filter=...)`; build it with
`qdrant_client.models.Filter` (see [AcornDB search](api-acorndb.md#search)).

## Payloads and ids

- `add` stores `{"text": <text>, **payload}` per point; `search` returns
  `text` separately from `payload` (see [Models](api-models.md#payload-key-convention)).
- Don't use `"text"` as your own payload key.
- Missing ids → `uuid4` strings. Supply `ids=` (or `Document.id`) for stable,
  idempotent upserts and easy `delete_points`.
- `add_vectors` with `texts=["", ...]` omits the `"text"` key for empty
  entries; `SearchHit.text` is then `None`.

## Error handling cheat-sheet

```python
try:
    db.create_database("notes")
except ValueError as e:
    # already exists (without overwrite=True) / empty name
    print("database:", e)

try:
    vec = db.embed("")
except ValueError as e:
    print("embed:", e)  # empty string
```

| Operation | Raises | When |
|---|---|---|
| `AcornDB(config, qdrant=...)` | `ValueError` | Both styles passed |
| `QdrantConfig(url=..., path=...)` | `ValueError` | >1 connection mode |
| `EmbedConfig(truncate_dim=100)` | `ValueError` | Not in 128/256/512/768 |
| `EmbeddingGemmaEmbedder(backend="x")` | `ValueError` | Unknown backend |
| `create_database` existing | `ValueError` | Without `overwrite=True` |
| `add` empty / mixed / mismatched | `ValueError` | See [AcornDB](api-acorndb.md#error-summary) |
| missing `qdrant-client` | `ImportError` | Install hint included |
| missing `llama-cpp-python` / `huggingface-hub` | `ImportError` | Install hint included (`[embed]` / `[hub]` / `[all]`) |

## Testing without the model

`tests/test_acorndb.py` injects a fake embedder — the pattern to copy when
you don't want the ~500 MB download:

```python
from acorndb import AcornDB, QdrantConfig

class FakeEmbedder:
    dimension = 8
    def embed(self, text): ...
    def embed_many(self, texts): ...

db = AcornDB(qdrant=QdrantConfig(in_memory=True), embedder=FakeEmbedder())
```

Run: `pytest` (`testpaths = ["tests"]` in `pyproject.toml`).
