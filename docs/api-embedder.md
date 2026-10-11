# Embedder — `EmbeddingGemmaEmbedder`

Source: `acorndb/embedder.py`. Re-exported from `acorndb/__init__.py`.

Typed wrapper around the EmbeddingGemma-2 GGUF file
(`unsloth/embeddinggemma-2-GGUF` / `embeddinggemma-2-UD-Q4_K_XL.gguf`,
native dim 768, MRL-truncatable to 512/256/128, context 8192).

Backend is automatic: in-process `llama-cpp-python` when it can load the
model, otherwise a library-managed localhost `llama-server` (see
[Server](api-server.md)). Forced via `backend=`.

```python
from acorndb import EmbedConfig, EmbeddingGemmaEmbedder

embedder = EmbeddingGemmaEmbedder(EmbedConfig(truncate_dim=768))
vec = embedder.embed("hello world")   # len 768, L2-normalized
```

## Constructor

```python
EmbeddingGemmaEmbedder(
    config: EmbedConfig | None = None,
    *,
    backend: str = "auto",
    use_vulkan: bool = False,
)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `config` | `EmbedConfig \| None` | `None` → `EmbedConfig()` | Model path, context, truncation, normalization (see [Config](api-config.md)). |
| `backend` | `str` | `"auto"` | `"auto"` \| `"cpp"` \| `"server"`. Anything else raises `ValueError`. |
| `use_vulkan` | `bool` | `False` | Forwarded to the managed server path (`ensure_server_binary(use_vulkan=...)`); selects the Windows Vulkan binary. |

Attributes set at construction: `config`, `backend` (requested mode),
`use_vulkan`, `_llm = None`, `_server = None`, `backend_used = None`
(`"cpp"` / `"server"` once resolved — inspect after first `embed`).

### Backend resolution (`_get_backend`)

| `backend=` | Behaviour |
|---|---|
| `"cpp"` | `backend_used = "cpp"`; constructs `llama_cpp.Llama` (raises `ImportError` with install hint if missing). |
| `"server"` | `backend_used = "server"`; downloads model if needed, starts `ManagedLlamaServer`. |
| `"auto"` | Tries in-process first; on `ImportError` / `ValueError` (wheel can't load the arch) clears `_llm`, sets `backend_used = "server"`, starts the managed server. |

## Properties

### `dimension -> int`

Effective vector size after MRL truncation: `config.truncate_dim`.

### `model_path -> Path`

`Path(config.model_path)` — local GGUF location
(`model_dir / filename`, default `~/.cache/acorndb/models/<filename>`).

## Setup

### `ensure_model_downloaded() -> Path`

Download the GGUF from Hugging Face Hub if missing; return the local path.

- Returns immediately if `model_path` exists.
- Otherwise `hf_hub_download(repo_id, filename, local_dir=model_dir)` (creates
  parents), then normalizes to exactly `model_path` (some hub versions nest
  differently — renames if resolved paths differ).
- Raises `ImportError` with install hint (`pip install acorndb[hub]` or manual
  download path) if `huggingface-hub` is missing.

### `close() -> None`

Shut down the managed server if one was started; no-op otherwise. Safe to
call unconditionally (also called from `AcornDB` teardown paths and tests).

## Core embedding

All outputs go through `_post_process`: truncate to `truncate_dim`, then
L2-normalize if `config.normalize` (via `_l2_normalize`; zero vectors pass
through unchanged).

### `embed(text: str) -> list[float]`

Embed one string → unit-norm vector of length `dimension`.

Raises `ValueError` on empty string.

### `embed_many(texts: Sequence[str], batch_size: int = 32) -> list[list[float]]`

Embed many strings (looped one-at-a-time per batch on either backend —
`llama_cpp` and the `/embedding` endpoint both take single inputs here).

- Empty input → `[]` (no backend touched).
- Any empty string in the batch → `ValueError`.
- `batch_size` controls chunking only, not server parallelism.

### Aliases (sentence-transformers conventions)

```python
embed_query(text: str) -> list[float]          # == embed(text)
embed_documents(texts: Sequence[str]) -> list[list[float]]  # == embed_many(texts)
```

## Multimodal (`embed_image` / `embed_video` / `embed_audio` / `embed_composed` / `embed_interleaved` + `_many` variants)

EmbeddingGemma-2 is multimodal but the GGUF runtimes are text-only (no
vision/audio projectors ship with the prebuilt server binary), so these
methods delegate to a lazily-built `EmbeddingGemmaMultimodalEmbedder` with
default settings — same unified 768-dim space as the GGUF text vectors. They
work on any `backend=` setting and need `pip install acorndb[multimodal]`
(plus `acorndb[video]` for video *files*). All take an optional
keyword-only `prompt_name`. See [Multimodal](api-multimodal.md).

## In-process backend details (`_get_llm`)

Lazily constructs `llama_cpp.Llama` with:

```python
Llama(
    model_path=...,
    embedding=config.embedding,      # True
    n_ctx=config.n_ctx,              # 8192
    n_batch=config.n_batch,          # 512
    n_gpu_layers=config.n_gpu_layers,  # -1
    verbose=config.verbose,          # False
    n_threads=...                    # only if config.n_threads is not None
)
```

Then `backend.embed(text)` is called on the resolved backend object
(`Llama` exposes `.embed`; `ManagedLlamaServer` exposes `.embed` with the
same signature, which is why `_get_backend` is interchangeable).

## Server backend details (`_get_server`)

Ensures the model file, constructs
`ManagedLlamaServer(model_file, use_vulkan=...)`, calls `.start()`, caches in
`_server`. See [Server](api-server.md) and [Binaries](api-llama-bin.md).

## Example: check which backend won

```python
emb = EmbeddingGemmaEmbedder(EmbedConfig(), backend="auto")
vec = emb.embed("hello")
print(emb.backend_used)  # "cpp" or "server"
emb.close()
```

## Error summary

| Situation | Exception |
|---|---|
| `backend="bogus"` | `ValueError` at construction |
| `embed("")` / empty member of `embed_many` | `ValueError` |
| `llama-cpp-python` missing (`"cpp"` path) | `ImportError` with `pip install acorndb[embed]` hint |
| `huggingface-hub` missing + model absent | `ImportError` with `pip install acorndb[hub]` hint |
| `truncate_dim` not in 128/256/512/768 | `ValueError` (from `EmbedConfig`) |
| Managed server fails to boot | `RuntimeError` / `TimeoutError` (from `ManagedLlamaServer`) |

## See also

- [Config](api-config.md) — every `EmbedConfig` knob
- [Server](api-server.md) — managed `llama-server` lifecycle + response parsing
- [Binaries](api-llama-bin.md) — where the server binary comes from
- [AcornDB](api-acorndb.md) — `embed` / `embed_many` passthrough, `dimension`
