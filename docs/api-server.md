# Server — `ManagedLlamaServer`, `parse_embedding_response`

Source: `acorndb/server_embedder.py`. Both names re-exported from
`acorndb/__init__.py`.

Managed localhost `llama-server`: the invisible embedding backend used
automatically when in-process `llama-cpp-python` cannot load the model (e.g.
its bundled llama.cpp predates the model architecture). The binary is
auto-downloaded (see [Binaries](api-llama-bin.md)), started on 127.0.0.1 with
an ephemeral port, and torn down on `close()`.

## `parse_embedding_response(data: object) -> list[float]`

Handle both server response shapes:

- Old: `{"embedding": [...]}` (flat vector, or `[ [ ... ] ]` nested once)
- New (b11457+): `[{"index": 0, "embedding": [[...]]}]`

```python
from acorndb import parse_embedding_response

parse_embedding_response({"embedding": [0.1, 0.2, 0.3]})
# [0.1, 0.2, 0.3]

parse_embedding_response([{"index": 0, "embedding": [[0.1, 0.2, 0.3]]}])
# [0.1, 0.2, 0.3]
```

Raises `ValueError` on any other shape (message truncated to 120 chars).

## `ManagedLlamaServer`

A `llama-server` subprocess owned by the library. Start via `start()`.

```python
ManagedLlamaServer(
    model_path: Path | str,
    *,
    use_vulkan: bool = False,
    startup_timeout: float = 180.0,
)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `model_path` | `Path \| str` | required | GGUF file (usually from `EmbedConfig.model_path` / `ensure_model_downloaded`). Stored as `Path`. |
| `use_vulkan` | `bool` | `False` | Select the Windows Vulkan binary via `ensure_server_binary`. |
| `startup_timeout` | `float` | `180.0` | Seconds to wait for `/health` → 200 before `TimeoutError`. |

Attributes: `port: int | None`, `base_url: str | None`
(`http://127.0.0.1:<port>` after `start`), `log_path =
~/.cache/acorndb/bin/llama-server.log` (via `cache_dir()`).

### `start() -> str`

Idempotent: returns existing `base_url` if already started.

1. Resolves the binary via `ensure_server_binary(use_vulkan=...)`.
2. Picks a free 127.0.0.1 port (`_free_port`), sets `base_url`.
3. Spawns `[binary, "-m", model, "--embeddings", "--port", port]` with
   stdout+stderr appended to `log_path`.
4. Polls `GET base_url/health` every 0.5 s until 200 or `startup_timeout`.
   - Process exit during boot → `RuntimeError("llama-server exited during startup; see <log>")`.
   - Timeout → `close()` + `TimeoutError("llama-server did not start in time (...)")`.

Returns `base_url` on success.

### `embed(text: str) -> list[float]`

`POST base_url/embedding` with body `{"content": text}`, 300 s timeout,
parsed via `parse_embedding_response`.

Raises `RuntimeError("server not started; call start() first")` if `start()`
wasn't called.

Note: this returns the **raw** server vector. Truncation + L2 normalization
happen one layer up in `EmbeddingGemmaEmbedder._post_process`.

### `embed_many(texts: Sequence[str], batch_size: int = 32) -> list[list[float]]`

Sequential `[self.embed(t) for t in texts]`. `batch_size` is accepted for
signature compatibility with the `cpp` backend but does not change behaviour.

### `close() -> None`

Terminate the subprocess if running (`terminate()` → `wait(15)` →
`kill()` fallback), clear `_proc`. Safe to call repeatedly; also invoked from
`__exit__` and best-effort `__del__`.

### Context-manager support

```python
from acorndb import ManagedLlamaServer

with ManagedLlamaServer("/path/to/model.gguf") as server:
    vec = server.embed("hello world")
# closed automatically
```

## Wiring

`EmbeddingGemmaEmbedder._get_server()` (see [Embedder](api-embedder.md)) is
the normal caller — you rarely construct this class directly. Direct use is
for debugging the server path or embedding without the truncation/normalize
layer.

## Log file

`~/.cache/acorndb/bin/llama-server.log` — check here first when `start()`
raises `RuntimeError`. The message includes the exact path.

## See also

- [Embedder](api-embedder.md) — backend selection, post-processing
- [Binaries](api-llama-bin.md) — `ensure_server_binary`, `LLAMA_BUILD`
- [Guides](guides.md) — backend troubleshooting
