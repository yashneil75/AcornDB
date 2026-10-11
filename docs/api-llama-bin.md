# Binaries — `ensure_server_binary`, `cache_dir`, `LLAMA_BUILD`

Source: `acorndb/llama_bin.py`. `LLAMA_BUILD` re-exported from
`acorndb/__init__.py`; `cache_dir` / `ensure_server_binary` are imported by
`server_embedder.py`.

Auto-download of prebuilt llama.cpp binaries so downstream users need no
compiler. Pinned to the first nightly containing embeddinggemma2 support.

## Constants

```python
LLAMA_BUILD = "b11457"
```

First nightly whose master includes ggml-org/llama.cpp PR #30054
(embeddinggemma2). Bump forward only. Release URL pattern:

```python
_GH = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}"
```

Platform asset matrix (`_ASSETS: (system, machine) -> (asset, exe)`):

| System | Machine | Asset | Executable |
|---|---|---|---|
| windows | amd64 / x86_64 | `llama-b11457-bin-win-cpu-x64.zip` | `llama-server.exe` |
| windows | arm64 | `llama-b11457-bin-win-cpu-arm64.zip` | `llama-server.exe` |
| linux | x86_64 | `llama-b11457-bin-ubuntu-x64.tar.gz` | `llama-server` |
| linux | aarch64 | `llama-b11457-bin-ubuntu-arm64.tar.gz` | `llama-server` |
| darwin | arm64 | `llama-b11457-bin-macos-arm64.tar.gz` | `llama-server` |
| darwin | x86_64 | `llama-b11457-bin-macos-x64.tar.gz` | `llama-server` |

Windows x64 Vulkan override (`use_vulkan=True`):

```python
_VULKAN_WIN = ("llama-b11457-bin-win-vulkan-x64.zip", "llama-server.exe")
```

```python
from acorndb import LLAMA_BUILD
print(LLAMA_BUILD)  # "b11457"
```

## `cache_dir() -> Path`

```python
def cache_dir() -> Path:
    return Path.home() / ".cache" / "acorndb" / "bin"
```

Root for binaries, downloaded archives, and `llama-server.log`. Created on
demand (`mkdir(parents=True, exist_ok=True)`).

Layout after first download (example, Windows x64 CPU):

```text
~/.cache/acorndb/bin/
  llama-b11457-bin-win-cpu-x64.zip
  llama-b11457-bin-win-cpu-x64/llama-server.exe (+ DLLs)
  llama-server.log
```

## `ensure_server_binary(*, use_vulkan: bool = False) -> Path`

Return a working `llama-server` binary path, downloading + extracting on
first use. Cached: returns immediately if the extracted exe already exists.

```python
from acorndb.llama_bin import ensure_server_binary

exe = ensure_server_binary()                    # CPU binary
exe = ensure_server_binary(use_vulkan=True)     # Windows x64 Vulkan binary
```

Steps:

1. Detect `(platform.system().lower(), platform.machine().lower())`.
   - Windows + `use_vulkan` + amd64/x86_64 → Vulkan asset; otherwise the
     `_ASSETS` matrix.
   - Unknown platform → `RuntimeError("no prebuilt llama.cpp b11457 binary
     for <sys>/<machine>. Install llama.cpp manually ...")`.
2. `dest_dir = cache_dir() / <asset without .tar.gz/.zip>`,
   `exe_path = dest_dir / <exe>`. Return if it exists.
3. Download `https://github.com/.../b11457/<asset>` to
   `cache_dir() / <asset>` if missing (`_download`: streaming copy via
   `urllib`, 256 KiB chunks, 300 s timeout, atomic `mkstemp` + `replace` so
   Windows file locking is safe; partial file removed on failure).
4. Extract (`.zip` → `zipfile`, else `tarfile` with `filter="data"` on
   Python ≥ 3.11.4, plain `extractall` fallback).
5. If the expected path still doesn't exist (asset layout varies), recursive
   `rglob(exe)` fallback; else `RuntimeError("<exe> not found inside
   <asset>")`.
6. `chmod 0o755` on non-Windows. Return the exe path.

## `use_vulkan` plumbing

`EmbeddingGemmaEmbedder(backend="server", use_vulkan=...)` →
`ManagedLlamaServer(use_vulkan=...)` → `ensure_server_binary(use_vulkan=...)`.
Only affects Windows x64; all other platforms ignore it (matrix lookup).

## See also

- [Server](api-server.md) — how the binary is launched and supervised
- [Embedder](api-embedder.md) — when the server backend is chosen
- [Guides](guides.md) — backend troubleshooting, log location
