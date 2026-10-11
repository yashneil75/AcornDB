# Config — `EmbedConfig`, `QdrantConfig`, `AcornDBConfig`

Source: `acorndb/config.py`. Re-exported from `acorndb/__init__.py`.

## Constants

```python
DEFAULT_REPO_ID = "unsloth/embeddinggemma-2-GGUF"
DEFAULT_FILENAME = "embeddinggemma-2-UD-Q4_K_XL.gguf"
ALLOWED_DIMS = (128, 256, 512, 768)
```

These are the defaults baked into `EmbedConfig`. Import them when you need to
reference (not hard-code) the pinned model:

```python
from acorndb import DEFAULT_FILENAME, DEFAULT_REPO_ID
```

## `EmbedConfig`

How to load the GGUF embedding model. Slots dataclass.

```python
@dataclass(slots=True)
class EmbedConfig:
    repo_id: str = DEFAULT_REPO_ID
    filename: str = DEFAULT_FILENAME
    model_dir: Path = ~/.cache/acorndb/models
    n_ctx: int = 8192
    n_batch: int = 512
    n_threads: int | None = None
    n_gpu_layers: int = -1
    embedding: bool = True
    normalize: bool = True
    truncate_dim: int = 768
    verbose: bool = False
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `repo_id` | `str` | `"unsloth/embeddinggemma-2-GGUF"` | Hugging Face repo for auto-download. |
| `filename` | `str` | `"embeddinggemma-2-UD-Q4_K_XL.gguf"` | GGUF file within the repo. |
| `model_dir` | `Path` | `~/.cache/acorndb/models` | Cache dir. `str` is coerced to `Path` in `__post_init__`. |
| `n_ctx` | `int` | `8192` | Context window passed to `llama_cpp.Llama`. |
| `n_batch` | `int` | `512` | Batch size passed to `llama_cpp.Llama`. |
| `n_threads` | `int \| None` | `None` | `None` → llama.cpp auto. Only forwarded when not `None`. |
| `n_gpu_layers` | `int` | `-1` | `-1` = offload all possible layers. |
| `embedding` | `bool` | `True` | Forwarded as `embedding=True` to `Llama`. |
| `normalize` | `bool` | `True` | L2-normalize output vectors (post-truncation). |
| `truncate_dim` | `int` | `768` | MRL truncation. **Must be one of `128/256/512/768`**, else `ValueError`. Native dim is 768. |
| `verbose` | `bool` | `False` | Verbose llama.cpp logging. |

### `model_path -> Path`

```python
@property
def model_path(self) -> Path:
    return self.model_dir / self.filename
```

Local GGUF path. `ensure_model_downloaded()` (see
[Embedder](api-embedder.md)) downloads into exactly this location.

### Custom model example

```python
from acorndb import AcornDB, EmbedConfig, QdrantConfig

embed = EmbedConfig(
    repo_id="unsloth/embeddinggemma-2-GGUF",
    filename="embeddinggemma-2-UD-Q4_K_XL.gguf",
    truncate_dim=256,   # must be 128/256/512/768
)
db = AcornDB(qdrant=QdrantConfig(in_memory=True), embed=embed)
```

After switching models: re-embed + re-`add` (vectors from different models
are not comparable), and recreate the database if the dim changed
(`create_database(name, overwrite=True)` or a new name). See
[Guides — swapping models](guides.md#swapping-models).

## `QdrantConfig`

How to connect to Qdrant. Pick **at most one** of `url` / `path` /
`in_memory`. Slots dataclass.

```python
@dataclass(slots=True)
class QdrantConfig:
    url: str | None = None
    api_key: str | None = None
    path: str | None = None
    in_memory: bool = False
    prefer_grpc: bool = False
    timeout: float = 60.0
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `url` | `str \| None` | `None` | e.g. `"http://localhost:6333"`. Server mode. |
| `api_key` | `str \| None` | `None` | Passed as `api_key=` with `url=`. |
| `path` | `str \| None` | `None` | e.g. `"./qdrant_data"`. Local on-disk mode. Mutually exclusive with `url`. |
| `in_memory` | `bool` | `False` | `QdrantClient(":memory:")`. Explicit opt-in; the store also defaults here when nothing is set. |
| `prefer_grpc` | `bool` | `False` | Forwarded to `QdrantClient` in `url` mode. |
| `timeout` | `float` | `60.0` | Forwarded to `QdrantClient` in `url` mode. |

`__post_init__` raises `ValueError("set only one of url / path / in_memory")`
if more than one is chosen. Zero chosen is valid → in-memory default in
`AcornDB._get_client`.

```python
QdrantConfig(url="http://localhost:6333")  # server
QdrantConfig(path="./qdrant_data")          # on-disk
QdrantConfig(in_memory=True)                # ephemeral
QdrantConfig()                              # also in-memory via store default
```

## `AcornDBConfig`

Top-level bundle. Slots dataclass.

```python
@dataclass(slots=True)
class AcornDBConfig:
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    default_distance: Distance = "Cosine"
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `embed` | `EmbedConfig` | `EmbedConfig()` | Model settings. |
| `qdrant` | `QdrantConfig` | `QdrantConfig()` | Connection settings. |
| `default_distance` | `Distance` | `"Cosine"` | Fallback for `create_database` when `distance=` is omitted. One of `"Cosine"`, `"Euclid"`, `"Dot"`. |

```python
from acorndb import AcornDB, AcornDBConfig, EmbedConfig, QdrantConfig

cfg = AcornDBConfig(
    embed=EmbedConfig(truncate_dim=256),
    qdrant=QdrantConfig(url="http://localhost:6333"),
    default_distance="Cosine",
)
db = AcornDB(cfg)
# Equivalent shortcut:
db = AcornDB(qdrant=QdrantConfig(url="http://localhost:6333"),
             embed=EmbedConfig(truncate_dim=256))
```

## See also

- [AcornDB](api-acorndb.md) — how configs are consumed
- [Models](api-models.md) — `Distance` literal
- [Guides](guides.md) — Qdrant modes, swapping models
