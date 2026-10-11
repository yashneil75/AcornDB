# `AcornDB` — main entry point

Source: `acorndb/store.py`. Re-exported from `acorndb/__init__.py`.

One typed object for databases, embedding, and search:

```python
from acorndb import AcornDB, QdrantConfig

db = AcornDB(qdrant=QdrantConfig(url="http://localhost:6333"))
db.create_database("notes")
db.add("notes", ["hello world", "qdrant is fast"])
hits = db.search("notes", "hello")
print(hits[0].text, hits[0].score)
```

## Constructor

```python
AcornDB(
    config: AcornDBConfig | None = None,
    *,
    qdrant: QdrantConfig | None = None,
    embed: EmbedConfig | None = None,
    embedder: EmbeddingGemmaEmbedder | None = None,
)
```

| Parameter | Type | Meaning |
|---|---|---|
| `config` | `AcornDBConfig \| None` | Pre-built bundle. Mutually exclusive with `qdrant=` / `embed=`. |
| `qdrant` | `QdrantConfig \| None` | Connection settings. Defaults to `QdrantConfig()` (in-memory client). |
| `embed` | `EmbedConfig \| None` | Model settings. Defaults to `EmbedConfig()` (EmbeddingGemma-2 defaults). |
| `embedder` | `EmbeddingGemmaEmbedder \| None` | Inject a ready embedder (used by tests to avoid the ~500 MB download). Overrides `embed=`. |

Raises `ValueError` if `config=` is combined with `qdrant=` / `embed=`.

`self.config: AcornDBConfig` stores the resolved config. The Qdrant client is
created lazily on first use (`QdrantClient(":memory:")` when nothing is
configured, so `AcornDB()` works with zero infra).

## Properties

### `embedder -> EmbeddingGemmaEmbedder`

The underlying embedder (passed in or constructed from `config.embed`).

### `dimension -> int`

Effective vector size: `self._embedder.dimension`, i.e.
`EmbedConfig.truncate_dim` (one of 128/256/512/768). `create_database`
auto-sizes to this unless `vector_size=` is given.

## Embedding passthrough

### `embed(text: str) -> list[float]`

Embed one string → unit-norm vector of length `dimension`.

Raises `ValueError` on empty string (from the embedder).

### `embed_many(texts: Sequence[str]) -> list[list[float]]`

Embed many strings. Returns `[]` for empty input.

## Databases

Qdrant collections are exposed as "databases".

### `create_database(name, *, vector_size=None, distance=None, overwrite=False) -> DatabaseInfo`

```python
db.create_database("notes", overwrite=True)
db.create_database("notes-euclid", vector_size=256, distance="Euclid")
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `name` | `str` | required | Collection name. Must be non-empty. |
| `vector_size` | `int \| None` | `None` → `self.dimension` | Vector width. Pass explicitly only when bypassing the embedder (see `add_vectors`). |
| `distance` | `Distance \| None` | `None` → `config.default_distance` (`"Cosine"`) | One of `"Cosine"`, `"Euclid"`, `"Dot"`. |
| `overwrite` | `bool` | `False` | If `True` and the collection exists, delete it first. |

Raises `ValueError` if `name` is empty, or if the collection exists and
`overwrite=False`.

Returns `DatabaseInfo(name, vector_size, distance)` (no `points_count` on
create — see `list_databases`).

### `delete_database(name: str) -> bool`

Delete a collection. Returns `True` if it existed, `False` otherwise.

### `list_databases() -> list[DatabaseInfo]`

List all collections with vector params + point counts. Returns `[]` when
empty. `distance` is normalized from Qdrant's enum (e.g. `Distance.COSINE` →
`"Cosine"`).

### `database_exists(name: str) -> bool`

Thin wrapper over `QdrantClient.collection_exists`.

## Write

### `add(database, texts, *, payloads=None, ids=None) -> list[PointId]`

Embed `texts` and upsert into `database`. **Auto-creates** the database with
the embedder's dimension on first use.

```python
# Plain strings:
ids = db.add("notes", ["hello world", "qdrant is fast"])

# With parallel payloads / ids (lengths must match texts):
ids = db.add("notes", ["a", "b"], payloads=[{"tag": "x"}, {}], ids=[1, 2])

# Typed documents (payloads/ids live inside Document, not alongside):
from acorndb import Document
ids = db.add("notes", [Document(text="hi", payload={"tag": "greet"}, id=7)])
```

| Parameter | Type | Meaning |
|---|---|---|
| `database` | `str` | Target collection (auto-created if missing). |
| `texts` | `Sequence[str] \| Sequence[Document]` | Either all `str` or all `Document`. Mixed/empty raises `ValueError`. |
| `payloads` | `Sequence[dict] \| None` | Only with `str` input; length must match. |
| `ids` | `Sequence[PointId] \| None` | Only with `str` input; length must match. Missing ids become `uuid4` strings. |

Storage detail: each point payload is `{"text": <text>, **payload}`.
Returns the point ids used (supplied or generated).

Raises `ValueError` if `texts` is empty, if `Document` input is combined with
`payloads=`/`ids=`, or on length mismatch.

### `add_vectors(database, vectors, texts=None, *, payloads=None, ids=None) -> list[PointId]`

Upsert pre-computed vectors, skipping the embedder. Useful for migration or
custom embedding pipelines.

```python
db.add_vectors("notes", [[0.1, 0.2, ...]], texts=["hello"], ids=["custom-1"])
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `vectors` | `Sequence[Sequence[float]]` | required | One vector per point. Empty → returns `[]` without touching Qdrant. |
| `texts` | `Sequence[str] \| None` | `None` → `["", ...]` | Stored under payload key `"text"` (empty strings are omitted from payload). Length must equal `len(vectors)`. |
| `payloads` | `Sequence[dict] \| None` | `None` → `[{}, ...]` | Extra payload per point. |
| `ids` | `Sequence[PointId] \| None` | `None` → `uuid4` strings | Point ids. |

If the collection is missing it is auto-created with
`vector_size=len(vectors[0])`. Raises `ValueError` on length mismatch.

### `delete_points(database: str, ids: Sequence[PointId]) -> None`

Delete points by id via `models.PointIdsList`. No-op if ids don't exist
(Qdrant semantics).

## Search

### `search(database, query, *, limit=10, filter=None, with_payload=True) -> list[SearchHit]`

Semantic search by text (auto-embedded) or raw vector.

```python
hits = db.search("notes", "hello", limit=2)          # text query
hits = db.search("notes", [0.1, 0.2, ...], limit=5)  # vector query

from qdrant_client import models
hits = db.search(
    "notes",
    "hello",
    limit=5,
    filter=models.Filter(must=[models.FieldCondition(
        key="tag", match=models.MatchValue(value="x"))]),
)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `query` | `str \| Sequence[float]` | required | `str` → embedded via `embedder.embed`; vector → used as-is. |
| `limit` | `int` | `10` | Max hits. |
| `filter` | `qdrant_client.models.Filter \| None` | `None` | Server-side payload filter. |
| `with_payload` | `bool` | `True` | If `False`, hits carry empty payloads and `text=None`. |

Each hit: `SearchHit(id, score, text, payload)` where `text` is popped from
payload key `"text"` (remaining keys stay in `payload`). Ordered by Qdrant
score descending.

## Error summary

| Situation | Exception |
|---|---|
| `config=` + `qdrant=`/`embed=` | `ValueError` |
| `create_database("")` | `ValueError` |
| `create_database` on existing name without `overwrite=True` | `ValueError` |
| `add` with empty `texts` | `ValueError` |
| `add` mixing `Document` with `payloads=`/`ids=` | `ValueError` |
| Length mismatch (`texts`/`payloads`/`ids`, or `vectors`/`texts`) | `ValueError` |
| `embed("")` | `ValueError` (from embedder) |
| `qdrant-client` not installed | `ImportError` with install hint |

## See also

- [Config](api-config.md) — `AcornDBConfig`, `QdrantConfig`, `EmbedConfig`
- [Models](api-models.md) — `DatabaseInfo`, `Document`, `SearchHit`
- [Embedder](api-embedder.md) — embedding behaviour behind `embed`/`add`
- [Guides](guides.md) — Qdrant modes, swapping models
