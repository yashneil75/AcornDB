# Models — `DatabaseInfo`, `Document`, `SearchHit`, aliases

Source: `acorndb/models.py`. Re-exported from `acorndb/__init__.py`.

## Type aliases

```python
Distance = Literal["Cosine", "Euclid", "Dot"]
PointId = Union[int, str]
```

| Alias | Meaning |
|---|---|
| `Distance` | Qdrant distance metric names. Literal so editors offer autocomplete. Mapped to `qdrant_client.models.Distance` (`COSINE`/`EUCLID`/`DOT`) inside `store.py::_to_distance`. |
| `PointId` | Point ids accepted by Qdrant: `int` or `str`. Returned by `add` / `add_vectors`, carried on `Document.id` and `SearchHit.id`. |

## `DatabaseInfo`

Metadata about one Qdrant collection (one "database"). Frozen + slots
dataclass.

```python
@dataclass(frozen=True, slots=True)
class DatabaseInfo:
    name: str
    vector_size: int
    distance: str
    points_count: int | None = None
```

| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | Collection name. |
| `vector_size` | `int` | Vector width (embedder dim, or explicit `vector_size=`). |
| `distance` | `str` | Metric name, e.g. `"Cosine"`. |
| `points_count` | `int \| None` | Present from `list_databases` (Qdrant's count); `None` on the `create_database` return. |

```python
info = db.create_database("notes", overwrite=True)
# DatabaseInfo(name='notes', vector_size=768, distance='Cosine', points_count=None)

for info in db.list_databases():
    print(info.name, info.vector_size, info.distance, info.points_count)
```

## `Document`

A single text record ready to be embedded + stored. Slots dataclass (mutable).

```python
@dataclass(slots=True)
class Document:
    text: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: PointId | None = None
    vector: list[float] | None = None
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `text` | `str` | required | Stored under payload key `"text"`; this is what gets embedded by `add`. |
| `payload` | `dict[str, Any]` | `{}` | Extra payload merged as `{**{"text": text}, **payload}` on upsert. |
| `id` | `PointId \| None` | `None` | Point id. `None` → generated `uuid4` string in `add`. |
| `vector` | `list[float] \| None` | `None` | Reserved field; `AcornDB.add` currently re-embeds `text` and does not read it. Use `add_vectors` for pre-computed vectors. |

Usage rule (enforced in `AcornDB._coerce_documents`): when passing
`list[Document]` to `add`, do **not** also pass `payloads=` / `ids=` —
they live inside each `Document`.

```python
from acorndb import Document

db.add("notes", [
    Document(text="hello world", payload={"tag": "greet"}, id=1),
    Document(text="qdrant is fast"),
])
```

## `SearchHit`

One typed search result. Frozen + slots dataclass.

```python
@dataclass(frozen=True, slots=True)
class SearchHit:
    id: PointId
    score: float
    text: str | None
    payload: dict[str, Any]
```

| Field | Type | Meaning |
|---|---|---|
| `id` | `PointId` | Qdrant point id. |
| `score` | `float` | Qdrant relevance score (higher = closer under Cosine default). |
| `text` | `str \| None` | Value of payload key `"text"`, popped out for convenience. `None` when missing or when `with_payload=False`. Non-string values are coerced with `str()`. |
| `payload` | `dict[str, Any]` | Remaining payload after `text` is removed. Empty when `with_payload=False`. |

```python
hits = db.search("notes", "hello", limit=2)
for h in hits:
    print(h.score, h.text, h.payload, h.id)
```

## Payload key convention

`store.py::_TEXT_KEY = "text"`. Both `add` and `add_vectors` write the
human-readable text under `"text"`; `search` pops it back into
`SearchHit.text`. Keep your own payload keys distinct from `"text"` to avoid
collisions.

## See also

- [AcornDB](api-acorndb.md) — `add`, `search`, `list_databases` returns
- [Config](api-config.md) — `Distance` default
