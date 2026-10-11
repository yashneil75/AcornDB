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
| `Modality` | `"text"` \| `"image"` \| `"video"` \| `"audio"` — which modality a point/hit belongs to. |

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

## `ImageDocument` / `VideoDocument`

Image/video equivalents of `Document` for `add_images` / `add_videos`.
`image` accepts a path/URL, bytes, file-like object, PIL image, or numpy
array; `video` accepts a path/URL, bytes, file-like object, or a list of
frames. `ref` is the payload ref (defaults: paths/URLs reuse their string,
everything else gets a generated id). Same rule as `Document`: when passing
`list[ImageDocument]` / `list[VideoDocument]`, don't also pass `refs=` /
`payloads=` / `ids=`.

```python
@dataclass(slots=True)
class ImageDocument:
    image: ImageInput
    payload: dict[str, Any] = field(default_factory=dict)
    id: PointId | None = None
    ref: str | None = None
    vector: list[float] | None = None   # reserved, like Document.vector

@dataclass(slots=True)
class VideoDocument:
    video: VideoInput
    payload: dict[str, Any] = field(default_factory=dict)
    id: PointId | None = None
    ref: str | None = None
    vector: list[float] | None = None
```

## `AudioDocument`

Audio equivalent for `add_audios`. `audio` accepts a path/URL, bytes,
file-like object, numpy waveform, or `(waveform, sample_rate)` tuple; `ref`
rules match `ImageDocument`.

```python
@dataclass(slots=True)
class AudioDocument:
    audio: AudioInput
    payload: dict[str, Any] = field(default_factory=dict)
    id: PointId | None = None
    ref: str | None = None
    vector: list[float] | None = None   # reserved, like Document.vector
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
    modality: Modality = "text"   # "text" | "image" | "video" | "audio"
    image: str | None = None      # image ref, when modality == "image"
    video: str | None = None      # video ref, when modality == "video"
    audio: str | None = None      # audio ref, when modality == "audio"
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
