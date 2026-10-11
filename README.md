# AcornDB

Typed Python library combining **Qdrant** (vector DB) with
**EmbeddingGemma-2 GGUF** (`unsloth/embeddinggemma-2-GGUF` /
`embeddinggemma-2-UD-Q4_K_XL.gguf`) via `llama-cpp-python`.

Three typed surfaces:

1. **Databases**: `create_database` / `delete_database` / `list_databases`
2. **Embed**: `embed(text)` / `embed_many(texts)` — plus `embed_image` /
   `embed_video` / `embed_audio`, single-pass `embed_composed` /
   `embed_interleaved`, and `_many` batch variants, all in one unified space
3. **Search**: `search(database, query, limit=...)` — or `search(database,
   image=...)` / `video=` / `audio=` / `composed=` for cross-modal retrieval

## Images, video, audio & interleaved inputs

EmbeddingGemma-2 is natively multimodal: text, images, video and audio share
one 768-dim space, so a text query finds images, clips and recordings, and
combined text+media inputs embed in a single pass. The GGUF runtimes are
text-only, so media goes through `sentence-transformers` with the
full-precision weights (needs `pip install -e ".[multimodal]"`, plus
`[video]` for video *files* — frame lists need no decoder):

```python
db.add_images("pics", ["cat.jpg", "dog.jpg"])   # paths, URLs, bytes, PIL, numpy
db.add_videos("clips", ["clip.mp4", ["f1.jpg", "f2.jpg"]])
db.add_audios("sounds", ["meow.wav"])           # paths, URLs, bytes, waveforms

hits = db.search("pics", "a cat")               # text -> images
hits = db.search("pics", image="q.jpg")         # image -> images
hits = db.search("clips", video="q.mp4")        # video -> videos
hits = db.search("sounds", audio="q.wav")       # audio -> recordings
hits = db.search("pics", composed={"text": "a cat", "image": "q.jpg"})
print(hits[0].modality, hits[0].image, hits[0].video, hits[0].audio)

vec = db.embed_interleaved("a cat <|image|>", ["cat.jpg"])  # text+image, one pass
```

`embed_*` methods (and `search`) take an optional `prompt_name=` (`"query"`,
`"document"`, ...) for task-prompted embeddings. For cross-modal search,
embed a database's texts and media with one backend: pass
`AcornDB(embedder=EmbeddingGemmaMultimodalEmbedder())`. See
[`docs/api-multimodal.md`](docs/api-multimodal.md).

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

- `"cpp"`: in-process `llama-cpp-python`. Fastest, but requires a build
  whose bundled llama.cpp knows the model arch since upstream support did not yet land.
- `"server"`: library-managed localhost `llama-server`. The prebuilt binary
  (first build with embeddinggemma2 support, currently b11457) is
  auto-downloaded to `~/.cache/acorndb/bin/`, started on 127.0.0.1 with an
  ephemeral port, and stopped on `embedder.close()`. No compiler, no setup.
- `"auto"`: tries `"cpp"`, falls back to `"server"` when the installed
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

## Changing the model

The default GGUF is just `EmbedConfig` defaults (`repo_id`, `filename`,
`model_dir`). Point at any compatible GGUF like this:

```python
from acorndb import AcornDB, EmbedConfig, QdrantConfig

embed = EmbedConfig(
    repo_id="unsloth/embeddinggemma-2-GGUF",
    filename="embeddinggemma-2-UD-Q4_K_XL.gguf",
    # model_dir="~/.cache/acorndb/models",  # override cache location
    # truncate_dim=256,  # must be one of 128/256/512/768
)
db = AcornDB(qdrant=QdrantConfig(in_memory=True), embed=embed)
```

Things to know when you swap models:

- `create_database` auto-sizes to `embedder.dimension` (i.e. `truncate_dim`)
  unless you pass `vector_size=` explicitly.
- Vectors from different models are not comparable. Re-embed and re-`add`
  your texts after switching; old points will score meaninglessly.
- If the new model has a different dim, recreate the database
  (`create_database(name, overwrite=True)`) or use a new name — otherwise
  upserts will fail on size mismatch.
- First `embed()` call auto-downloads the new file into `model_dir`.

## API

| Method | Signature | Notes |
|---|---|---|
| `create_database` | `(name, *, vector_size=None, distance="Cosine", overwrite=False) -> DatabaseInfo` | auto-sizes to embedder dim |
| `delete_database` | `(name) -> bool` | True if it existed |
| `list_databases` | `() -> list[DatabaseInfo]` | |
| `database_exists` | `(name) -> bool` | |
| `embed` | `(text: str) -> list[float]` | |
| `embed_many` | `(texts) -> list[list[float]]` | |
| `embed_image` / `embed_images` | `(image, *, prompt_name) -> list[float]` | needs `pip install -e ".[multimodal]"` |
| `embed_video` / `embed_videos` | `(video, *, prompt_name) -> list[float]` | video files also need `.[video]` |
| `embed_audio` / `embed_audios` | `(audio, *, prompt_name) -> list[float]` | |
| `embed_composed` / `embed_interleaved` | `(dict \| text+images, *, prompt_name)` | single-pass text+media |
| `add` | `(database, texts: list[str] \| list[Document], *, payloads, ids) -> list[id]` | embeds + upserts, auto-creates DB |
| `add_images` / `add_videos` / `add_audios` | `(database, media, *, refs, payloads, ids)` | raw inputs or `ImageDocument` / `VideoDocument` / `AudioDocument` |
| `add_vectors` | `(database, vectors, texts, *, payloads, ids)` | skip embedding (pairs with `embed_composed`) |
| `search` | `(database, query=None, *, image=None, video=None, audio=None, composed=None, prompt_name=None, limit, filter, with_payload) -> list[SearchHit]` | exactly one of query/image/video/audio/composed |
| `delete_points` | `(database, ids)` | |

See `examples/quickstart.py` for a runnable demo.

## Docs

Full API reference lives in [`docs/`](docs/index.md) (Markdown, rendered on GitHub).
For local browser viewing, open `docs/index.html` — same content as dependency-free static HTML.

- [Quickstart](docs/quickstart.md) — install, Qdrant setup, minimal demo
- [AcornDB](docs/api-acorndb.md) — databases, `add`, `search`, `embed`
- [Config](docs/api-config.md) — `EmbedConfig`, `QdrantConfig`, `AcornDBConfig`
- [Models](docs/api-models.md) — `DatabaseInfo`, `Document`, `SearchHit`
- [Embedder](docs/api-embedder.md) — backends, truncation, normalization
- [Server](docs/api-server.md) — managed `llama-server`
- [Binaries](docs/api-llama-bin.md) — auto-downloaded prebuilt binaries
- [Guides](docs/guides.md) — backends, swapping models, Qdrant modes, errors
