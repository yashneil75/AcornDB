# Multimodal — images, video, audio & composed inputs (`multimodal.py`)

Source: `acorndb/multimodal.py`. Re-exported from `acorndb/__init__.py`.

EmbeddingGemma-2 (`google/embeddinggemma-2`, 740M params) is natively
multimodal: text (incl. code), images, video and audio share one unified
768-dim space (MRL-truncatable to 512/256/128), so a text query finds images,
clips and recordings — and combined text+media inputs embed in a single pass.

The GGUF path (`EmbeddingGemmaEmbedder` via llama-cpp / the managed
llama-server) is **text-only** — the auto-downloaded server binary ships
without vision/audio projectors. Non-text inputs go through
`sentence-transformers` with the full-precision weights (Google's documented
path). Install it with:

```bash
pip install -e ".[multimodal]"          # ST, torch, torchvision, Pillow, soundfile, librosa
pip install -e ".[multimodal,video]"    # + opencv for video *files*
```

(`video` is only needed for video *files/bytes*; lists of frames need no
decoder. Audio files decode via soundfile, falling back to stdlib `wave`
(PCM WAV) and ffmpeg. `torchvision` serves the image processor and `librosa`
serves audio loading inside sentence-transformers — both are in the extra.
`torchcodec` is deliberately *not* required: video is pre-sampled and audio
pre-normalized locally, so no model-side media decoder is ever needed.
`all` intentionally excludes `multimodal` — torch is ~2 GB.)

```python
from acorndb import AcornDB, QdrantConfig

db = AcornDB(qdrant=QdrantConfig(in_memory=True))  # + pip install acorndb[multimodal]
db.add_images("pics", ["cat.jpg", "dog.jpg"])
db.add_audios("sounds", ["meow.wav"])
hits = db.search("pics", "a cat", limit=2)   # text query -> images
hits = db.search("pics", image="query.jpg")  # image query -> images
hits = db.search("sounds", "a cat meowing")  # text query -> audio
for h in hits:
    print(h.score, h.modality, h.image, h.audio)
```

## Backends

| Class | Text | Image / video / audio / composed | Weights |
|---|---|---|---|
| `EmbeddingGemmaEmbedder` | GGUF (`cpp`/`server`/`auto`) | delegated to a lazily-built `EmbeddingGemmaMultimodalEmbedder` (defaults) | Q4_K_XL text / full-precision media |
| `EmbeddingGemmaMultimodalEmbedder` | sentence-transformers | sentence-transformers | full-precision, all modalities |

For cross-modal search, use one backend consistently per database —
`EmbeddingGemmaMultimodalEmbedder` (pass as `AcornDB(embedder=...)`) is
recommended so text and media come from the same weights.

## `MultimodalConfig`

```python
@dataclass(slots=True)
class MultimodalConfig:
    repo_id: str = "google/embeddinggemma-2"
    model_dir: Path | None = None   # None -> ST's default HF cache
    device: str | None = None       # None -> ST auto ("cpu"/"cuda"/...)
    truncate_dim: int = 768         # one of 128/256/512/768 (MRL)
    normalize: bool = True
    max_frames: int = 16            # evenly-spaced frames sampled per video
```

`truncate_dim` outside 128/256/512/768 and `max_frames < 1` raise
`ValueError`.

## Inputs

```python
ImageInput = str | Path | bytes | bytearray | file-like | PIL.Image | np.ndarray
VideoInput = str | Path | bytes | bytearray | file-like | list[ImageInput]
AudioInput = str | Path | bytes | bytearray | file-like | np.ndarray | (np.ndarray, sample_rate)
```

`str` is a local file path, or an `http(s)://` URL (downloaded). `bytes`
are encoded file contents. numpy images are HWC (`uint8` preferred; float
`0-1` accepted; grayscale expanded to RGB). Audio normalizes to 16 kHz mono
float32 (`AUDIO_SAMPLE_RATE`); bare arrays are assumed 16 kHz unless given as
an `(array, sample_rate)` tuple.

### `load_image(inp: ImageInput) -> PIL.Image`

Load anything into an RGB PIL image. Raises `ImportError` without Pillow,
`ValueError` on missing/undecodable input.

### `load_audio(inp: AudioInput, sample_rate=None) -> np.ndarray`

Load anything into mono float32 @ 16 kHz (resampled as needed). Files decode
via soundfile → stdlib `wave` → ffmpeg. Raises `ValueError` on
missing/undecodable input.

### `sample_video_frames(video: VideoInput, max_frames=16) -> list[PIL.Image]`

Evenly-spaced RGB frames — the same sampling `embed_video` applies to every
input before its single native encode (no model-side decoder needed). Frame
lists need no decoder; files/URLs/bytes decode via OpenCV, falling back to
imageio (`ValueError` on empty/undecodable, `ImportError` with install hint
when neither decoder is present).

### `media_ref(inp, kind, explicit=None) -> str`

Payload ref: explicit wins; paths/URLs reuse their string; anything else
gets `image-<hex>` / `video-<hex>` / `audio-<hex>`.

## `EmbeddingGemmaMultimodalEmbedder`

```python
emb = EmbeddingGemmaMultimodalEmbedder(MultimodalConfig(max_frames=8))
vec = emb.embed("hello")              # text, len truncate_dim
vec = emb.embed_image("cat.jpg")      # image
vec = emb.embed_video("clip.mp4")     # pre-sampled frames, one joint encode
vec = emb.embed_audio("clip.wav")     # 16 kHz mono
vec = emb.embed_interleaved("a cat <|image|>", ["cat.jpg"])  # joint, one pass
```

Every `embed_*` method takes an optional keyword-only `prompt_name`
(`"query"`, `"document"`, `"Classification"`, ... — see the model card);
prompts are never applied unless requested.

| Method | Notes |
|---|---|
| `embed(text)` / `embed_many(texts, batch_size=32)` | Same contract as the GGUF embedder; `ValueError` on empty. |
| `embed_image(image)` / `embed_images(images, batch_size=32)` | `[]` → `[]` without touching the model. |
| `embed_video(video)` / `embed_videos(videos, batch_size=32)` | Pre-sampled to `max_frames` frames, then one native joint encode. |
| `embed_audio(audio)` / `embed_audios(audios, batch_size=32)` | Every input normalized to 16 kHz mono and staged as a temp WAV (cleaned up after encode). |
| `embed_composed(item)` / `embed_composed_many(items, batch_size=32)` | Any `{"text"?, "image"?, "audio"?, "video"?}` dict → one vector. |
| `embed_interleaved(text, images)` | `embed_composed({"text", "image"})` shortcut; `<|image|>` placeholders supported. |
| `embed_query` / `embed_documents` | sentence-transformers aliases. |
| `dimension` | `config.truncate_dim`. |
| `close()` | Releases the model handle (no server involved). |
| `backend_used` | `None` until first encode, then `"st"`. |

Missing `sentence-transformers` raises `ImportError` with the
`pip install acorndb[multimodal]` hint. First use downloads
`google/embeddinggemma-2` from HF Hub.

## Composed inputs (`ComposedInput`)

```python
class ComposedInput(TypedDict, total=False):
    text: str
    image: ImageInput | list[ImageInput]
    audio: AudioInput | list[AudioInput]
    video: VideoInput                      # one video per dict; many -> many dicts
```

Without `<|image|>` / `<|video|>` / `<|audio|>` placeholders in `text`,
modalities encode in dict-key order; with placeholders they interleave at the
exact positions (counts must match the media items). No-text combos work too:

```python
emb.embed_composed({"text": "a cat", "image": "cat.jpg"})
emb.embed_composed({"image": photo, "audio": "meow.wav"})
emb.embed_composed_many([
    "plain text",
    {"image": "cat.jpg"},
    {"text": "a cat", "image": "cat.jpg"},
    {"audio": "meow.wav"},
])  # heterogeneous batches encode in one pass
```

Validation (`ValueError`, before the model loads): non-dict/empty input,
unknown keys, empty text, empty media lists, missing files.

## Store surface (`store.py`)

`AcornDB(embedder=...)` accepts either embedder class (or any duck-typed
object with the same methods; missing media methods raise `RuntimeError`
with an install hint).

| Method | Notes |
|---|---|
| `embed_image(image)` / `embed_images(...)` | Passthrough (`prompt_name=` supported). |
| `embed_video(video)` / `embed_videos(...)` | Passthrough (`prompt_name=` supported). |
| `embed_audio(audio)` / `embed_audios(...)` | Passthrough (`prompt_name=` supported). |
| `embed_composed(item)` / `embed_composed_many(...)` / `embed_interleaved(text, images)` | Passthrough. |
| `add_images(db, images \| list[ImageDocument], *, refs, payloads, ids)` | Auto-creates DB; payload `{"modality": "image", "image": ref, ...}`. |
| `add_videos(db, videos \| list[VideoDocument], *, refs, payloads, ids)` | Payload `{"modality": "video", "video": ref, ...}`. |
| `add_audios(db, audios \| list[AudioDocument], *, refs, payloads, ids)` | Payload `{"modality": "audio", "audio": ref, ...}`. |
| `search(db, query=None, *, image=None, video=None, audio=None, composed=None, prompt_name=None, ...)` | Exactly one of `query`/`image`/`video`/`audio`/`composed` (`ValueError` otherwise). |

`Document` rules carry over: `ImageDocument`/`VideoDocument`/`AudioDocument`
input forbids parallel `refs=`/`payloads=`/`ids=`; raw inputs require
matching lengths. To store a composed text+media record, embed it with
`embed_composed` and upsert via `add_vectors` with the caption as `texts=`.

`SearchHit` gains `modality: "text" | "image" | "video" | "audio"` (default
`"text"`, so old hits are unchanged) plus `image` / `video` / `audio` refs
popped from the payload, mirroring `text`.

## See also

- [AcornDB](api-acorndb.md) — `add`, `search`, databases
- [Models](api-models.md) — `ImageDocument`, `VideoDocument`, `AudioDocument`, `SearchHit`
- [Embedder](api-embedder.md) — GGUF text backends
