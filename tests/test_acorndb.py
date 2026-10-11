"""Tests that run without downloading the ~500MB GGUF (fake embedder)."""
from __future__ import annotations

import io
import math

import pytest


class FakeEmbedder:
    dimension = 8

    def embed(self, text: str) -> list[float]:
        v = [float((hash(text) >> (8 * i)) & 0xFF) for i in range(8)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_many(self, texts) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _stable_key(obj) -> bytes:
    if isinstance(obj, str):
        return obj.encode()
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if hasattr(obj, "tobytes"):  # PIL image / numpy array
        return bytes(obj.tobytes())
    if isinstance(obj, (list, tuple)):
        return b"|".join(_stable_key(o) for o in obj)
    return repr(obj).encode()


class FakeMultimodalEmbedder(FakeEmbedder):
    """Fake joint text+image+video+audio space: stable hash of content bytes."""

    def _vec(self, key: bytes) -> list[float]:
        import hashlib

        d = hashlib.sha256(key).digest()
        v = [float(b) for b in d[:8]]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_image(self, image, prompt_name=None) -> list[float]:
        return self._vec(b"img:" + _stable_key(image))

    def embed_images(self, images, batch_size=32, prompt_name=None) -> list[list[float]]:
        return [self.embed_image(i) for i in images]

    def embed_video(self, video, prompt_name=None) -> list[float]:
        return self._vec(b"vid:" + _stable_key(video))

    def embed_videos(self, videos, batch_size=32, prompt_name=None) -> list[list[float]]:
        return [self.embed_video(v) for v in videos]

    def embed_audio(self, audio, prompt_name=None) -> list[float]:
        return self._vec(b"aud:" + _stable_key(audio))

    def embed_audios(self, audios, batch_size=32, prompt_name=None) -> list[list[float]]:
        return [self.embed_audio(a) for a in audios]

    def _composed_key(self, item) -> bytes:
        import json

        def norm(v):
            if isinstance(v, dict):
                return {k: norm(v[k]) for k in sorted(v)}
            if isinstance(v, (list, tuple)):
                return [norm(o) for o in v]
            if isinstance(v, (bytes, bytearray)):
                return {"bytes": bytes(v).hex()}
            if hasattr(v, "tobytes"):
                return {"bytes": bytes(v.tobytes()).hex()}
            if hasattr(v, "read"):
                return {"repr": repr(v)}
            return v

        return json.dumps(norm(item), sort_keys=True, default=str).encode()

    def embed_composed(self, item, prompt_name=None) -> list[float]:
        if not isinstance(item, dict) or not item:
            raise ValueError("composed inputs must be non-empty dicts")
        return self._vec(b"mix:" + self._composed_key(item))

    def embed_composed_many(self, items, batch_size=32, prompt_name=None) -> list[list[float]]:
        return [self.embed_composed(i) for i in items]

    def embed_interleaved(self, text, images, prompt_name=None) -> list[float]:
        if not text:
            raise ValueError("text must be a non-empty string")
        seq = list(images) if isinstance(images, (list, tuple)) else [images]
        if not seq:
            raise ValueError("images must be non-empty")
        return self.embed_composed({"text": text, "image": seq})


def make_db():  # type: ignore[no-untyped-def]
    from acorndb import AcornDB, QdrantConfig

    return AcornDB(qdrant=QdrantConfig(in_memory=True), embedder=FakeEmbedder())  # type: ignore[arg-type]


def make_media_db():  # type: ignore[no-untyped-def]
    from acorndb import AcornDB, QdrantConfig

    return AcornDB(qdrant=QdrantConfig(in_memory=True), embedder=FakeMultimodalEmbedder())  # type: ignore[arg-type]


def test_databases_crud():
    db = make_db()
    assert db.list_databases() == []
    info = db.create_database("t1")
    assert info.name == "t1" and info.vector_size == 8
    assert db.database_exists("t1")
    try:
        db.create_database("t1")
        raise AssertionError("should have raised")
    except ValueError:
        pass
    assert len(db.list_databases()) == 1
    assert db.delete_database("t1") is True
    assert db.delete_database("t1") is False


def test_embed_and_search():
    db = make_db()
    vec = db.embed("hello")
    assert len(vec) == 8
    assert abs(sum(x * x for x in vec) - 1.0) < 1e-6
    db.add("notes", ["hello world", "totally unrelated xyz"], ids=[1, 2])
    hits = db.search("notes", "hello world", limit=2)
    assert len(hits) == 2
    assert hits[0].text == "hello world"
    assert hits[0].id == 1
    assert hits[0].score >= hits[1].score


def test_config_validation():
    from acorndb import EmbedConfig

    try:
        EmbedConfig(truncate_dim=100)
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_backend_param_validation():
    from acorndb import EmbedConfig, EmbeddingGemmaEmbedder

    try:
        EmbeddingGemmaEmbedder(EmbedConfig(), backend="bogus")
        raise AssertionError("should have raised")
    except ValueError:
        pass
    e = EmbeddingGemmaEmbedder(EmbedConfig(), backend="auto")
    assert e.backend_used is None
    e.close()  # no server started -> no-op


def test_parse_embedding_response_shapes():
    from acorndb import parse_embedding_response

    old = {"embedding": [0.1, 0.2, 0.3]}
    assert parse_embedding_response(old) == [0.1, 0.2, 0.3]
    new = [{"index": 0, "embedding": [[0.1, 0.2, 0.3]]}]
    assert parse_embedding_response(new) == [0.1, 0.2, 0.3]
    try:
        parse_embedding_response({"nope": 1})
        raise AssertionError("should have raised")
    except ValueError:
        pass


def _red_png() -> bytes:
    pytest.importorskip("PIL")
    from PIL import Image as _I

    buf = io.BytesIO()
    _I.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_embed_image_and_video():
    db = make_media_db()
    img = _red_png()
    vec = db.embed_image(img)
    assert len(vec) == 8
    assert abs(sum(x * x for x in vec) - 1.0) < 1e-6
    assert db.embed_image(img) == vec  # deterministic
    assert db.embed_images([img, img]) == [vec, vec]
    assert db.embed_images([]) == []
    frames = [img, img, img]
    vvec = db.embed_video(frames)
    assert len(vvec) == 8
    assert abs(sum(x * x for x in vvec) - 1.0) < 1e-6
    assert db.embed_videos([frames]) == [vvec]
    assert db.embed_videos([]) == []


def test_add_images_and_search_by_image():
    from acorndb import ImageDocument

    db = make_media_db()
    red, blue = _red_png(), _red_png() + b"x"
    ids = db.add_images("pics", [red, blue], refs=["red", "blue"], ids=[1, 2])
    assert ids == [1, 2]
    hits = db.search("pics", image=red, limit=2)
    assert len(hits) == 2
    assert hits[0].id == 1
    assert hits[0].modality == "image" and hits[0].image == "red"
    assert hits[0].text is None and hits[0].score >= hits[1].score
    # Typed documents carry refs/ids inside.
    ids2 = db.add_images("pics2", [ImageDocument(image=red, ref="r", id=7)])
    assert ids2 == [7]
    hits2 = db.search("pics2", image=red)
    assert hits2[0].image == "r" and hits2[0].modality == "image"


def test_add_videos_and_search_by_video():
    from acorndb import VideoDocument

    db = make_media_db()
    red = _red_png()
    v1, v2 = [red, red], [red + b"x"]
    ids = db.add_videos("clips", [v1, v2], payloads=[{"tag": "a"}, {}], ids=[1, 2])
    assert ids == [1, 2]
    hits = db.search("clips", video=v1, limit=2)
    assert hits[0].id == 1 and hits[0].modality == "video"
    assert hits[0].payload.get("tag") == "a"
    ids2 = db.add_videos("clips2", [VideoDocument(video=v1, ref="clip-1")])
    assert len(ids2) == 1
    assert db.search("clips2", video=v1)[0].video == "clip-1"


def test_search_query_validation():
    db = make_media_db()
    db.add("notes", ["hello"], ids=[1])
    for bad in ({"query": None}, {"query": "a", "image": b"b"}, {"audio": b"a", "composed": {"text": "t"}}):
        try:
            db.search("notes", **bad)  # type: ignore[arg-type]
            raise AssertionError(f"should have raised for {bad}")
        except ValueError:
            pass
    # prompt_name is not a query kind: valid alongside exactly one query.
    assert db.search("notes", "hello", prompt_name="query")[0].id == 1
    # Old positional/text behaviour still works.
    assert db.search("notes", "hello")[0].id == 1
    assert db.search("notes", "hello")[0].modality == "text"


def test_media_unsupported_embedder():
    db = make_db()  # plain FakeEmbedder: no image/video methods
    for fn in (db.embed_image, db.embed_video, db.embed_audio):
        try:
            fn(b"bytes")
            raise AssertionError("should have raised")
        except RuntimeError as exc:
            assert "does not support" in str(exc)
    try:
        db.add_images("pics", [b"bytes"])
        raise AssertionError("should have raised")
    except RuntimeError:
        pass
    try:
        db.add_audios("sounds", [b"bytes"])
        raise AssertionError("should have raised")
    except RuntimeError:
        pass


def test_media_coerce_validation():
    from acorndb import ImageDocument

    db = make_media_db()
    for fn in (db.add_images, db.add_videos):
        try:
            fn("d", [])
            raise AssertionError("should have raised")
        except ValueError:
            pass
    try:
        db.add_images("d", ["a", "b"], ids=[1])
        raise AssertionError("should have raised")
    except ValueError:
        pass
    try:
        db.add_images("d", [ImageDocument(image="a")], ids=[1])
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_media_ref_and_config():
    from acorndb.multimodal import media_ref
    from acorndb import MultimodalConfig

    assert media_ref("a.jpg", "image") == "a.jpg"
    assert media_ref(b"bytes", "image").startswith("image-")
    assert media_ref(b"bytes", "video", explicit="v1") == "v1"
    try:
        media_ref(b"b", "image", explicit="")
        raise AssertionError("should have raised")
    except ValueError:
        pass
    assert MultimodalConfig().truncate_dim == 768
    for bad in ({"truncate_dim": 100}, {"max_frames": 0}):
        try:
            MultimodalConfig(**bad)  # type: ignore[arg-type]
            raise AssertionError("should have raised")
        except ValueError:
            pass


def test_load_image_helpers():
    pytest.importorskip("PIL")
    import numpy as _np

    from acorndb import load_image, sample_video_frames

    img = load_image(_red_png())
    assert img.size == (8, 8) and img.mode == "RGB"
    assert load_image(img).mode == "RGB"  # PIL passthrough
    assert load_image(_np.zeros((4, 4, 3), dtype=_np.uint8)).size == (4, 4)
    assert load_image(io.BytesIO(_red_png())).size == (8, 8)
    for bad in ("./does-not-exist.png", b"not-an-image", 123):
        try:
            load_image(bad)  # type: ignore[arg-type]
            raise AssertionError(f"should have raised for {bad!r}")
        except ValueError:
            pass
    # Frame lists need no video decoder.
    frames = sample_video_frames([_red_png()] * 10, max_frames=4)
    assert len(frames) == 4
    try:
        sample_video_frames([], max_frames=4)
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_sample_video_file(tmp_path):
    """mp4 file/bytes decode (skipped when no imageio+ffmpeg available)."""
    pytest.importorskip("PIL")
    imageio = pytest.importorskip("imageio.v2")
    np = pytest.importorskip("numpy")
    from PIL import Image as _I

    from acorndb import sample_video_frames

    frames = [_I.new("RGB", (16, 16), (i * 25, 0, 0)) for i in range(10)]
    clip = tmp_path / "clip.mp4"
    try:
        imageio.mimsave(str(clip), [np.asarray(f) for f in frames], fps=5)
    except Exception:
        pytest.skip("ffmpeg binary unavailable")
    try:
        assert len(sample_video_frames(clip, max_frames=4)) == 4
        assert len(sample_video_frames(clip.read_bytes(), max_frames=3)) == 3
        assert len(sample_video_frames(str(clip), max_frames=20)) == 10
    except (ValueError, ImportError, OSError):
        pytest.skip("video decoding unavailable")


def test_even_indices_and_sane_total():
    from acorndb.multimodal import _even_indices, _sane_total

    assert _even_indices(3, 10) == [0, 1, 2]
    assert _even_indices(10, 4) == [0, 3, 6, 9]
    assert _sane_total(10) == 10
    assert _sane_total(10.0) == 10
    for bad in (0, -5, float("inf"), "10", None, True, 10**9):
        assert _sane_total(bad) is None


def _sine_wav(sr=16000, secs=1.0, channels=1, freq=440.0) -> bytes:
    import wave as _w

    import numpy as _np

    t = _np.arange(int(sr * secs)) / sr
    mono = 0.5 * _np.sin(2 * _np.pi * freq * t)
    data = _np.stack([mono, mono], axis=1) if channels == 2 else mono
    buf = io.BytesIO()
    with _w.open(buf, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((data * 32767).astype(_np.int16).tobytes())
    return buf.getvalue()


def test_load_audio():
    np = pytest.importorskip("numpy")
    from acorndb import AUDIO_SAMPLE_RATE, load_audio

    assert AUDIO_SAMPLE_RATE == 16000
    wav = _sine_wav(sr=16000)
    a = load_audio(wav)
    assert a.dtype == np.float32 and a.shape == (16000,)
    assert abs(abs(a).max() - 0.5) < 0.01
    # Resample + stereo downmix.
    b = load_audio(_sine_wav(sr=8000, channels=2))
    assert b.shape == (16000,)
    # Bare arrays default to 16 kHz; tuples carry their rate.
    arr8k = (np.zeros(8000, dtype=np.float32), 8000)
    assert load_audio(arr8k).shape == (16000,)
    assert load_audio(np.zeros(16000, dtype=np.float32)).shape == (16000,)
    assert load_audio(io.BytesIO(wav)).shape == (16000,)
    # int16 PCM arrays normalize to [-1, 1].
    c = load_audio(np.array([0, 32767, -32768], dtype=np.int16))
    assert c.dtype == np.float32 and c.max() <= 1.0
    for bad in ("./does-not-exist.wav", b"not-audio", 123, np.zeros((2, 2, 2))):
        try:
            load_audio(bad)  # type: ignore[arg-type]
            raise AssertionError(f"should have raised for {bad!r}")
        except ValueError:
            pass


def test_load_audio_ffmpeg_fallback(tmp_path):
    """Non-wav codecs decode via ffmpeg (skipped when ffmpeg unavailable)."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("imageio_ffmpeg")
    from acorndb import load_audio

    import subprocess

    from imageio_ffmpeg import get_ffmpeg_exe

    clip = tmp_path / "tone.mp3"
    try:
        r = subprocess.run(
            [
                get_ffmpeg_exe(), "-y", "-v", "error",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-ac", "2", "-ar", "44100", str(clip),
            ],
            capture_output=True,
            timeout=120,
        )
        assert r.returncode == 0
    except Exception:
        pytest.skip("ffmpeg synthesis unavailable")
    a = load_audio(clip.read_bytes())
    assert a.dtype == np.float32 and a.shape == (16000,)
    b = load_audio(clip)  # local paths decode without soundfile too
    assert b.shape == (16000,)


def test_add_audios_and_search_by_audio():
    from acorndb import AudioDocument

    db = make_media_db()
    a1, a2 = _sine_wav(), _sine_wav(freq=880.0)
    ids = db.add_audios("sounds", [a1, a2], refs=["a440", "a880"], ids=[1, 2])
    assert ids == [1, 2]
    hits = db.search("sounds", audio=a1, limit=2)
    assert len(hits) == 2
    assert hits[0].id == 1
    assert hits[0].modality == "audio" and hits[0].audio == "a440"
    assert hits[0].text is None and hits[0].score >= hits[1].score
    ids2 = db.add_audios("s2", [AudioDocument(audio=a1, ref="x")])
    assert len(ids2) == 1
    assert db.search("s2", audio=a1)[0].audio == "x"
    # Prompted audio query works too (fake ignores the prompt).
    assert db.search("s2", audio=a1, prompt_name="query")[0].id == ids2[0]


def test_search_composed_and_interleaved():
    db = make_media_db()
    red = _red_png()
    vec = db.embed_interleaved("red square", [red])
    assert len(vec) == 8
    assert abs(sum(x * x for x in vec) - 1.0) < 1e-6
    assert db.embed_interleaved("red square", red) == vec  # single == list-of-one
    assert db.embed_composed({"text": "red square", "image": [red]}) == vec
    db.add_images("pics", [red], refs=["r"], ids=[1])
    hits = db.search("pics", composed={"text": "red square", "image": red})
    assert len(hits) == 1 and hits[0].id == 1
    try:
        db.embed_interleaved("", [red])
        raise AssertionError("should have raised")
    except ValueError:
        pass
    try:
        db.add_audios("s", [])
        raise AssertionError("should have raised")
    except ValueError:
        pass


class StubModel:
    """Records model.encode calls; returns deterministic +1 arange rows."""

    def __init__(self, dim=768):
        self.calls: list = []
        self.dim = dim

    def encode(self, inputs, **kw):
        import numpy as np

        self.calls.append((inputs, kw))
        n = len(inputs) if isinstance(inputs, list) else 1
        return np.arange(n * self.dim, dtype=float).reshape(n, self.dim) + 1.0


def _stub_embedder(**cfg):
    from acorndb import EmbeddingGemmaMultimodalEmbedder, MultimodalConfig

    emb = EmbeddingGemmaMultimodalEmbedder(MultimodalConfig(**cfg))
    emb._model = StubModel()
    return emb


def test_stub_audio_video_composed_shapes():
    from pathlib import Path

    pytest.importorskip("PIL")
    from PIL.Image import Image as _Img

    emb = _stub_embedder()
    stub = emb._model
    red = _red_png()
    wav = _sine_wav()

    vecs = emb.embed_audios([wav, wav])  # wav bytes -> temp wav paths
    assert len(vecs) == 2 and len(vecs[0]) == 768
    (inputs, kw) = stub.calls[-1]
    assert [list(d) for d in inputs] == [["audio"], ["audio"]]
    assert inputs[0]["audio"].endswith(".wav") and inputs[1]["audio"].endswith(".wav")
    assert not Path(inputs[0]["audio"]).exists()  # temp cleaned up

    vec = emb.embed_video([red, red, red])  # frame list, native single pass
    assert len(vec) == 768
    (inputs, kw) = stub.calls[-1]
    assert len(inputs) == 1 and isinstance(inputs[0]["video"], list)
    assert all(isinstance(f, _Img) for f in inputs[0]["video"])
    assert kw["processing_kwargs"] == {"video": {"max_frames": 16}}

    vec = emb.embed_composed(
        {"text": "a cat <|image|>", "image": red}, prompt_name="query"
    )
    assert len(vec) == 768
    (inputs, kw) = stub.calls[-1]
    assert inputs[0]["text"] == "a cat <|image|>"
    assert isinstance(inputs[0]["image"], _Img)
    assert kw["prompt_name"] == "query"

    vecs = emb.embed("hi", prompt_name="STS")
    (inputs, kw) = stub.calls[-1]
    assert inputs == ["hi"] and kw["prompt_name"] == "STS"
    assert "processing_kwargs" not in kw


def test_stub_truncation_and_max_frames():
    emb = _stub_embedder(truncate_dim=128, max_frames=8)
    assert len(emb.embed_audio(_sine_wav())) == 128
    emb.embed_video([_red_png()])
    (inputs, kw) = emb._model.calls[-1]
    assert len(inputs[0]["video"]) <= 8
    assert kw["processing_kwargs"] == {"video": {"max_frames": 8}}


def test_composed_validation_without_model():
    from acorndb import EmbeddingGemmaMultimodalEmbedder, MultimodalConfig

    emb = EmbeddingGemmaMultimodalEmbedder(MultimodalConfig())
    assert emb.backend_used is None
    for bad in (
        "nope",
        {},
        {"nope": 1},
        {"text": ""},
        {"text": 123},
        {"image": []},
        {"audio": []},
        {"image": "missing.png"},
        {"audio": "missing.wav"},
        {"video": "missing.mp4"},
    ):
        try:
            emb.embed_composed(bad)  # type: ignore[arg-type]
            raise AssertionError(f"should have raised for {bad!r}")
        except ValueError:
            pass
    assert emb.backend_used is None  # validation happens before any load
    # Undecodable audio fails before the model is touched (ffmpeg HDR / hint).
    try:
        emb.embed_audio(b"definitely-not-audio")
        raise AssertionError("should have raised")
    except ValueError as exc:
        assert "decoder" in str(exc) or "decode" in str(exc)
    assert emb.backend_used is None


def test_gguf_multimodal_delegation():
    from acorndb import EmbedConfig, EmbeddingGemmaEmbedder

    e = EmbeddingGemmaEmbedder(EmbedConfig(), backend="auto")
    e._mm = FakeMultimodalEmbedder()  # stand-in: no ST download
    assert len(e.embed_audio(b"a")) == 8
    assert len(e.embed_audios([b"a", b"b"])) == 2
    assert len(e.embed_composed({"text": "t"})) == 8
    assert len(e.embed_composed_many([{"text": "t"}])) == 1
    assert len(e.embed_interleaved("t", [b"a"])) == 8
    e.close()
