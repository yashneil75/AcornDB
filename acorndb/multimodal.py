"""Multimodal (text + image + video + audio) embeddings via EmbeddingGemma-2.

EmbeddingGemma-2 (`google/embeddinggemma-2`, 740M params) is natively
multimodal: text (incl. code), images, video and audio share one unified
768-dim space (MRL-truncatable to 512/256/128), so text queries match images,
clips and recordings, and combined text+media inputs embed in a single pass.

The GGUF path (`EmbeddingGemmaEmbedder` via llama-cpp-python / the managed
llama-server) is text-only: the auto-downloaded server binary ships without
vision/audio projectors, so non-text inputs go through `sentence-transformers`
with the full-precision weights (Google's documented path:
`pip install -U sentence-transformers[image,audio,video]`). Text embedded
either way lands in the same unified space, but for cross-modal search use
one backend consistently per database (`EmbeddingGemmaMultimodalEmbedder`).

Video inputs are pre-sampled locally to evenly-spaced frames
(:func:`sample_video_frames`, capped by `MultimodalConfig.max_frames`) and
encoded in one native pass — no model-side video decoder needed.
"""
from __future__ import annotations

import os
import tempfile
import urllib.request
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Iterator, TypedDict, Union
from uuid import uuid4

from .config import ALLOWED_DIMS
from .embedder import _l2_normalize

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from numpy import ndarray as NDArray

#: Anything :func:`load_image` accepts: a local file path or http(s) URL
#: (str), a Path, raw encoded bytes (PNG/JPEG/...), a binary file-like
#: object, a PIL image, or a numpy HWC array.
ImageInput = Union[str, Path, bytes, bytearray, BinaryIO, "PILImage", "NDArray"]

#: Anything :func:`sample_video_frames` accepts: a local video file path or
#: http(s) URL (str), a Path, raw encoded video bytes, a binary file-like
#: object, or a list/tuple of frames (each any :data:`ImageInput`).
VideoInput = Union[str, Path, bytes, bytearray, BinaryIO, Sequence["ImageInput"]]

#: Sample rate (Hz) all audio is normalized to. The model's audio tower
#: consumes 16 kHz mono float32; :func:`load_audio` resamples as needed.
AUDIO_SAMPLE_RATE = 16000

#: Anything :func:`load_audio` accepts: a local file path or http(s) URL
#: (str), a Path, raw encoded bytes (WAV/MP3/FLAC/...), a binary file-like
#: object, a mono float/int numpy waveform (assumed 16 kHz unless given as
#: an `(array, sample_rate)` tuple), or an `(array, sample_rate)` tuple.
AudioInput = Union[
    str, Path, bytes, bytearray, BinaryIO, "NDArray", tuple["NDArray", int]
]


class ComposedInput(TypedDict, total=False):
    """One multi-modality input -> one joint embedding.

    Any non-empty subset of keys; values are single items or lists. Without
    `<|image|>` / `<|video|>` / `<|audio|>` placeholders in `text`, modalities
    encode in dict-key order; with placeholders they interleave at the exact
    positions (counts must match the media items).

    Example:
        {"text": "A jacket similar to <|image|> or <|image|>",
         "image": [img1, img2]}
        {"image": photo, "audio": "clip.wav"}  # no text required
    """

    text: str
    image: Union[ImageInput, Sequence[ImageInput]]
    audio: Union[AudioInput, Sequence[AudioInput]]
    video: Union[VideoInput, Sequence[VideoInput]]


_COMPOSED_KEYS = ("text", "image", "audio", "video")

_DEFAULT_ST_REPO_ID = "google/embeddinggemma-2"


@dataclass(slots=True)
class MultimodalConfig:
    """How to load the full-precision EmbeddingGemma-2 model."""

    repo_id: str = _DEFAULT_ST_REPO_ID
    # Local snapshot dir. None -> sentence-transformers' default HF cache.
    model_dir: Path | None = None
    # Torch device ("cpu", "cuda", ...). None -> sentence-transformers auto.
    device: str | None = None
    truncate_dim: int = 768
    normalize: bool = True
    # Local pre-sampling budget per video (evenly-spaced RGB frames).
    max_frames: int = 16

    def __post_init__(self) -> None:
        if self.truncate_dim not in ALLOWED_DIMS:
            raise ValueError(
                f"truncate_dim must be one of {ALLOWED_DIMS}, got {self.truncate_dim}"
            )
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)
        if self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {self.max_frames}")


# -- media loading ------------------------------------------------------


def _need_pil():  # type: ignore[no-untyped-def]
    try:
        from PIL import Image as _PIL
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for image inputs. "
            "Install with `pip install acorndb[multimodal]`."
        ) from exc
    return _PIL


def _download_bytes(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.read()
    except OSError as exc:
        raise ValueError(f"could not download {url!r}: {exc}") from exc


def load_image(inp: ImageInput) -> PILImage:
    """Load any :data:`ImageInput` into an RGB PIL image.

    Raises `ImportError` if Pillow is missing, `ValueError` on unreadable
    input (missing file, bad bytes, unsupported array shape).
    """
    PIL = _need_pil()
    from PIL.Image import Image as _Img

    if isinstance(inp, _Img):
        return inp.convert("RGB")
    if isinstance(inp, (bytes, bytearray)):
        try:
            return PIL.open(BytesIO(bytes(inp))).convert("RGB")
        except Exception as exc:
            raise ValueError(f"could not decode image bytes: {exc}") from exc
    if hasattr(inp, "read"):  # binary file-like
        try:
            data = inp.read()  # type: ignore[union-attr]
        except Exception as exc:
            raise ValueError(f"could not read image file object: {exc}") from exc
        return load_image(bytes(data))
    if isinstance(inp, str) and inp.startswith(("http://", "https://")):
        return load_image(_download_bytes(inp))
    if isinstance(inp, (str, Path)):
        p = Path(inp)
        if not p.is_file():
            raise ValueError(f"image file not found: {p}")
        try:
            return PIL.open(p).convert("RGB")
        except Exception as exc:
            raise ValueError(f"could not decode image file {p}: {exc}") from exc
    # numpy array (optional dep): uint8 HWC preferred; float 0-1 accepted.
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(inp, np.ndarray):
        arr = inp
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"numpy image must be HWC with 1/3/4 channels, got shape {arr.shape}"
            )
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max(initial=0) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype("uint8")
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype("uint8")
        return PIL.fromarray(arr[:, :, :3], mode="RGB").convert("RGB")
    raise ValueError(
        f"unsupported image input {type(inp).__name__}: expected file path/URL, "
        "bytes, file-like object, PIL image, or numpy HWC array"
    )


def _even_sample(frames: list[PILImage], n: int) -> list[PILImage]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(frames) <= n:
        return list(frames)
    return [frames[i] for i in _even_indices(len(frames), n)]


def _even_indices(total: int, n: int) -> list[int]:
    """Evenly-spaced indices into `total` frames, capped at `n` (no big lists)."""
    if total <= n:
        return list(range(total))
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def _sane_total(total: object) -> int | None:
    # Decoders sometimes report bogus counts (0, negative, inf); only trust
    # small positive ints, else fall back to sequential reads.
    if isinstance(total, bool):
        return None
    if isinstance(total, float) and total.is_integer():
        total = int(total)
    if isinstance(total, int) and 0 < total <= 1_000_000:
        return total
    return None


def _decode_frames_cv2(path: Path) -> list[PILImage]:
    import cv2  # type: ignore[import-not-found]

    from PIL import Image as _PIL

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"could not open video file: {path}")
        total = _sane_total(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total is not None:
            # Seek only to the frames we need (cheap on long videos).
            out: list[PILImage] = []
            for i in _even_indices(total, 1024):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if ok:
                    out.append(
                        _PIL.fromarray(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        ).convert("RGB")
                    )
            return out
        out = []
        while len(out) < 1024:
            ok, frame = cap.read()
            if not ok:
                break
            out.append(
                _PIL.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
            )
        return out
    finally:
        cap.release()


def _decode_frames_imageio(path: Path) -> list[PILImage]:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "video files need a decoder: `pip install acorndb[video]` "
            "(opencv-python-headless) or `pip install imageio imageio-ffmpeg`."
        ) from exc
    from PIL import Image as _PIL

    reader = imageio.get_reader(str(path))
    try:
        try:
            total = _sane_total(len(reader))  # type: ignore[arg-type]
        except Exception:
            total = None
        if total is not None:
            return [
                _PIL.fromarray(reader.get_data(i)).convert("RGB")
                for i in _even_indices(total, 1024)
            ]
        out = [_PIL.fromarray(f).convert("RGB") for f in reader]
        return out[:1024]
    finally:
        reader.close()


def _decode_video_file(path: Path) -> list[PILImage]:
    """Decode frames from a local video file via cv2, falling back to imageio."""
    try:
        return _decode_frames_cv2(path)
    except ImportError:
        pass  # cv2 absent -> try imageio
    try:
        return _decode_frames_imageio(path)
    except ImportError:
        raise ImportError(
            "video files need a decoder: `pip install acorndb[video]` "
            "(opencv-python-headless) or `pip install imageio imageio-ffmpeg`."
        )


@contextmanager
def _local_video_file(video: VideoInput) -> Iterator[Path]:
    """Materialize path/URL/bytes/file-like video input as a local file."""
    if isinstance(video, (list, tuple)):
        raise AssertionError("_local_video_file is not for frame lists")
    if isinstance(video, Path) or (
        isinstance(video, str) and not video.startswith(("http://", "https://"))
    ):
        p = Path(video)
        if not p.is_file():
            raise ValueError(f"video file not found: {p}")
        yield p
        return
    if isinstance(video, str):  # URL
        data = _download_bytes(video)
    elif isinstance(video, (bytes, bytearray)):
        data = bytes(video)
    elif hasattr(video, "read"):
        data = video.read()  # type: ignore[union-attr]
    else:
        raise ValueError(
            f"unsupported video input {type(video).__name__}: expected file "
            "path/URL, bytes, file-like object, or a list of frames"
        )
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        yield Path(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)


def sample_video_frames(
    video: VideoInput, max_frames: int = 16
) -> list[PILImage]:
    """Sample up to `max_frames` evenly-spaced RGB frames from any input.

    Frame lists need no decoder; file/URL/bytes inputs need cv2 or imageio.
    """
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if isinstance(video, (list, tuple)):
        frames = list(video)
        if not frames:
            raise ValueError("video frame list must be non-empty")
        loaded = [load_image(f) for f in frames]
    else:
        with _local_video_file(video) as path:
            loaded = _decode_video_file(path)
        if not loaded:
            raise ValueError(f"no frames decoded from {str(video)[:80]!r}")
    return _even_sample(loaded, max_frames)


# -- audio loading ------------------------------------------------------


def _decode_audio_ffmpeg(data: bytes) -> tuple[Any, int]:
    """Decode any ffmpeg-readable audio bytes -> (mono float32 @16k, 16000)."""
    import subprocess

    try:
        import imageio_ffmpeg as _ff
    except ImportError as exc:
        raise ImportError("imageio-ffmpeg not installed") from exc
    import numpy as _np

    fd, tmp = tempfile.mkstemp(suffix=".acorndb-audio")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        proc = subprocess.run(
            [
                _ff.get_ffmpeg_exe(),
                "-y",
                "-v",
                "error",
                "-i",
                tmp,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                "-f",
                "s16le",
                "-",
            ],
            capture_output=True,
            timeout=300,
        )
    finally:
        Path(tmp).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg could not decode audio: {proc.stderr.decode()[:200]}")
    pcm = _np.frombuffer(proc.stdout, dtype=_np.int16).astype("float32") / 32768.0
    if pcm.size == 0:
        raise ValueError("ffmpeg decoded zero audio samples")
    return pcm, AUDIO_SAMPLE_RATE


def _decode_audio_bytes(data: bytes) -> tuple[Any, int]:
    """Decode raw audio bytes -> (waveform array, sample rate).

    Tries soundfile (wav/mp3/flac/ogg), stdlib wave (PCM wav), then ffmpeg
    via imageio-ffmpeg (any codec). Returns channels-last or mono audio.
    """
    errors: list[str] = []
    try:
        import soundfile as _sf

        arr, sr = _sf.read(BytesIO(data))
        return arr, int(sr)
    except ImportError:
        errors.append("soundfile not installed")
    except Exception as exc:  # present but can't read this payload
        errors.append(f"soundfile: {exc}")
    if data[:4] == b"RIFF":
        try:
            import wave as _wave

            with _wave.open(BytesIO(data), "rb") as f:
                nch, sw, sr = f.getnchannels(), f.getsampwidth(), f.getframerate()
                raw = f.readframes(f.getnframes())
            import numpy as _np

            dtype = {1: "uint8", 2: "int16", 4: "int32"}[sw]
            arr = _np.frombuffer(raw, dtype=_np.dtype(dtype)).reshape(-1, nch)
            return (arr[:, 0] if nch == 1 else arr), int(sr)
        except (ImportError, KeyError, Exception) as exc:
            errors.append(f"wave: {exc}")
    try:
        return _decode_audio_ffmpeg(data)
    except ImportError:
        errors.append("imageio-ffmpeg not installed")
    except Exception as exc:
        errors.append(f"ffmpeg: {exc}")
    raise ValueError(
        "could not decode audio bytes "
        f"({'; '.join(errors)}). Install a decoder with "
        "`pip install soundfile` (wav/mp3/flac/ogg)."
    )


def _to_mono_float(arr: Any) -> Any:
    """Any waveform array -> 1-D float32 in [-1, 1]."""
    import numpy as _np

    a = _np.asarray(arr)
    if _np.issubdtype(a.dtype, _np.integer):
        info = _np.iinfo(a.dtype)
        a = a.astype("float32") / float(max(info.max, 1))
        if info.min == 0:  # unsigned (e.g. uint8 PCM): recenter
            a = a * 2.0 - 1.0
    else:
        a = a.astype("float32")
    if a.ndim == 1:
        mono = a
    elif a.ndim == 2:
        if a.shape[0] <= 2:
            mono = a.mean(axis=0)
        elif a.shape[1] <= 8:
            mono = a.mean(axis=1)
        else:
            raise ValueError(
                f"2-D audio must have a channel axis (size <= 8), got {a.shape}"
            )
    else:
        raise ValueError(f"audio must be 1-D or 2-D, got {a.ndim}-D")
    return _np.clip(mono, -1.0, 1.0).astype("float32")


def _resample_mono(wave: Any, src_sr: int) -> Any:
    """Resample mono float32 audio to :data:`AUDIO_SAMPLE_RATE`."""
    import numpy as _np

    if src_sr == AUDIO_SAMPLE_RATE:
        return wave
    if src_sr <= 0:
        raise ValueError(f"sample rate must be > 0, got {src_sr}")
    n_out = max(1, round(len(wave) * AUDIO_SAMPLE_RATE / src_sr))
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(AUDIO_SAMPLE_RATE, src_sr)
        return resample_poly(wave, AUDIO_SAMPLE_RATE // g, src_sr // g).astype(
            "float32"
        )[:n_out]
    except ImportError:
        old = _np.arange(len(wave))
        new = _np.linspace(0, len(wave) - 1, n_out)
        return _np.interp(new, old, wave).astype("float32")


def load_audio(
    inp: AudioInput, sample_rate: int | None = None
) -> NDArray:
    """Load any :data:`AudioInput` -> mono float32 @ 16 kHz.

    Bare arrays use `sample_rate` (default 16 kHz); `(array, sr)` tuples carry
    their own. Files/URLs/bytes detect the rate from the container/codec and
    are resampled as needed.
    """
    import numpy as _np

    sr = sample_rate or AUDIO_SAMPLE_RATE
    if isinstance(inp, tuple):
        if len(inp) != 2 or not isinstance(inp[0], _np.ndarray):
            raise ValueError(
                "audio tuples must be (waveform array, sample_rate int)"
            )
        arr, sr = inp[0], int(inp[1])
    elif isinstance(inp, _np.ndarray):
        arr = inp
    elif isinstance(inp, (bytes, bytearray)):
        arr, sr = _decode_audio_bytes(bytes(inp))
    elif hasattr(inp, "read"):  # binary file-like
        try:
            data = inp.read()  # type: ignore[union-attr]
        except Exception as exc:
            raise ValueError(f"could not read audio file object: {exc}") from exc
        arr, sr = _decode_audio_bytes(bytes(data))
    elif isinstance(inp, str) and inp.startswith(("http://", "https://")):
        arr, sr = _decode_audio_bytes(_download_bytes(inp))
    elif isinstance(inp, (str, Path)):
        p = Path(inp)
        if not p.is_file():
            raise ValueError(f"audio file not found: {p}")
        try:
            import soundfile as _sf

            arr, sr = _sf.read(str(p))
        except ImportError:
            arr, sr = _decode_audio_bytes(p.read_bytes())
        except Exception as exc:
            raise ValueError(f"could not decode audio file {p}: {exc}") from exc
    else:
        raise ValueError(
            f"unsupported audio input {type(inp).__name__}: expected file "
            "path/URL, bytes, file-like object, numpy waveform, or "
            "(waveform, sample_rate) tuple"
        )
    return _resample_mono(_to_mono_float(arr), int(sr))


def _write_wav_mono16(path: Path, wave: Any) -> None:
    """Write 16 kHz mono float audio as 16-bit PCM WAV (stdlib only)."""
    import wave as _wave

    import numpy as _np

    pcm = (_np.clip(_np.asarray(wave, dtype="float32"), -1.0, 1.0) * 32767).astype(
        "int16"
    )
    with _wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(AUDIO_SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


def media_ref(inp: object, kind: str, explicit: str | None = None) -> str:
    """Human-readable payload ref for an image/video input.

    Explicit ref wins; path/URL inputs reuse their string; everything else
    (bytes, PIL, arrays, frame lists) gets a short `kind-<hex>` id.
    """
    if explicit is not None:
        if not explicit:
            raise ValueError("ref must be a non-empty string")
        return explicit
    if isinstance(inp, (str, Path)):
        return str(inp)
    return f"{kind}-{uuid4().hex[:8]}"


# -- embedder -----------------------------------------------------------


class EmbeddingGemmaMultimodalEmbedder:
    """Full-precision EmbeddingGemma-2 via sentence-transformers.

    Same unified 768-dim space as the GGUF text path, plus native image,
    video, audio and single-pass composed (text+media) inputs. Needs
    `pip install acorndb[multimodal]`; the weights
    (`google/embeddinggemma-2`) download from HF Hub on first use.

    Example:
        emb = EmbeddingGemmaMultimodalEmbedder()
        vec = emb.embed_image("photo.jpg")   # len 768, L2-normalized
        vec = emb.embed_audio("clip.wav")
        vec = emb.embed_interleaved("a cat <|image|>", ["cat.jpg"])
    """

    def __init__(self, config: MultimodalConfig | None = None) -> None:
        self.config: MultimodalConfig = config or MultimodalConfig()
        self._model: Any | None = None  # SentenceTransformer, loaded lazily
        self.backend_used: str | None = None  # "st" once loaded

    @property
    def dimension(self) -> int:
        return self.config.truncate_dim

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for image/video/audio and "
                "composed embeddings. Install with "
                "`pip install acorndb[multimodal]`."
            ) from exc
        kwargs: dict[str, Any] = {}
        if self.config.model_dir is not None:
            kwargs["cache_folder"] = str(self.config.model_dir)
        if self.config.device is not None:
            kwargs["device"] = self.config.device
        self._model = SentenceTransformer(self.config.repo_id, **kwargs)
        self.backend_used = "st"
        return self._model

    def close(self) -> None:
        """Release the model handle (no server involved; safe no-op otherwise)."""
        self._model = None

    # -- core ---------------------------------------------------------
    def _post_process(self, vec: Sequence[float]) -> list[float]:
        out = list(vec[: self.config.truncate_dim])
        if self.config.normalize:
            out = _l2_normalize(out)
        return out

    def _encode(
        self,
        inputs: list[Any],
        batch_size: int,
        prompt_name: str | None = None,
        processing_kwargs: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """One model.encode call per batch chunk; truncate + normalize rows."""
        model = self._get_model()
        kw: dict[str, Any] = {
            "batch_size": batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": False,
        }
        if prompt_name is not None:
            kw["prompt_name"] = prompt_name
        if processing_kwargs is not None:
            kw["processing_kwargs"] = processing_kwargs
        import numpy as _np

        out: list[list[float]] = []
        for i in range(0, len(inputs), batch_size):
            arr = model.encode(inputs[i : i + batch_size], **kw)
            mat = _np.atleast_2d(_np.asarray(arr, dtype=float))
            out.extend(self._post_process(row) for row in mat.tolist())
        return out

    def _encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        return self._encode(list(texts), batch_size, prompt_name)

    def _encode_images(
        self,
        images: Sequence[PILImage],
        batch_size: int,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        return self._encode(list(images), batch_size, prompt_name)

    def _video_kwargs(self) -> dict[str, Any]:
        # Frames are already capped locally; mirror the cap model-side.
        return {"video": {"max_frames": self.config.max_frames}}

    def _encode_dicts(
        self,
        dicts: list[dict[str, Any]],
        batch_size: int,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        return self._encode(dicts, batch_size, prompt_name, self._video_kwargs())

    def _prepare_video(self, video: VideoInput) -> list[PILImage]:
        """Model-ready video value: evenly-sampled PIL frames.

        Everything (files, URLs, bytes, frame lists) is sampled locally via
        :func:`sample_video_frames` and capped at `config.max_frames`, so no
        model-side video decoder (torchcodec) is ever needed.
        """
        return sample_video_frames(video, self.config.max_frames)

    def _prepare_audio(self, audio: AudioInput, tmpdir: Path) -> str:
        """Model-ready audio value: always a temp 16 kHz mono WAV path.

        Everything (including file paths) is decoded + normalized locally via
        :func:`load_audio`, so the model only ever sees clean WAVs and no
        audio decoder backend is needed at encode time.
        """
        wave = load_audio(audio)
        dest = tmpdir / f"audio-{uuid4().hex[:8]}.wav"
        _write_wav_mono16(dest, wave)
        return str(dest)

    def _normalize_composed(
        self, item: ComposedInput, tmpdir: Path
    ) -> dict[str, Any]:
        """Validate + normalize one composed input to an ST-ready dict."""
        if not isinstance(item, dict):
            raise ValueError(
                "composed inputs must be dicts keyed by "
                "'text'/'image'/'audio'/'video'"
            )
        unknown = [k for k in item if k not in _COMPOSED_KEYS]
        if unknown:
            raise ValueError(
                f"unknown composed keys {unknown}; expected a subset of "
                f"{list(_COMPOSED_KEYS)}"
            )
        if not item:
            raise ValueError(
                "composed input must include at least one of "
                "'text'/'image'/'audio'/'video'"
            )
        out: dict[str, Any] = {}
        if "text" in item:
            text = item["text"]
            if not isinstance(text, str) or not text:
                raise ValueError("composed 'text' must be a non-empty string")
            out["text"] = text
        if "image" in item:
            imgs = item["image"]
            single = not isinstance(imgs, (list, tuple))
            loaded = [load_image(i) for i in ([imgs] if single else list(imgs))]  # type: ignore[union-attr]
            if not loaded:
                raise ValueError("composed 'image' must be non-empty")
            out["image"] = loaded[0] if single else loaded
        if "audio" in item:
            auds = item["audio"]
            # Tuples are always one (waveform, sample_rate) audio; lists hold
            # several audio segments in one composed embedding.
            single = isinstance(auds, tuple) or not isinstance(auds, list)
            seq = [auds] if single else list(auds)
            if not seq:
                raise ValueError("composed 'audio' must be non-empty")
            prepared = [self._prepare_audio(a, tmpdir) for a in seq]
            out["audio"] = prepared[0] if single else prepared
        if "video" in item:
            # One video per dict (path/URL, bytes, or a frame list); pass
            # several videos as several dicts to embed_composed_many.
            out["video"] = self._prepare_video(item["video"])
        return out

    def embed(self, text: str, *, prompt_name: str | None = None) -> list[float]:
        """Embed a single string -> unit-norm vector."""
        if not text:
            raise ValueError("text must be a non-empty string")
        return self._encode_texts([text], 1, prompt_name)[0]

    def embed_many(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        *,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Embed many strings."""
        if not texts:
            return []
        for t in texts:
            if not t:
                raise ValueError("texts must all be non-empty strings")
        return self._encode_texts(texts, batch_size, prompt_name)

    def embed_image(
        self, image: ImageInput, *, prompt_name: str | None = None
    ) -> list[float]:
        """Embed a single image -> unit-norm vector (unified space)."""
        return self.embed_images([image], prompt_name=prompt_name)[0]

    def embed_images(
        self,
        images: Sequence[ImageInput],
        batch_size: int = 32,
        *,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Embed many images (file path/URL, bytes, PIL, or numpy array)."""
        if not images:
            return []
        loaded = [load_image(im) for im in images]
        out: list[list[float]] = []
        for i in range(0, len(loaded), batch_size):
            out.extend(
                self._encode_images(loaded[i : i + batch_size], batch_size, prompt_name)
            )
        return out

    def embed_video(
        self, video: VideoInput, *, prompt_name: str | None = None
    ) -> list[float]:
        """Embed a video in one native pass -> unit-norm vector.

        Any input is pre-sampled locally to `max_frames` evenly-spaced
        frames first (frame lists need no decoder; files/bytes need cv2 or
        imageio), then encoded jointly.
        """
        return self.embed_videos([video], prompt_name=prompt_name)[0]

    def embed_videos(
        self,
        videos: Sequence[VideoInput],
        batch_size: int = 32,
        *,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Embed many videos (file path/URL, bytes, or lists of frames)."""
        if not videos:
            return []
        dicts = [{"video": self._prepare_video(v)} for v in videos]
        return self._encode_dicts(dicts, batch_size, prompt_name)

    def embed_audio(
        self, audio: AudioInput, *, prompt_name: str | None = None
    ) -> list[float]:
        """Embed audio (file path/URL, bytes, waveform, `(wave, sr)`) -> vector."""
        return self.embed_audios([audio], prompt_name=prompt_name)[0]

    def embed_audios(
        self,
        audios: Sequence[AudioInput],
        batch_size: int = 32,
        *,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Embed many audio inputs."""
        if not audios:
            return []
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            dicts = [{"audio": self._prepare_audio(a, tmpdir)} for a in audios]
            return self._encode_dicts(dicts, batch_size, prompt_name)

    def embed_composed(
        self, item: ComposedInput, *, prompt_name: str | None = None
    ) -> list[float]:
        """Embed one multi-modality dict in a single pass -> unit-norm vector.

        Example:
            emb.embed_composed({"text": "a cat", "image": "cat.jpg"})
            emb.embed_composed({"image": photo, "audio": "meow.wav"})
        """
        return self.embed_composed_many([item], prompt_name=prompt_name)[0]

    def embed_composed_many(
        self,
        items: Sequence[ComposedInput],
        batch_size: int = 32,
        *,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Embed many multi-modality dicts (heterogeneous batches welcome)."""
        if not items:
            return []
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            dicts = [self._normalize_composed(d, tmpdir) for d in items]
            return self._encode_dicts(dicts, batch_size, prompt_name)

    def embed_interleaved(
        self,
        text: str,
        images: ImageInput | Sequence[ImageInput],
        *,
        prompt_name: str | None = None,
    ) -> list[float]:
        """Single-pass joint embedding of text + one or more images.

        Put `<|image|>` placeholders in `text` to interleave images at exact
        positions (one per image); otherwise images encode in order.
        """
        if not text:
            raise ValueError("text must be a non-empty string")
        seq = list(images) if isinstance(images, (list, tuple)) else [images]
        if not seq:
            raise ValueError("images must be non-empty")
        return self.embed_composed(
            {"text": text, "image": seq}, prompt_name=prompt_name
        )

    # Aliases matching sentence-transformers conventions.
    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_many(texts)


__all__ = [
    "AUDIO_SAMPLE_RATE",
    "AudioInput",
    "ComposedInput",
    "EmbeddingGemmaMultimodalEmbedder",
    "ImageInput",
    "MultimodalConfig",
    "VideoInput",
    "load_audio",
    "load_image",
    "media_ref",
    "sample_video_frames",
]
