"""Typed dataclasses shared across AcornDB."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Union

if TYPE_CHECKING:
    from .multimodal import AudioInput, ImageInput, VideoInput

# Qdrant distance metric names (kept as Literal so users get autocomplete).
Distance = Literal["Cosine", "Euclid", "Dot"]

# Which modality a stored point / search hit belongs to.
Modality = Literal["text", "image", "video", "audio"]

# Point ids accepted by Qdrant.
PointId = Union[int, str]


@dataclass(frozen=True, slots=True)
class DatabaseInfo:
    """Metadata about one Qdrant collection (one 'database')."""

    name: str
    vector_size: int
    distance: str
    points_count: int | None = None


@dataclass(slots=True)
class Document:
    """A single text record ready to be embedded + stored."""

    text: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: PointId | None = None
    vector: list[float] | None = None


@dataclass(slots=True)
class ImageDocument:
    """A single image record ready to be embedded + stored."""

    image: ImageInput
    payload: dict[str, Any] = field(default_factory=dict)
    id: PointId | None = None
    # Payload ref for the image (path/URL string). None -> derived from input
    # (paths/URLs reuse their string; bytes/PIL get a short generated id).
    ref: str | None = None
    vector: list[float] | None = None


@dataclass(slots=True)
class VideoDocument:
    """A single video record ready to be embedded + stored."""

    video: VideoInput
    payload: dict[str, Any] = field(default_factory=dict)
    id: PointId | None = None
    # Payload ref for the video (path/URL string). None -> derived from input.
    ref: str | None = None
    vector: list[float] | None = None


@dataclass(slots=True)
class AudioDocument:
    """A single audio record ready to be embedded + stored."""

    audio: AudioInput
    payload: dict[str, Any] = field(default_factory=dict)
    id: PointId | None = None
    # Payload ref for the audio (path/URL string). None -> derived from input.
    ref: str | None = None
    vector: list[float] | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One typed search result."""

    id: PointId
    score: float
    text: str | None
    payload: dict[str, Any]
    modality: Modality = "text"
    image: str | None = None
    video: str | None = None
    audio: str | None = None
