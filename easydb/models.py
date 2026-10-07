"""Typed dataclasses shared across EasyDB."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

# Qdrant distance metric names (kept as Literal so users get autocomplete).
Distance = Literal["Cosine", "Euclid", "Dot"]

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


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One typed search result."""

    id: PointId
    score: float
    text: str | None
    payload: dict[str, Any]
