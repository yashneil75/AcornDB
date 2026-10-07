"""Typed configuration for EasyDB."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import Distance

DEFAULT_REPO_ID = "unsloth/embeddinggemma-2-GGUF"
DEFAULT_FILENAME = "embeddinggemma-2-UD-Q4_K_XL.gguf"

# EmbeddingGemma-2 native dim is 768. MRL supports 128/256/512/768.
ALLOWED_DIMS = (128, 256, 512, 768)


@dataclass(slots=True)
class EmbedConfig:
    """How to load the GGUF embedding model."""

    repo_id: str = DEFAULT_REPO_ID
    filename: str = DEFAULT_FILENAME
    # Where the .gguf is cached. Defaults to ~/.cache/easydb/models.
    model_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "easydb" / "models"
    )
    n_ctx: int = 8192
    n_batch: int = 512
    n_threads: int | None = None  # None -> llama.cpp auto
    n_gpu_layers: int = -1  # -1 = offload all possible layers
    embedding: bool = True
    normalize: bool = True
    truncate_dim: int = 768
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.truncate_dim not in ALLOWED_DIMS:
            raise ValueError(
                f"truncate_dim must be one of {ALLOWED_DIMS}, got {self.truncate_dim}"
            )
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.filename


@dataclass(slots=True)
class QdrantConfig:
    """How to connect to Qdrant. Pick exactly one of url / path / in-memory."""

    url: str | None = None  # e.g. "http://localhost:6333"
    api_key: str | None = None
    # Local on-disk path, e.g. "./qdrant_data". Mutually exclusive with url.
    path: str | None = None
    # If True and neither url nor path is set, use QdrantClient(":memory:").
    in_memory: bool = False
    prefer_grpc: bool = False
    timeout: float = 60.0


@dataclass(slots=True)
class EasyDBConfig:
    """Top-level config bundling embedding + vector-db settings."""

    embed: EmbedConfig = field(default_factory=EmbedConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    default_distance: Distance = "Cosine"
