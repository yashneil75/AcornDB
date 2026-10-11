"""AcornDB - typed Qdrant + EmbeddingGemma-2-GGUF vector library."""
from .config import (
    DEFAULT_FILENAME,
    DEFAULT_REPO_ID,
    AcornDBConfig,
    EmbedConfig,
    QdrantConfig,
)
from .embedder import EmbeddingGemmaEmbedder
from .llama_bin import LLAMA_BUILD
from .models import (
    AudioDocument,
    DatabaseInfo,
    Document,
    ImageDocument,
    SearchHit,
    VideoDocument,
)
from .multimodal import (
    AUDIO_SAMPLE_RATE,
    AudioInput,
    ComposedInput,
    EmbeddingGemmaMultimodalEmbedder,
    ImageInput,
    MultimodalConfig,
    VideoInput,
    load_audio,
    load_image,
    sample_video_frames,
)
from .server_embedder import ManagedLlamaServer, parse_embedding_response
from .store import AcornDB

__all__ = [
    "AUDIO_SAMPLE_RATE",
    "DEFAULT_FILENAME",
    "DEFAULT_REPO_ID",
    "LLAMA_BUILD",
    "AudioDocument",
    "AudioInput",
    "ComposedInput",
    "DatabaseInfo",
    "Document",
    "AcornDB",
    "AcornDBConfig",
    "EmbedConfig",
    "EmbeddingGemmaEmbedder",
    "EmbeddingGemmaMultimodalEmbedder",
    "ImageDocument",
    "ImageInput",
    "ManagedLlamaServer",
    "MultimodalConfig",
    "QdrantConfig",
    "SearchHit",
    "VideoDocument",
    "VideoInput",
    "load_audio",
    "load_image",
    "parse_embedding_response",
    "sample_video_frames",
]

__version__ = "0.1.0"
