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
from .models import DatabaseInfo, Document, SearchHit
from .server_embedder import ManagedLlamaServer, parse_embedding_response
from .store import AcornDB

__all__ = [
    "DEFAULT_FILENAME",
    "DEFAULT_REPO_ID",
    "LLAMA_BUILD",
    "DatabaseInfo",
    "Document",
    "AcornDB",
    "AcornDBConfig",
    "EmbedConfig",
    "EmbeddingGemmaEmbedder",
    "ManagedLlamaServer",
    "QdrantConfig",
    "SearchHit",
    "parse_embedding_response",
]

__version__ = "0.1.0"
