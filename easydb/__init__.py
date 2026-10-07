"""EasyDB - typed Qdrant + EmbeddingGemma-2-GGUF vector library."""
from .config import DEFAULT_FILENAME, DEFAULT_REPO_ID, EasyDBConfig, EmbedConfig, QdrantConfig
from .embedder import EmbeddingGemmaEmbedder
from .llama_bin import LLAMA_BUILD
from .models import DatabaseInfo, Document, SearchHit
from .server_embedder import ManagedLlamaServer, parse_embedding_response
from .store import EasyDB

__all__ = [
    "EasyDB",
    "EmbeddingGemmaEmbedder",
    "ManagedLlamaServer",
    "parse_embedding_response",
    "LLAMA_BUILD",
    "EasyDBConfig",
    "EmbedConfig",
    "QdrantConfig",
    "DatabaseInfo",
    "Document",
    "SearchHit",
    "DEFAULT_REPO_ID",
    "DEFAULT_FILENAME",
]

__version__ = "0.1.0"
