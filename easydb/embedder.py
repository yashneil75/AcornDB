"""GGUF embedder backed by llama-cpp-python.

Model: unsloth/embeddinggemma-2-GGUF / embeddinggemma-2-UD-Q4_K_XL.gguf
Native dim 768, MRL-truncatable to 512/256/128.
"""
from __future__ import annotations

import math
from typing import Sequence

from .config import EmbedConfig


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class EmbeddingGemmaEmbedder:
    """Typed wrapper around the EmbeddingGemma-2 GGUF file.

    Backend is automatic: in-process llama-cpp-python when it can load the
    model, otherwise a library-managed localhost llama-server (binary
    auto-downloaded, no compiler or ops needed). Forced via backend=.

    Example:
        from easydb import EmbedConfig, EmbeddingGemmaEmbedder

        embedder = EmbeddingGemmaEmbedder(EmbedConfig(truncate_dim=768))
        vec = embedder.embed("hello world")
    """

    def __init__(
        self,
        config: EmbedConfig | None = None,
        *,
        backend: str = "auto",
        use_vulkan: bool = False,
    ) -> None:
        if backend not in ("auto", "cpp", "server"):
            raise ValueError("backend must be 'auto', 'cpp' or 'server'")
        self.config: EmbedConfig = config or EmbedConfig()
        self.backend = backend
        self.use_vulkan = use_vulkan
        self._llm: object | None = None  # llama_cpp.Llama, imported lazily
        self._server: object | None = None  # ManagedLlamaServer, started lazily
        self.backend_used: str | None = None  # "cpp" | "server" once resolved

    # -- properties -----------------------------------------------------
    @property
    def dimension(self) -> int:
        """Effective vector size after MRL truncation."""
        return self.config.truncate_dim

    @property
    def model_path(self: EmbeddingGemmaEmbedder):  # type: ignore[no-untyped-def]
        from pathlib import Path

        return Path(self.config.model_path)

    # -- setup ----------------------------------------------------------
    def ensure_model_downloaded(self):  # type: ignore[no-untyped-def]
        """Download the GGUF from Hugging Face Hub if missing. Returns local path."""
        from pathlib import Path

        dest: Path = Path(self.config.model_path)
        if dest.exists():
            return dest
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface-hub is required for auto-download. "
                "Install with `pip install easydb[hub]` or "
                "manually download "
                f"{self.config.repo_id}/{self.config.filename} to {dest}."
            ) from exc
        dest.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=self.config.repo_id,
            filename=self.config.filename,
            local_dir=str(dest.parent),
        )
        # hf_hub_download with local_dir nests strangely on some versions;
        # normalise to exactly dest.
        from pathlib import Path as _P

        dl = _P(downloaded)
        if dl.resolve() != dest.resolve():
            if dest.exists():
                pass
            else:
                dl.rename(dest)
                downloaded = str(dest)
        return _P(downloaded)

    def _get_server(self):  # type: ignore[no-untyped-def]
        """Lazily start the managed localhost llama-server."""
        from .server_embedder import ManagedLlamaServer

        if self._server is not None:
            return self._server
        model_file = str(self.ensure_model_downloaded())
        server = ManagedLlamaServer(model_file, use_vulkan=self.use_vulkan)
        server.start()
        self._server = server
        return server

    def _get_backend(self):  # type: ignore[no-untyped-def]
        """Resolve to llama-cpp (in-process) or managed server, per backend=."""
        if self.backend == "cpp":
            self.backend_used = "cpp"
            return self._get_llm()
        if self.backend == "server":
            self.backend_used = "server"
            return self._get_server()
        # auto: prefer in-process, fall back to managed server on load failure
        try:
            llm = self._get_llm()
        except (ImportError, ValueError):
            self._llm = None
            self.backend_used = "server"
            return self._get_server()
        self.backend_used = "cpp"
        return llm

    def close(self) -> None:
        """Shut down the managed server, if one was started."""
        server, self._server = self._server, None
        if server is not None:
            server.close()  # type: ignore[union-attr]

    def _get_llm(self):  # type: ignore[no-untyped-def]
        """Lazily construct llama_cpp.Llama with embedding=True."""
        if self._llm is not None:
            return self._llm
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required. Install with "
                "`pip install easydb[embed]` (or `pip install llama-cpp-python`)."
            ) from exc
        model_file = str(self.ensure_model_downloaded())
        kwargs: dict = {
            "model_path": model_file,
            "embedding": self.config.embedding,
            "n_ctx": self.config.n_ctx,
            "n_batch": self.config.n_batch,
            "n_gpu_layers": self.config.n_gpu_layers,
            "verbose": self.config.verbose,
        }
        if self.config.n_threads is not None:
            kwargs["n_threads"] = self.config.n_threads
        self._llm = Llama(**kwargs)
        return self._llm

    # -- core -----------------------------------------------------------
    def _post_process(self, vec: Sequence[float]) -> list[float]:
        out = list(vec[: self.config.truncate_dim])
        if self.config.normalize:
            out = _l2_normalize(out)
        return out

    def embed(self, text: str) -> list[float]:
        """Embed a single string -> unit-norm vector."""
        if not text:
            raise ValueError("text must be a non-empty string")
        backend = self._get_backend()
        raw = backend.embed(text)  # type: ignore[union-attr]
        return self._post_process(raw)

    def embed_many(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> list[list[float]]:
        """Embed many strings, one at a time, on either backend."""
        if not texts:
            return []
        backend = self._get_backend()
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            for text in texts[i : i + batch_size]:
                if not text:
                    raise ValueError("texts must all be non-empty strings")
                raw = backend.embed(text)  # type: ignore[union-attr]
                out.append(self._post_process(raw))
        return out

    # Aliases matching sentence-transformers conventions.
    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_many(texts)
