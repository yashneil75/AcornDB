"""Managed localhost llama-server: invisible embedding backend, no user ops.

Used automatically when in-process llama-cpp-python cannot load the model
(e.g. its bundled llama.cpp predates the model architecture). The binary
is auto-downloaded (see llama_bin), started on 127.0.0.1 with an ephemeral
port, and torn down on close().
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from .llama_bin import cache_dir, ensure_server_binary


def parse_embedding_response(data: object) -> list[float]:
    """Handle both server response shapes.

    Old: {"embedding": [...]}. New (b11457+): [{"index": 0, "embedding": [[...]]}].
    """
    if isinstance(data, dict) and isinstance(data.get("embedding"), list):
        vec = data["embedding"]
        return list(vec[0] if vec and isinstance(vec[0], list) else vec)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return parse_embedding_response(data[0])
    raise ValueError(f"unrecognized embedding response shape: {str(data)[:120]}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ManagedLlamaServer:
    """A llama-server process owned by the library. Start via start()."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        use_vulkan: bool = False,
        startup_timeout: float = 180.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.use_vulkan = use_vulkan
        self.startup_timeout = startup_timeout
        self.port: int | None = None
        self.base_url: str | None = None
        self._proc: subprocess.Popen | None = None
        self.log_path = cache_dir() / "llama-server.log"

    def start(self) -> str:
        if self._proc is not None:
            return self.base_url  # type: ignore[return-value]
        binary = ensure_server_binary(use_vulkan=self.use_vulkan)
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        with open(self.log_path, "ab") as log:
            self._proc = subprocess.Popen(
                [str(binary), "-m", str(self.model_path), "--embeddings", "--port", str(self.port)],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        deadline = time.time() + self.startup_timeout
        last_err = ""
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited during startup; see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return self.base_url
            except OSError as exc:  # server not up yet
                last_err = str(exc)
            time.sleep(0.5)
        self.close()
        raise TimeoutError(f"llama-server did not start in time ({last_err})")

    def embed(self, text: str) -> list[float]:
        if self.base_url is None:
            raise RuntimeError("server not started; call start() first")
        body = json.dumps({"content": text}).encode()
        req = urllib.request.Request(
            self.base_url + "/embedding",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return parse_embedding_response(json.loads(r.read()))

    def embed_many(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    def __enter__(self) -> ManagedLlamaServer:  # noqa: PYI034
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - best-effort cleanup in __del__
            pass


__all__ = ["ManagedLlamaServer", "parse_embedding_response"]
