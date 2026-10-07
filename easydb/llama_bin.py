"""Auto-download of prebuilt llama.cpp binaries (no compiler needed downstream).

Pinned to the first nightly containing PR #30054 (embeddinggemma2 support).
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# First nightly whose master includes #30054 (gemma-embedding2). Bump forward only.
LLAMA_BUILD = "b11457"
_GH = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}"

# (system, machine) -> (asset_name, exe_name)
_ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ("windows", "amd64"): (f"llama-{LLAMA_BUILD}-bin-win-cpu-x64.zip", "llama-server.exe"),
    ("windows", "x86_64"): (f"llama-{LLAMA_BUILD}-bin-win-cpu-x64.zip", "llama-server.exe"),
    ("windows", "arm64"): (f"llama-{LLAMA_BUILD}-bin-win-cpu-arm64.zip", "llama-server.exe"),
    ("linux", "x86_64"): (f"llama-{LLAMA_BUILD}-bin-ubuntu-x64.tar.gz", "llama-server"),
    ("linux", "aarch64"): (f"llama-{LLAMA_BUILD}-bin-ubuntu-arm64.tar.gz", "llama-server"),
    ("darwin", "arm64"): (f"llama-{LLAMA_BUILD}-bin-macos-arm64.tar.gz", "llama-server"),
    ("darwin", "x86_64"): (f"llama-{LLAMA_BUILD}-bin-macos-x64.tar.gz", "llama-server"),
}

_VULKAN_WIN = (f"llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip", "llama-server.exe")


def cache_dir() -> Path:
    return Path.home() / ".cache" / "easydb" / "bin"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp (not NamedTemporaryFile): the fd is closed before replace(),
    # which Windows requires.
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent))
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url, timeout=300) as r:
            shutil.copyfileobj(r, f, length=1024 * 256)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    Path(tmp).replace(dest)


def ensure_server_binary(*, use_vulkan: bool = False) -> Path:
    """Return path to a working llama-server binary, downloading it if needed."""
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    if sysname == "windows" and use_vulkan and machine in ("amd64", "x86_64"):
        asset, exe = _VULKAN_WIN
    else:
        try:
            asset, exe = _ASSETS[(sysname, machine)]
        except KeyError:
            raise RuntimeError(
                f"no prebuilt llama.cpp {LLAMA_BUILD} binary for {sysname}/{machine}. "
                "Install llama.cpp manually and point easydb at it."
            ) from None
    dest_dir = cache_dir() / asset.replace(".tar.gz", "").replace(".zip", "")
    exe_path = dest_dir / exe
    if exe_path.exists():
        return exe_path
    archive = cache_dir() / asset
    if not archive.exists():
        _download(f"{_GH}/{asset}", archive)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest_dir)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest_dir, filter="data")
    if not exe_path.exists():
        # Fall back to a recursive search (asset layout varies by platform).
        found = next(dest_dir.rglob(exe), None)
        if found is None:
            raise RuntimeError(f"{exe} not found inside {asset}")
        return found
    if sys.platform != "win32":
        exe_path.chmod(0o755)
    return exe_path


__all__ = ["LLAMA_BUILD", "cache_dir", "ensure_server_binary"]
