from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

POPPLER_VERSION = "26.09.0-0"
POPPLER_SHA256 = "7a6f256a0ddf7536182246a5733331bf4677cbcc34f4663774947ad34556c8d0"
POPPLER_URL = (
    "https://github.com/oschwartz10612/poppler-windows/releases/download/"
    f"v{POPPLER_VERSION}/Release-{POPPLER_VERSION}.zip"
)


def ensure_poppler() -> str:
    app_dir = Path(__file__).resolve().parent
    tools_dir = app_dir / "tools" / "poppler"
    candidates = (
        tools_dir / "Library" / "bin" / "pdftotext.exe",
        tools_dir / "bin" / "pdftotext.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    tools_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ffcs-poppler-") as temp:
        archive = Path(temp) / "poppler.zip"
        request = urllib.request.Request(
            POPPLER_URL,
            headers={"User-Agent": "ForeFlight-ContentPack-Studio"},
        )
        with urllib.request.urlopen(request, timeout=90) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)

        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest.lower() != POPPLER_SHA256:
            raise RuntimeError(
                f"Poppler SHA-256 mismatch: expected {POPPLER_SHA256}, received {digest}"
            )

        extract_dir = Path(temp) / "extract"
        extract_dir.mkdir()
        with zipfile.ZipFile(archive) as package:
            package.extractall(extract_dir)

        executable = next(extract_dir.rglob("pdftotext.exe"), None)
        if executable is None:
            raise RuntimeError("The downloaded Poppler archive does not contain pdftotext.exe.")

        library_dir = executable.parent.parent
        staged = Path(temp) / "Library"
        shutil.copytree(library_dir, staged)
        if tools_dir.exists():
            shutil.rmtree(tools_dir)
        tools_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged, tools_dir / "Library")

    installed = tools_dir / "Library" / "bin" / "pdftotext.exe"
    if not installed.is_file():
        raise RuntimeError("Poppler installation completed without pdftotext.exe.")
    return str(installed)
