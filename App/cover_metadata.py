from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentMetadata:
    document_type: str
    effective_date: str
    effective_from: str
    effective_to: str


def _pdftotext() -> str:
    app_dir = Path(__file__).resolve().parent
    candidates = (
        app_dir / "tools" / "poppler" / "Library" / "bin" / "pdftotext.exe",
        app_dir / "tools" / "poppler" / "bin" / "pdftotext.exe",
        app_dir / "tools" / "poppler" / "bin" / "pdftotext",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("pdftotext")
    if found:
        return found
    raise RuntimeError(
        "Poppler pdftotext is not installed. In Visual Studio, open App/setup_poppler.ps1 "
        "and run it in a PowerShell window, or install Poppler and add its bin folder to PATH."
    )


def read_cover_metadata(pdf_path: Path) -> DocumentMetadata:
    result = subprocess.run(
        [_pdftotext(), "-f", "1", "-l", "1", "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = result.stdout.replace("\u00a0", " ")
    type_match = re.search(r"\(\s*([A-Z][A-Z0-9 -]*FLIP)\s*\)", text, re.IGNORECASE)
    if not type_match:
        raise ValueError("The document type could not be read from the PDF cover.")

    effective_match = re.search(
        r"EFFECTIVE\s+(\d{1,2}\s+[A-Z]{3}\s+\d{4})\s+TO\s+(\d{1,2}\s+[A-Z]{3}\s+\d{4})",
        text,
        re.IGNORECASE,
    )
    if not effective_match:
        raise ValueError("The effective date range could not be read from the PDF cover.")

    date_from, date_to = (part.upper() for part in effective_match.groups())
    month_year = date_to.split()
    effective_date = f"{month_year[1]}{month_year[2]}"
    return DocumentMetadata(
        document_type="FLIP",
        effective_date=effective_date,
        effective_from=date_from,
        effective_to=date_to,
    )
