from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dependency_setup import ensure_poppler

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DEFAULT_OUTPUT = PROJECT_DIR / "HEDEF" / "00-Initial"


@dataclass
class ChartInfo:
    pdf_page: int
    chart_no: str
    airport: str
    category: str
    procedure: str
    filename: str


def _normalise(value: str) -> str:
    return value.replace("\u00a0", " ").replace("\r", "\n")


def _airport_code(text: str) -> Optional[str]:
    matches = re.findall(r"\((LT[A-Z0-9]{2}|LC[A-Z0-9]{2}|ZZZZ)\)", text)
    return matches[-1] if matches else None


def _chart_number(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        values = re.findall(r"(?<!\d)(\d{1,2}-\d{1,2})(?!\d)", lines[1])
        if values:
            return values[0]
    marked = re.findall(r"(?<!\d)(\d{1,2}-\d{1,2})\s*SEP\s*2026\b", text)
    if marked:
        return marked[-1]
    repeated = re.findall(r"(?<!\d)(\d{1,2}-\d{1,2})\s+\1(?!\d)", text)
    if repeated:
        return repeated[-1]
    return None


def _procedure_and_category(text: str) -> tuple[Optional[str], Optional[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return None, None
    title = re.split(r"\s{2,}", lines[2].strip(), maxsplit=1)[0].strip().upper()
    title = re.sub(r"\s+", " ", title)
    title = title.replace("HI - TACAN", "HI TACAN").replace("LO - TACAN", "LO TACAN")
    title = title.replace("HI/LO TACAN", "HI/LO TACAN")

    if "AERODROME DIAGRAM" in title:
        return "AERODROME DIAGRAM", "Diagram"
    if "PARKING/DOCKING" in title:
        return "PARKING DOCKING", "Diagram"
    if "GROUND MOVEMENT" in title or "TAXI" in title:
        return title, "Taxi"
    if "ARRIVAL" in title or title.startswith("STAR"):
        return title, "Arrival"
    if title.startswith(("SID ", "TACAN SID", "VOR/DME SID", "RNAV SID", "RNAV (GNSS) SID", "HPMA SID")):
        return title, "Departure"
    if "RADAR INSTRUMENT APPROACH MINIMUM" in title:
        return title, "Diagram"
    if "RADAR MINIMUM ALTITUDES" in title or "RADAR MINIMUM ALTITUDE" in title:
        return "RADAR MINIMUM ALTITUDES", "Diagram"
    if title.startswith("TAC TO PAR"):
        return title, "Approach"
    if any(token in title for token in ("TACAN", "ILS", "LOC", "RNP", "VOR", "NDB")):
        return title, "Approach"

    compact = re.sub(r"\s+", " ", text.upper())
    fallback_rules = (
        (r"RNAV\s*\(GNSS\)\s+ARRIVAL(?:\s+RWY\s+\d{1,2}[LRC]?)?", "Arrival"),
        (r"RNAV\s*\(GNSS\)\s+SID(?:\s+RWY\s+\d{1,2}[LRC]?(?:\s*/\s*\d{1,2}[LRC]?)?)?", "Departure"),
    )
    for pattern, category in fallback_rules:
        found = re.search(pattern, compact)
        if found:
            return found.group(0), category
    return None, None


def inspect_pdf(path: Path, effective_date: str, document_type: str) -> tuple[list[ChartInfo], list[tuple[int, str]]]:
    charts: list[ChartInfo] = []
    unresolved: list[tuple[int, str]] = []
    date = re.sub(r"\s+", "", effective_date.strip().upper())
    doc_type = document_type.strip().upper()
    bundled_candidates = (
        APP_DIR / "tools" / "poppler" / "Library" / "bin" / "pdftotext.exe",
        APP_DIR / "tools" / "poppler" / "bin" / "pdftotext.exe",
        APP_DIR / "tools" / "poppler" / "bin" / "pdftotext",
    )
    pdftotext = next((str(candidate) for candidate in bundled_candidates if candidate.is_file()), None)
    pdftotext = pdftotext or shutil.which("pdftotext") or ensure_poppler()

    result = subprocess.run(
        [pdftotext, "-layout", str(path), "-"],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    page_texts = result.stdout.split("\f")
    raw_page_texts: Optional[list[str]] = None
    last_airport: Optional[str] = None
    for index, raw_text in enumerate(page_texts, start=1):
        text = _normalise(raw_text)
        if not text.strip():
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        first_line = lines[0].upper() if lines else ""
        if not (first_line.startswith("DİYAGRAM") or first_line.startswith("MEYDANLAR")):
            if raw_page_texts is None:
                raw_result = subprocess.run(
                    [pdftotext, "-raw", str(path), "-"],
                    check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
                )
                raw_page_texts = raw_result.stdout.split("\f")
            if index - 1 >= len(raw_page_texts):
                continue
            raw_candidate = _normalise(raw_page_texts[index - 1])
            if "DİYAGRAM" not in raw_candidate[-300:].upper():
                continue
            text = raw_candidate
        chart_no = _chart_number(text)
        airport = _airport_code(text) or last_airport
        if airport:
            last_airport = airport
        procedure, category = _procedure_and_category(text)
        if not (chart_no and airport and procedure and category):
            details = "; ".join(
                item for item in (
                    f"chart={chart_no}" if chart_no else "",
                    f"airport={airport}" if airport else "",
                    f"procedure={procedure}" if procedure else "",
                ) if item
            )
            unresolved.append((index, details or "Chart başlığı ayrıştırılamadı"))
            continue
        safe_procedure = re.sub(r"[\\/:*?\"<>|]", "", procedure)
        safe_procedure = re.sub(r"\s+", " ", safe_procedure).strip()
        filename = f"{airport}_{category}_{doc_type} {chart_no} {safe_procedure} to {date}.pdf"
        charts.append(ChartInfo(index, chart_no, airport, category, safe_procedure, filename))
    return charts, unresolved
