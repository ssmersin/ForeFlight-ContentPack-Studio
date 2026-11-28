from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pymupdf

from planview_roi import PlanviewRoi

# --------------------------------------------------------------------------
# Kural (kullanici tanimi):
#   * Tek referans ROI icindeki "variation-north" gostergesidir:
#     "VAR xx°E/W" yazisi + oklarin kesisiminde yil (ör. 2023).
#   * Gosterge bulunamazsa sayfa duz kabul edilir (North = 0), HICBIR baska
#     yaziya/cizgiye bakilarak dondurme yapilmaz.
#   * Tum koordinatlar "gorunen sayfa" (page.rect, /Rotate uygulanmis) uzayindadir.
# --------------------------------------------------------------------------

VAR_RE = re.compile(r"\bVAR(?:IATION)?\b\.?", re.I)
DEG_RE = re.compile(
    r"(\d{1,2}(?:[.,]\d+)?)\s*[°º˚]?\s*(?:(\d{1,2})\s*['′’]?)?\s*([EW])\b", re.I
)
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


@dataclass(frozen=True)
class ReferenceBox:
    """Referans bolge (gorunen sayfa uzayi, PDF point)."""
    x0: float
    y0: float
    x1: float
    y1: float
    angle_cw: int
    weight: float
    kind: str  # "north_arrow" | "arrow_axis"
    label: str = ""
    # arrow_axis icin: (x_tail, y_tail, x_head, y_head)
    line: Optional[tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class OrientationResult:
    rotation_cw: int
    method: str  # "variation_indicator" | "none"
    references: list[ReferenceBox]
    var_text: str = ""
    year: str = ""
    north_angle_deg: Optional[float] = None  # kuzeyin sayfa "yukari"sindan CW acisi
    residual_deg: Optional[float] = None
    var_check: str = "n/a"  # "OK" | "MISMATCH" | "n/a"
    basis: str = ""  # "arrow_N_label" | "arrow_pair" | "indicator_text" | "lat_labels"
    text_lines: int = 0  # ROI icinde okunan metin satiri sayisi (0 ise metin katmani yok/outline)
    apply_rotation: bool = True  # Arrival ilk turda sadece diagnostic; onay sonrasi etkinlestirilecek


# ------------------------------------------------------------------ geometri
def _resolve_roi_rect(page: pymupdf.Page, roi) -> pymupdf.Rect:
    if isinstance(roi, PlanviewRoi):
        rect = roi.to_rect()
    elif isinstance(roi, pymupdf.Rect):
        rect = roi
    else:
        rect = pymupdf.Rect(
            float(roi.get("x0", 0.0)), float(roi.get("y0", 0.0)),
            float(roi.get("x1", page.rect.width)), float(roi.get("y1", page.rect.height)),
        )
    clipped = pymupdf.Rect(rect) & page.rect
    return clipped if not clipped.is_empty else page.rect


def _overlap(r: pymupdf.Rect, w: pymupdf.Rect) -> bool:
    """Rect.intersects() bos (genisligi/yuksekligi 0) dikdortgenlerde False doner;
    tam dikey/yatay cizgiler (kuzey oklari!) bu yuzden kaciyordu."""
    return r.x0 <= w.x1 and r.x1 >= w.x0 and r.y0 <= w.y1 and r.y1 >= w.y0


def _vec_through(m: pymupdf.Matrix, dx: float, dy: float) -> tuple[float, float]:
    return m.a * dx + m.c * dy, m.b * dx + m.d * dy


def _angle_cw_from_up(vx: float, vy: float) -> float:
    """Sayfa koordinati (y asagi) vektorunun 'yukari'dan saat yonunde acisi, 0..360."""
    return math.degrees(math.atan2(vx, -vy)) % 360.0


def _snap_rotation(theta_deg: float) -> tuple[int, float]:
    """Kuzey theta (CW, yukariya gore) ise kuzeyi yukari getirecek CW dondurme + artik aci."""
    snapped = int(round(theta_deg / 90.0)) % 4 * 90
    residual = (theta_deg - snapped + 180.0) % 360.0 - 180.0
    return (360 - snapped) % 360, residual


def _pt_seg_dist(px, py, x0, y0, x1, y1) -> float:
    vx, vy = x1 - x0, y1 - y0
    c2 = vx * vx + vy * vy
    if c2 == 0:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * vx + (py - y0) * vy) / c2))
    return math.hypot(px - (x0 + t * vx), py - (y0 + t * vy))


# ------------------------------------------------------------------ metin
def _differences_map(doc: pymupdf.Document, font_xref: int) -> dict[int, str]:
    """Font /Encoding /Differences -> {kod: karakter}. FLIP PDF'lerinde yazilar ToUnicode'suz,
    glyph adlari 'MT78' gibi (MT + cp1252 kodu) oldugu icin metin bozuk gelir; buradan cozulur."""
    try:
        kind, val = doc.xref_get_key(font_xref, "Encoding")
        if kind == "xref":
            val = doc.xref_object(int(val.split()[0]))
        elif kind != "dict":
            return {}
        m = re.search(r"/Differences\s*\[(.*?)\]", val, re.S)
        if not m:
            return {}
        out: dict[int, str] = {}
        code = 0
        for tok in re.findall(r"/[^\s/\[\]]+|\d+", m.group(1)):
            if tok.startswith("/"):
                name = tok[1:]
                ch = ""
                mt = re.fullmatch(r"MT(\d{1,3})", name)
                if mt and int(mt.group(1)) < 256:
                    try:
                        ch = bytes([int(mt.group(1))]).decode("cp1252")
                    except Exception:
                        ch = ""
                else:
                    try:
                        from fontTools.agl import toUnicode
                        ch = toUnicode(name) or ""
                    except Exception:
                        ch = ""
                if ch:
                    out[code] = ch
                code += 1
            else:
                code = int(tok)
        return out
    except Exception:
        return {}


def _font_decoders(page: pymupdf.Page) -> dict[str, dict[int, str]]:
    doc = page.parent  # onbellek YOK: id(doc) yeniden kullanilinca baska PDF'in kodlamasi karisiyordu
    dec: dict[str, dict[int, str]] = {}
    for f in page.get_fonts(full=True):
        name = str(f[3]).split("+")[-1]
        mp = _differences_map(doc, f[0])
        if mp:
            dec.setdefault(name, {}).update(mp)
    return dec


def _span_text(span: dict, decoders: dict[str, dict[int, str]]) -> str:
    chars = [c["c"] for c in span.get("chars", [])]
    mp = decoders.get(str(span.get("font", "")).split("+")[-1])
    if mp and any(ord(c) < 32 for c in chars):  # bozuk (kontrol karakterli) metin -> coz
        return "".join(mp.get(ord(c), c) for c in chars)
    return "".join(chars)


def _collect_text_lines(page: pymupdf.Page, roi_rect: pymupdf.Rect) -> list[dict]:
    rm = page.rotation_matrix  # PyMuPDF metni dondurulmemis uzayda verir -> gorunen uzaya cevir
    decoders = _font_decoders(page)
    out: list[dict] = []
    for block in page.get_text("rawdict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(_span_text(s, decoders) for s in line.get("spans", [])).strip()
            if not text:
                continue
            rect = pymupdf.Rect(line["bbox"]) * rm
            rect.normalize()
            c = pymupdf.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
            if not roi_rect.contains(c):
                continue
            dx, dy = line.get("dir", (1.0, 0.0))
            dx, dy = _vec_through(rm, float(dx), float(dy))
            out.append({
                "rect": rect,
                "text": " ".join(text.split()),
                "nchars": len("".join(text.split())),
                "dir": (dx, dy),
                # yazinin "yukari" vektoru (okuma yonunun 90° CCW'si)
                "up": (dy, -dx),
            })
    return out


def _parse_var_value(text: str) -> Optional[float]:
    m = DEG_RE.search(text)
    if not m:
        return None
    deg = float(m.group(1).replace(",", "."))
    if m.group(2):
        deg += float(m.group(2)) / 60.0
    return deg if m.group(3).upper() == "E" else -deg


def _find_indicator_text(lines: list[dict]) -> Optional[dict]:
    best = None
    for v in lines:
        if not VAR_RE.search(v["text"]):
            continue
        vc = pymupdf.Point((v["rect"].x0 + v["rect"].x1) / 2, (v["rect"].y0 + v["rect"].y1) / 2)
        rect = pymupdf.Rect(v["rect"])
        value = _parse_var_value(v["text"])
        year_m = YEAR_RE.search(v["text"])
        year = year_m.group(1) if year_m else ""
        score = 10.0
        members = [v]
        for o in lines:
            if o is v:
                continue
            oc = pymupdf.Point((o["rect"].x0 + o["rect"].x1) / 2, (o["rect"].y0 + o["rect"].y1) / 2)
            if math.hypot(oc.x - vc.x, oc.y - vc.y) > 80.0:
                continue
            t = o["text"]
            got = False
            if value is None and _parse_var_value(t) is not None and len(t) <= 14:
                value = _parse_var_value(t)
                got = True
            ym = YEAR_RE.search(t)
            if not year and ym and len(t) <= 14:
                year = ym.group(1)
                got = True
            if got:
                rect |= o["rect"]
                members.append(o)
        if value is not None:
            score += 6.0
        if year:
            score += 6.0
        if value is None and not year:
            continue  # "VAR" tek basina yeterli degil
        if best is None or score > best["score"]:
            best = {"score": score, "rect": rect, "value": value, "year": year,
                    "anchor": v, "members": members,
                    "text": " | ".join(m["text"] for m in members)}
    return best


# ------------------------------------------------------------------ oklar
def _collect_segments(page: pymupdf.Page, window: pymupdf.Rect) -> tuple[list, list]:
    """Pencere icindeki duz cizgiler (gorunen uzayda) ve kucuk dolu sekiller."""
    rm = page.rotation_matrix
    segs: list[tuple[float, float, float, float]] = []
    heads: list[pymupdf.Rect] = []
    for d in page.get_drawings():
        r = pymupdf.Rect(d.get("rect", (0, 0, 0, 0))) * rm
        r.normalize()
        if not _overlap(r, window):
            continue
        items = d.get("items", [])
        if d.get("fill") is not None and 3 <= len(items) <= 5 and max(r.width, r.height) <= 24:
            heads.append(r)
        for it in items:
            if it[0] != "l":
                continue
            p1, p2 = it[1] * rm, it[2] * rm
            s = (p1.x, p1.y, p2.x, p2.y)
            if _overlap(pymupdf.Rect(min(s[0], s[2]), min(s[1], s[3]), max(s[0], s[2]), max(s[1], s[3])), window):
                segs.append(s)
    return segs, heads


def _head_score(px: float, py: float, segs, heads, own) -> float:
    score = 0.0
    for s in segs:
        if s is own:
            continue
        ln = math.hypot(s[2] - s[0], s[3] - s[1])
        if not (2.0 <= ln <= 22.0):
            continue
        if math.hypot(s[0] - px, s[1] - py) <= 2.5 or math.hypot(s[2] - px, s[3] - py) <= 2.5:
            score += 1.0
    for h in heads:
        if pymupdf.Rect(h.x0 - 2.5, h.y0 - 2.5, h.x1 + 2.5, h.y1 + 2.5).contains(pymupdf.Point(px, py)):
            score += 3.0
    return score


LAT_RE = re.compile(r"^\d{1,2}\s*[°º˚]\s*\d{1,2}\s*['′’]\s*(?:\d{1,2}\s*(?:''|\"|″|’’)?)?\s*[NS]$", re.I)


def _find_arrow_pair(page, region: pymupdf.Rect, n_letters: list, wide: bool = False) -> Optional[dict]:
    """Ortak kuyruktan cikan iki oku (true N + magnetic N) bulur. Metin gerekmez (outline yazilarda da calisir).
    Semboldeki aci stilizedir (VAR=4° iken oklar ~45° olabilir) -> aci VAR ile karsilastirilmaz."""
    segs, heads = _collect_segments(page, region)
    cands = [s for s in segs if 10.0 <= math.hypot(s[2] - s[0], s[3] - s[1]) <= 80.0]
    arrows = []
    for s in cands:
        ln = math.hypot(s[2] - s[0], s[3] - s[1])
        ha = _head_score(s[0], s[1], segs, heads, s)
        hb = _head_score(s[2], s[3], segs, heads, s)
        if max(ha, hb) < 2.0 or abs(ha - hb) < 1.0:
            continue
        head, tail = ((s[2], s[3]), (s[0], s[1])) if hb > ha else ((s[0], s[1]), (s[2], s[3]))
        arrows.append({"seg": s, "len": ln, "tail": tail, "head": head, "head_ev": max(ha, hb),
                       "vec": (head[0] - tail[0], head[1] - tail[1])})
    best = None
    for i, a in enumerate(arrows):
        for b in arrows[i + 1:]:
            if math.hypot(a["tail"][0] - b["tail"][0], a["tail"][1] - b["tail"][1]) > 5.0:
                continue
            ang = abs((_angle_cw_from_up(*a["vec"]) - _angle_cw_from_up(*b["vec"]) + 180.0) % 360.0 - 180.0)
            ratio = max(a["len"], b["len"]) / min(a["len"], b["len"])
            if wide:
                # FLIP VAR-north simgesi her chartta ayni: oklar ~19.0 ve ~21.9 pt, aralari ~39.8°.
                # Metin okunamasa bile bu parmak izi (sekil) yeterince ozgun.
                if not (34.0 <= ang <= 46.0 and 1.05 <= ratio <= 1.30
                        and 14.0 <= min(a["len"], b["len"]) and max(a["len"], b["len"]) <= 30.0):
                    continue
            else:
                if not (25.0 <= ang <= 60.0 and ratio <= 1.6):
                    continue
            score = a["head_ev"] + b["head_ev"] - abs(a["len"] - b["len"]) * 0.05
            if best is None or score > best[0]:
                best = (score, a, b)
    if best is None:
        return None
    _, a, b = best
    true_arrow = None
    if n_letters:  # "N" harfine en yakin ok ucu = true north
        def d_to_n(ar):
            return min(math.hypot(ar["head"][0] - (n["rect"].x0 + n["rect"].x1) / 2,
                                  ar["head"][1] - (n["rect"].y0 + n["rect"].y1) / 2) for n in n_letters)
        da, db = d_to_n(a), d_to_n(b)
        if min(da, db) <= 45.0 and abs(da - db) >= 4.0:
            true_arrow = a if da < db else b
    if true_arrow is not None:
        return {"vec": true_arrow["vec"], "basis": "arrow_N_label", "pair": (a, b)}
    # N harfi okunamadi: simgede gercek kuzey oku KISA olandir (19.0 pt; manyetik 21.9 pt, 39.8° CW)
    if abs(a["len"] - b["len"]) / max(a["len"], b["len"]) >= 0.08:
        short = a if a["len"] < b["len"] else b
        return {"vec": short["vec"], "basis": "arrow_short", "pair": (a, b)}
    va, vb = a["vec"], b["vec"]
    na, nb = math.hypot(*va), math.hypot(*vb)
    bis = (va[0] / na + vb[0] / nb, va[1] / na + vb[1] / nb)
    return {"vec": bis, "basis": "arrow_pair", "pair": (a, b)}


def _lat_label_vote(lines: list[dict]) -> Optional[dict]:
    """Enlem etiketleri (35°10'00'' N): yazinin 'yukari'si kuzeyi gosterir.
    Kural: tek etiket/eksen varsa kullanilir; 2+ etiket birbiriyle celisiyorsa dikkate alinmaz."""
    found = []
    for ln in lines:
        if LAT_RE.match(ln["text"].strip()):
            rot, resid = _snap_rotation(_angle_cw_from_up(*ln["up"]))
            if abs(resid) <= 8.0:
                found.append((rot, ln["rect"]))
    if not found:
        return None
    rots = {r for r, _ in found}
    if len(rots) > 1:
        return {"rot": None, "n": len(found), "boxes": [r for _, r in found], "ignored": True}
    return {"rot": rots.pop(), "n": len(found), "boxes": [r for _, r in found], "ignored": False}


def _pair_has_text_signature(pair: dict, lines: list[dict]) -> bool:
    """Metin okunamasa bile gercek gosterge: oka yakin, ayni yonlu >=3 kisa satir ve bir tek-glifli 'N'."""
    pts = [p for a in pair["pair"] for p in (a["tail"], a["head"])]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    near = [l for l in lines
            if math.hypot((l["rect"].x0 + l["rect"].x1) / 2 - cx, (l["rect"].y0 + l["rect"].y1) / 2 - cy) <= 50.0
            and l["nchars"] <= 10]
    if len(near) < 3 or not any(l["nchars"] == 1 for l in near):
        return False
    base = _angle_cw_from_up(*near[0]["up"])
    return sum(1 for l in near if abs((_angle_cw_from_up(*l["up"]) - base + 180) % 360 - 180) <= 8.0) >= 3



RMA_LAT_RE = re.compile(
    r"^\s*(\d{1,2})(?:\s*[°º˚]\s*|\s*[-–]\s*)(\d{1,2})?"
    r"(?:\s*['′’])?\s*(?:[°º˚])?\s*N?\s*$", re.I
)


# ------------------------------------------------------------------ ortak chart referanslari: MSA + renk/irtifa skalasi
ALTITUDE_FT_RE = re.compile(r"^\s*(\d{1,5})\s*(?:FT|FEET)\s*$", re.I)
ALTITUDE_VALUE_RE = re.compile(r"^\s*(\d{3,5})\s*$")
DISTANCE_25NM_RE = re.compile(r"\b25\s*(?:NM|N\.?M\.?)\b", re.I)
MSA_WORD_RE = re.compile(r"\bMSA\b", re.I)


def _line_rotation(line: dict) -> Optional[int]:
    rot, residual = _snap_rotation(_angle_cw_from_up(*line["up"]))
    return rot if abs(residual) <= 8.0 else None


def _rect_inside(a: pymupdf.Rect, b: pymupdf.Rect, pad: float = 0.0) -> bool:
    return (b.x0 - pad <= a.x0 and b.y0 - pad <= a.y0
            and b.x1 + pad >= a.x1 and b.y1 + pad >= a.y1)


def _filled_color_rectangles(page: pymupdf.Page, roi: pymupdf.Rect) -> list[pymupdf.Rect]:
    """Return swatches and compact filled legend bars from vector chart legends."""
    rm = page.rotation_matrix
    found = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if fill is None:
            continue
        for item in drawing.get("items", []):
            if item[0] != "re":
                continue
            rect = pymupdf.Rect(item[1]) * rm
            rect.normalize()
            if not roi.contains(pymupdf.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)):
                continue
            if 2.0 <= rect.width <= 28.0 and 2.0 <= rect.height <= 28.0:
                found.append(rect)
        # Some FLIP legends are drawn as one compact filled path, not separate
        # rectangle primitives. Keep only narrow, legend-sized filled drawings;
        # the nearby altitude labels are the second-stage discriminator.
        rect = pymupdf.Rect(drawing.get("rect", (0, 0, 0, 0))) * rm
        rect.normalize()
        if not roi.contains(pymupdf.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)):
            continue
        ratio = rect.width / max(rect.height, 0.01)
        if 8.0 <= rect.width <= 42.0 and 14.0 <= rect.height <= 140.0 and 0.15 <= ratio <= 1.5:
            found.append(rect)
    # Deduplicate overlapping rectangles from the same compound legend artwork.
    unique = []
    for rect in found:
        if not any(_overlap(rect, old) and abs(rect.width - old.width) < 1.5
                   and abs(rect.height - old.height) < 1.5 for old in unique):
            unique.append(rect)
    return unique


def _group_color_swatches(rects: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Group adjacent swatches into a single horizontal or vertical legend bar."""
    groups: list[list[pymupdf.Rect]] = []
    for rect in sorted(rects, key=lambda r: (r.y0, r.x0)):
        placed = False
        for group in groups:
            g = group[0]
            same_row = abs((rect.y0 + rect.y1 - g.y0 - g.y1) / 2) <= max(3.0, min(rect.height, g.height) * 0.45)
            same_col = abs((rect.x0 + rect.x1 - g.x0 - g.x1) / 2) <= max(3.0, min(rect.width, g.width) * 0.45)
            adjacent = any(
                (abs(rect.x0 - other.x1) <= 4.0 or abs(other.x0 - rect.x1) <= 4.0)
                and abs((rect.y0 + rect.y1 - other.y0 - other.y1) / 2) <= 4.0
                or (abs(rect.y0 - other.y1) <= 4.0 or abs(other.y0 - rect.y1) <= 4.0)
                and abs((rect.x0 + rect.x1 - other.x0 - other.x1) / 2) <= 4.0
                for other in group
            )
            if adjacent and (same_row or same_col):
                group.append(rect)
                placed = True
                break
        if not placed:
            groups.append([rect])

    result = []
    for group in groups:
        if len(group) < 4:
            continue
        result.append(pymupdf.Rect(
            min(r.x0 for r in group), min(r.y0 for r in group),
            max(r.x1 for r in group), max(r.y1 for r in group),
        ))
    return result


def _detect_color_altitude_scale(
    page: pymupdf.Page, lines: list[dict], roi: pymupdf.Rect
) -> Optional[dict]:
    swatches = _filled_color_rectangles(page, roi)
    groups = _group_color_swatches(swatches)
    # Include a single compound-path legend bar as a candidate as well.
    for rect in swatches:
        ratio = rect.width / max(rect.height, 0.01)
        if 8.0 <= rect.width <= 42.0 and 14.0 <= rect.height <= 140.0 and 0.15 <= ratio <= 1.5:
            if not any(_overlap(rect, group) and abs(rect.width - group.width) < 1.5
                       and abs(rect.height - group.height) < 1.5 for group in groups):
                groups.append(rect)
    best = None
    for group in groups:
        pad = max(18.0, min(group.width, group.height) * 1.8)
        nearby = []
        vertical = group.height >= group.width
        major = group.height if vertical else group.width
        minor = group.width if vertical else group.height
        for line in lines:
            if not ALTITUDE_FT_RE.match(line["text"]):
                continue
            r = line["rect"]
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            if vertical:
                along = cy - group.y0
                cross = min(abs(cx - group.x0), abs(cx - group.x1))
                in_major = group.y0 - pad <= cy <= group.y1 + pad
            else:
                along = cx - group.x0
                cross = min(abs(cy - group.y0), abs(cy - group.y1))
                in_major = group.x0 - pad <= cx <= group.x1 + pad
            # Altitude labels must sit beside the legend and be distributed
            # along its long axis, not merely happen to be nearby.
            if in_major and cross <= max(12.0, minor * 1.8):
                rot = _line_rotation(line)
                if rot is not None:
                    nearby.append((rot, line, along))
        if len(nearby) < 3:
            continue
        positions = sorted(item[2] for item in nearby)
        if positions[-1] - positions[0] < major * 0.50:
            continue
        rotations = {rot for rot, _, _ in nearby}
        if len(rotations) != 1:
            continue
        score = len(nearby) * 3 + min(major, 100.0)
        if best is None or score > best["score"]:
            best = {"rot": rotations.pop(), "rect": group, "labels": [ln for _, ln, _ in nearby],
                    "score": score, "swatches": swatches}
    return best


def _reference_boxes_for_scale(scale: dict) -> list[ReferenceBox]:
    refs: list[ReferenceBox] = []
    r = scale["rect"]
    refs.append(ReferenceBox(r.x0, r.y0, r.x1, r.y1, scale["rot"], 8.0,
                             "altitude_scale", "COLOR / ALTITUDE SCALE"))
    for line in scale["labels"]:
        r = line["rect"]
        refs.append(ReferenceBox(r.x0, r.y0, r.x1, r.y1, scale["rot"], 6.0,
                                 "altitude_scale_label", line["text"]))
    for swatch in scale.get("swatches", []):
        if _overlap(swatch, scale["rect"]):
            refs.append(ReferenceBox(swatch.x0, swatch.y0, swatch.x1, swatch.y1,
                                     scale["rot"], 2.0, "altitude_swatch", "COLOR SWATCH"))
    return refs


def _detect_non_route_text_vote(
    page: pymupdf.Page, lines: list[dict], roi: pymupdf.Rect,
    excluded_rects: Optional[list[pymupdf.Rect]] = None,
) -> Optional[dict]:
    """Use distributed, map-fixed text only; reject labels close to long route-like vectors."""
    segments, _ = _collect_segments(page, roi)
    route_like = []
    for x0, y0, x1, y1 in segments:
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 38.0:
            continue
        # Ignore the planview frame itself.
        if ((abs(x0 - roi.x0) < 4 and abs(x1 - roi.x0) < 4)
                or (abs(x0 - roi.x1) < 4 and abs(x1 - roi.x1) < 4)
                or (abs(y0 - roi.y0) < 4 and abs(y1 - roi.y0) < 4)
                or (abs(y0 - roi.y1) < 4 and abs(y1 - roi.y1) < 4)):
            continue
        route_like.append((x0, y0, x1, y1))

    excluded_rects = excluded_rects or []
    candidates = []
    for line in lines:
        text = line["text"].strip()
        compact = re.sub(r"\s+", "", text)
        if len(compact) < 3 or ALTITUDE_FT_RE.match(text) or ALTITUDE_VALUE_RE.match(text):
            continue
        if re.fullmatch(r"[\d./\-+]+", compact):
            continue
        rect = line["rect"]
        center = ((rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0)
        if any(_overlap(rect, ex) for ex in excluded_rects):
            continue
        # Conservative clearance: labels near long straight route legs are never votes.
        if any(_pt_seg_dist(center[0], center[1], *seg) < 20.0 for seg in route_like):
            continue
        rot = _line_rotation(line)
        if rot is None:
            continue
        candidates.append((rot, line))

    if len(candidates) < 5:
        return None
    counts: dict[int, int] = {}
    for rot, _ in candidates:
        counts[rot] = counts.get(rot, 0) + 1
    rot, count = max(counts.items(), key=lambda item: item[1])
    if count < 5 or count / len(candidates) < 0.70:
        return {"ignored": True, "n": len(candidates),
                "boxes": [line["rect"] for _, line in candidates]}
    selected = [line for candidate_rot, line in candidates if candidate_rot == rot]
    # Require the text evidence to be spatially distributed, not one route annotation cluster.
    centers = [((ln["rect"].x0 + ln["rect"].x1) / 2,
                (ln["rect"].y0 + ln["rect"].y1) / 2) for ln in selected]
    spread_x = max(p[0] for p in centers) - min(p[0] for p in centers)
    spread_y = max(p[1] for p in centers) - min(p[1] for p in centers)
    if spread_x < roi.width * 0.12 and spread_y < roi.height * 0.12:
        return None
    return {"rot": rot, "n": len(selected), "boxes": [ln["rect"] for ln in selected],
            "labels": selected, "ignored": False}


def _departure_orientation(page: pymupdf.Page, roi: pymupdf.Rect) -> OrientationResult:
    lines = _collect_text_lines(page, roi)
    scale = _detect_color_altitude_scale(page, lines, roi)
    scale_rects = [scale["rect"]] if scale else []
    text_vote = _detect_non_route_text_vote(page, lines, roi, excluded_rects=scale_rects)
    refs: list[ReferenceBox] = []
    votes: list[tuple[str, int]] = []

    if scale:
        refs.extend(_reference_boxes_for_scale(scale))
        votes.append(("SCALE", scale["rot"]))
    if text_vote:
        for rect in text_vote["boxes"]:
            refs.append(ReferenceBox(rect.x0, rect.y0, rect.x1, rect.y1,
                                     text_vote.get("rot", 0), 4.0, "departure_text", "NON-ROUTE TEXT"))
        if not text_vote["ignored"]:
            votes.append(("MAP_TEXT", text_vote["rot"]))

    if not votes:
        state = "DEPARTURE_TEXT_CONFLICT" if text_vote and text_vote["ignored"] else "DEPARTURE_REFERENCES_NOT_FOUND"
        return OrientationResult(0, "none", refs, var_check=state,
                                 basis="color_altitude+non_route_text", text_lines=len(lines),
                                 apply_rotation=False)
    rotations = {rot for _, rot in votes}
    if len(rotations) != 1:
        detail = ",".join(f"{name}={rot}" for name, rot in votes)
        return OrientationResult(0, "none", refs, var_check=f"DEPARTURE_REFERENCE_CONFLICT({detail})",
                                 basis="color_altitude+non_route_text", text_lines=len(lines),
                                 apply_rotation=False)
    rotation = rotations.pop()
    basis = "+".join(name.lower() for name, _ in votes)
    check = f"DEPARTURE_AGREE({basis})" if len(votes) > 1 else f"DEPARTURE_SINGLE({basis})"
    return OrientationResult(rotation, "departure_references", refs, var_check=check,
                             basis=basis, text_lines=len(lines), apply_rotation=False)


def _detect_msa_reference(
    page: pymupdf.Page, lines: list[dict], roi: pymupdf.Rect
) -> Optional[dict]:
    """Find a circular 25 NM MSA graphic and its central altitude label."""
    rm = page.rotation_matrix
    circles = []
    for drawing in page.get_drawings():
        items = drawing.get("items", [])
        if not any(item[0] in ("c", "qu") for item in items):
            continue
        rect = pymupdf.Rect(drawing.get("rect", (0, 0, 0, 0))) * rm
        rect.normalize()
        if not roi.contains(pymupdf.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)):
            continue
        if rect.width < 35.0 or rect.height < 35.0 or rect.width > 190.0 or rect.height > 190.0:
            continue
        if not 0.78 <= rect.width / max(rect.height, 0.01) <= 1.28:
            continue
        circles.append(rect)

    best = None
    for circle in circles:
        nearby_text = []
        altitude_lines = []
        for line in lines:
            r = line["rect"]
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            if circle.x0 - 12 <= cx <= circle.x1 + 12 and circle.y0 - 12 <= cy <= circle.y1 + 12:
                nearby_text.append(line)
                if ALTITUDE_VALUE_RE.match(line["text"]) and circle.x0 + circle.width * 0.25 <= cx <= circle.x0 + circle.width * 0.75 and circle.y0 + circle.height * 0.25 <= cy <= circle.y0 + circle.height * 0.75:
                    altitude_lines.append(line)
        has_range = any(DISTANCE_25NM_RE.search(line["text"]) for line in lines
                        if math.hypot((line["rect"].x0 + line["rect"].x1) / 2 - (circle.x0 + circle.x1) / 2,
                                      (line["rect"].y0 + line["rect"].y1) / 2 - (circle.y0 + circle.y1) / 2) <= 100.0)
        has_msa = any(MSA_WORD_RE.search(line["text"]) for line in nearby_text)
        if not altitude_lines or not (has_range or has_msa):
            continue
        valid = [(rot, line) for line in altitude_lines if (rot := _line_rotation(line)) is not None]
        if not valid:
            continue
        rotations = {rot for rot, _ in valid}
        if len(rotations) != 1:
            continue
        score = 8 + len(valid) * 2 + (4 if has_range else 0) + (4 if has_msa else 0)
        if best is None or score > best["score"]:
            best = {"rot": rotations.pop(), "rect": circle, "labels": [ln for _, ln in valid],
                    "score": score, "nearby": nearby_text}
    return best


def _detect_arrival_orientation(page: pymupdf.Page, roi: pymupdf.Rect) -> OrientationResult:
    lines = _collect_text_lines(page, roi)
    msa = _detect_msa_reference(page, lines, roi)
    scale = _detect_color_altitude_scale(page, lines, roi)
    refs: list[ReferenceBox] = []
    votes: list[tuple[str, int, float]] = []

    if msa:
        r = msa["rect"]
        refs.append(ReferenceBox(r.x0, r.y0, r.x1, r.y1, msa["rot"], 8.0, "msa_circle", "MSA 25 NM"))
        for line in msa["labels"]:
            r = line["rect"]
            refs.append(ReferenceBox(r.x0, r.y0, r.x1, r.y1, msa["rot"], 6.0, "msa_label", line["text"]))
        votes.append(("MSA", msa["rot"], msa["score"]))

    if scale:
        refs.extend(_reference_boxes_for_scale(scale))
        votes.append(("SCALE", scale["rot"], scale["score"]))

    if not votes:
        return OrientationResult(0, "none", refs, var_check="ARRIVAL_REFERENCES_NOT_FOUND",
                                 basis="msa+color_altitude", text_lines=len(lines))
    rotations = {rot for _, rot, _ in votes}
    if len(rotations) != 1:
        details = ",".join(f"{name}={rot}" for name, rot, _ in votes)
        return OrientationResult(0, "none", refs, var_check=f"ARRIVAL_REFERENCE_CONFLICT({details})",
                                 basis="msa+color_altitude", text_lines=len(lines))
    rotation = rotations.pop()
    basis = "+".join(name.lower() for name, _, _ in votes)
    check = f"ARRIVAL_AGREE({basis})" if len(votes) > 1 else f"ARRIVAL_SINGLE({basis})"
    return OrientationResult(rotation, "arrival_references", refs, var_check=check,
                             basis=basis, text_lines=len(lines), apply_rotation=False)




APPROACH_COORD_RE = re.compile(
    r"\b\d{1,2}[- ]\d{2}(?:[- ]\d{2}(?:\.\d+)?)?\s*[NS]\b.*"
    r"\b\d{2,3}[- ]\d{2}(?:[- ]\d{2}(?:\.\d+)?)?\s*[EW]\b", re.I
)
APPROACH_LAT_PART_RE = re.compile(
    r"\b\d{1,2}\s*[°º˚-]\s*\d{1,2}(?:[.,]\d+)?(?:\s*['′’])?\s*N\b", re.I
)
APPROACH_LON_PART_RE = re.compile(
    r"\b\d{2,3}\s*[°º˚-]\s*\d{1,2}(?:[.,]\d+)?(?:\s*['′’])?\s*[EW]\b", re.I
)
APPROACH_ANCHOR_RE = re.compile(
    r"\b(?:EMG\s*SAFE\s*ALT|EMERG(?:ENCY)?\s*SAFE\s*ALT|CAUTION|IAF|IF|FAF|MAPt|S/S|HOLDING|HOLD|HLDG)\b", re.I
)


def _collect_roi_rectangles(page: pymupdf.Page, roi: pymupdf.Rect) -> list[pymupdf.Rect]:
    """Collect compact vector information boxes fully associated with the planview ROI."""
    rm = page.rotation_matrix
    candidates: list[pymupdf.Rect] = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if item[0] != "re":
                continue
            rect = pymupdf.Rect(item[1]) * rm
            rect.normalize()
            if not roi.contains(pymupdf.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)):
                continue
            if not (18.0 <= rect.width <= 260.0 and 8.0 <= rect.height <= 140.0):
                continue
            if rect.width * rect.height > 25000.0:
                continue
            if not 0.20 <= rect.width / max(rect.height, 0.01) <= 18.0:
                continue
            if not any(
                abs(rect.x0 - old.x0) < 1.5 and abs(rect.y0 - old.y0) < 1.5
                and abs(rect.x1 - old.x1) < 1.5 and abs(rect.y1 - old.y1) < 1.5
                for old in candidates
            ):
                candidates.append(rect)
    return candidates


def _collect_roi_edge_ticks(page: pymupdf.Page, roi: pymupdf.Rect) -> list[ReferenceBox]:
    """Find short grid/tick marks touching any ROI edge, independent of their page direction.

    These marks are retained as georeferencing features only; they never vote on chart rotation.
    """
    segments, _ = _collect_segments(page, roi)
    rm = page.rotation_matrix
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if item[0] != "re":
                continue
            rect = pymupdf.Rect(item[1]) * rm
            rect.normalize()
            if not roi.intersects(rect):
                continue
            if 1.5 <= rect.width <= 20.0 and rect.height <= 2.5:
                segments.append((rect.x0, (rect.y0 + rect.y1) / 2.0,
                                 rect.x1, (rect.y0 + rect.y1) / 2.0))
            elif 1.5 <= rect.height <= 20.0 and rect.width <= 2.5:
                segments.append(((rect.x0 + rect.x1) / 2.0, rect.y0,
                                 (rect.x0 + rect.x1) / 2.0, rect.y1))

    edge_tol = max(4.0, min(roi.width, roi.height) * 0.008)
    found: list[ReferenceBox] = []
    seen: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in segments:
        length = math.hypot(x1 - x0, y1 - y0)
        if not 1.5 <= length <= 20.0:
            continue
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        edge_options = (
            ("LEFT", abs(mx - roi.x0), abs(x1 - x0) >= abs(y1 - y0) * 1.5),
            ("RIGHT", abs(mx - roi.x1), abs(x1 - x0) >= abs(y1 - y0) * 1.5),
            ("TOP", abs(my - roi.y0), abs(y1 - y0) >= abs(x1 - x0) * 1.5),
            ("BOTTOM", abs(my - roi.y1), abs(y1 - y0) >= abs(x1 - x0) * 1.5),
        )
        edge, distance, perpendicular = min(edge_options, key=lambda item: item[1])
        if distance > edge_tol or not perpendicular:
            continue
        if any(math.hypot(mx - ((a + c) / 2.0), my - ((b + d) / 2.0)) < 2.0
               for a, b, c, d in seen):
            continue
        seen.append((x0, y0, x1, y1))
        found.append(ReferenceBox(
            min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
            0, 0.0, "edge_tick", edge, (x0, y0, x1, y1),
        ))
    return found


def _approach_orientation(page: pymupdf.Page, roi: pymupdf.Rect) -> OrientationResult:
    lines = _collect_text_lines(page, roi)
    segments, _ = _collect_segments(page, roi)
    route_like = [
        seg for seg in segments
        if math.hypot(seg[2] - seg[0], seg[3] - seg[1]) >= 38.0
        and not (
            (abs(seg[0] - roi.x0) < 4 and abs(seg[2] - roi.x0) < 4)
            or (abs(seg[0] - roi.x1) < 4 and abs(seg[2] - roi.x1) < 4)
            or (abs(seg[1] - roi.y0) < 4 and abs(seg[3] - roi.y0) < 4)
            or (abs(seg[1] - roi.y1) < 4 and abs(seg[3] - roi.y1) < 4)
        )
    ]

    refs: list[ReferenceBox] = _collect_roi_edge_ticks(page, roi)
    votes: dict[str, list[int]] = {
        "EMG_SAFE_ALT": [], "CAUTION": [], "HOLDING": [],
        "IAF_COORD": [], "AIRPORT_INFO": [], "TERRAIN": [],
    }
    used_lines: set[int] = set()

    def add_box_vote(rect: pymupdf.Rect, members: list[tuple[int, dict]], text: str,
                     family: str, kind: str, weight: float) -> None:
        if not members:
            return
        rotations: dict[int, int] = {}
        for _, member in members:
            rot = _line_rotation(member)
            if rot is None:
                continue
            center_x = (member["rect"].x0 + member["rect"].x1) / 2.0
            center_y = (member["rect"].y0 + member["rect"].y1) / 2.0
            if any(_pt_seg_dist(center_x, center_y, *seg) < 20.0 for seg in route_like):
                continue
            rotations[rot] = rotations.get(rot, 0) + 1
        if rotations:
            rotation, count = max(rotations.items(), key=lambda item: item[1])
            if count / sum(rotations.values()) >= 0.75:
                votes[family].append(rotation)
        refs.append(ReferenceBox(rect.x0, rect.y0, rect.x1, rect.y1, 0, weight, kind, text))
        used_lines.update(index for index, _ in members)

    # Prefer complete vector boxes: their border is more useful to the user than
    # a tight rectangle around one extracted text line.
    for box in _collect_roi_rectangles(page, roi):
        members = []
        for idx, line in enumerate(lines):
            r = line["rect"]
            cx, cy = (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0
            if box.x0 - 2.0 <= cx <= box.x1 + 2.0 and box.y0 - 2.0 <= cy <= box.y1 + 2.0:
                members.append((idx, line))
        if not members:
            continue
        text = " ".join(line["text"] for _, line in members)
        compact = re.sub(r"\s+", " ", text).upper()
        lat_found = bool(APPROACH_LAT_PART_RE.search(compact))
        lon_found = bool(APPROACH_LON_PART_RE.search(compact))
        if re.search(r"\b(?:EMG|EMERG(?:ENCY)?)\s*SAFE\s*ALT\b", compact, re.I):
            add_box_vote(box, members, text, "EMG_SAFE_ALT", "approach_emg", 9.0)
        if re.search(r"\bCAUTION\b", compact, re.I):
            add_box_vote(box, members, text, "CAUTION", "approach_box_text", 7.0)
        if re.search(r"\b(?:HOLDING|HOLD|HLDG|S/S)\b", compact, re.I):
            add_box_vote(box, members, text, "HOLDING", "approach_holding", 7.0)
        if re.search(r"\b(?:IAF|IF|FAF|MAPt)\b", compact, re.I) or APPROACH_COORD_RE.search(compact):
            add_box_vote(box, members, text, "IAF_COORD", "approach_iaf", 8.0)
        elif lat_found and lon_found:
            add_box_vote(box, members, text, "AIRPORT_INFO", "approach_airport_info", 8.0)

    # Text-only fallback covers boxes whose borders are rasterized or absent.
    for idx, line in enumerate(lines):
        if idx in used_lines:
            continue
        text = line["text"].strip()
        compact = re.sub(r"\s+", " ", text).upper()
        rect = line["rect"]
        if re.search(r"\b(?:EMG|EMERG(?:ENCY)?)\s*SAFE\s*ALT\b", compact, re.I):
            kind, family, weight = "approach_emg", "EMG_SAFE_ALT", 9.0
        elif re.search(r"\bCAUTION\b", compact, re.I):
            kind, family, weight = "approach_box_text", "CAUTION", 7.0
        elif re.search(r"\b(?:HOLDING|HOLD|HLDG|S/S)\b", compact, re.I):
            kind, family, weight = "approach_holding", "HOLDING", 7.0
        elif re.search(r"\b(?:IAF|IF|FAF|MAPt)\b", compact, re.I) or APPROACH_COORD_RE.search(compact):
            kind, family, weight = "approach_iaf", "IAF_COORD", 8.0
        else:
            continue
        cx, cy = (rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0
        rot = _line_rotation(line)
        if rot is not None and not any(_pt_seg_dist(cx, cy, *seg) < 20.0 for seg in route_like):
            votes[family].append(rot)
        refs.append(ReferenceBox(rect.x0, rect.y0, rect.x1, rect.y1, 0, weight, kind, text))
        used_lines.add(idx)

    # Coordinate pairs may be extracted as separate text lines in an airport
    # information box. Pair them spatially without depending on line order.
    lat_lines = [(i, line) for i, line in enumerate(lines)
                 if i not in used_lines and APPROACH_LAT_PART_RE.search(line["text"])]
    lon_lines = [(i, line) for i, line in enumerate(lines)
                 if i not in used_lines and APPROACH_LON_PART_RE.search(line["text"])]
    for lat_idx, lat_line in lat_lines:
        lx = (lat_line["rect"].x0 + lat_line["rect"].x1) / 2.0
        ly = (lat_line["rect"].y0 + lat_line["rect"].y1) / 2.0
        matches = []
        for lon_idx, lon_line in lon_lines:
            ox = (lon_line["rect"].x0 + lon_line["rect"].x1) / 2.0
            oy = (lon_line["rect"].y0 + lon_line["rect"].y1) / 2.0
            if math.hypot(ox - lx, oy - ly) <= 55.0:
                matches.append((lon_idx, lon_line))
        if not matches:
            continue
        lon_idx, lon_line = min(matches, key=lambda item: math.hypot(
            (item[1]["rect"].x0 + item[1]["rect"].x1) / 2.0 - lx,
            (item[1]["rect"].y0 + item[1]["rect"].y1) / 2.0 - ly))
        box = pymupdf.Rect(lat_line["rect"]) | pymupdf.Rect(lon_line["rect"])
        members = [(lat_idx, lat_line), (lon_idx, lon_line)]
        add_box_vote(box, members, f"{lat_line['text']} | {lon_line['text']}",
                     "AIRPORT_INFO", "approach_airport_info", 8.0)

    # Terrain/obstacle heights are supporting evidence only; route altitudes are
    # deliberately excluded unless a nearby terrain/obstacle cue is present.
    for idx, line in enumerate(lines):
        if idx in used_lines:
            continue
        text = line["text"].strip()
        if not (ALTITUDE_FT_RE.match(text) or ALTITUDE_VALUE_RE.match(text)):
            continue
        rect = line["rect"]
        cx, cy = (rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0
        if any(_pt_seg_dist(cx, cy, *seg) < 24.0 for seg in route_like):
            continue
        nearby = [
            other for other in lines if other is not line
            and math.hypot(
                (other["rect"].x0 + other["rect"].x1) / 2.0 - cx,
                (other["rect"].y0 + other["rect"].y1) / 2.0 - cy,
            ) <= 24.0
            and re.search(r"\b(?:MT|MOUNTAIN|PEAK|OBST|ELEV|ALT)\b", other["text"], re.I)
        ]
        if nearby:
            rot = _line_rotation(line)
            if rot is not None:
                votes["TERRAIN"].append(rot)
            refs.append(ReferenceBox(rect.x0, rect.y0, rect.x1, rect.y1, 0, 3.0,
                                     "approach_terrain", text))

    active = {name: values for name, values in votes.items() if values}
    if not active:
        return OrientationResult(0, "none", refs, var_check="APPROACH_REFERENCES_NOT_FOUND",
                                 basis="emg+caution+holding+iaf+airport_info+terrain",
                                 text_lines=len(lines), apply_rotation=False)

    # One vote per independent feature family prevents a busy box from
    # overpowering a contradictory family. Incomplete evidence remains REVIEW.
    family_rotations = {name: max(set(values), key=values.count) for name, values in active.items()}
    rotations = set(family_rotations.values())
    detail = ",".join(f"{name}={rot}" for name, rot in family_rotations.items())
    if len(rotations) != 1:
        return OrientationResult(0, "approach_references", refs,
                                 var_check=f"APPROACH_REFERENCE_CONFLICT({detail})",
                                 basis="+".join(active), text_lines=len(lines),
                                 apply_rotation=False)
    rotation = rotations.pop()
    state = "APPROACH_REFERENCES_AGREE" if len(active) > 1 else "APPROACH_SINGLE_REFERENCE"
    return OrientationResult(rotation, "approach_references", refs,
                             var_check=f"{state}({detail})",
                             basis="+".join(active), text_lines=len(lines),
                             apply_rotation=False)


def _find_rma_latitude_vote(
    page: pymupdf.Page,
    lines: list[dict],
    roi_rect: pymupdf.Rect,
) -> Optional[dict]:
    """Use only edge ticks that satisfy the RMA grid geometry and label rules.

    A tick must lie on an ROI edge. Multiple ticks along one axis must form
    opposite-edge pairs at the same pixel coordinate. A single unpaired tick
    is permitted only when it has a coordinate label. Each pair has exactly
    one label, and a series of three or more coordinates must be evenly spaced.
    """
    width, height = roi_rect.width, roi_rect.height
    if width <= 0 or height <= 0:
        return None

    segments, _ = _collect_segments(page, roi_rect)
    rm = page.rotation_matrix
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if item[0] != "re":
                continue
            rect = pymupdf.Rect(item[1]) * rm
            rect.normalize()
            if not roi_rect.intersects(rect):
                continue
            if 1.5 <= rect.width <= 16.0 and rect.height <= 2.5:
                segments.append((rect.x0, (rect.y0 + rect.y1) / 2.0,
                                 rect.x1, (rect.y0 + rect.y1) / 2.0))
            elif 1.5 <= rect.height <= 16.0 and rect.width <= 2.5:
                segments.append(((rect.x0 + rect.x1) / 2.0, rect.y0,
                                 (rect.x0 + rect.x1) / 2.0, rect.y1))

    edge_tol = max(3.0, min(6.0, min(width, height) * 0.006))
    pair_tol = 4.5
    raw_ticks: list[dict] = []
    seen: set[tuple[str, int]] = set()

    # Normalize every short segment into one of four ROI-edge tick families.
    for seg in segments:
        x0, y0, x1, y1 = map(float, seg)
        length = math.hypot(x1 - x0, y1 - y0)
        if not 1.5 <= length <= 16.0:
            continue
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        horizontal = abs(x1 - x0) >= abs(y1 - y0) * 1.5
        vertical = abs(y1 - y0) >= abs(x1 - x0) * 1.5
        options = []
        if horizontal:
            options.extend((("left", abs(mx - roi_rect.x0), my, "y"),
                            ("right", abs(mx - roi_rect.x1), my, "y")))
        if vertical:
            options.extend((("top", abs(my - roi_rect.y0), mx, "x"),
                            ("bottom", abs(my - roi_rect.y1), mx, "x")))
        if not options:
            continue
        edge, gap, coordinate, axis = min(options, key=lambda item: item[1])
        edge_coord = {"left": roi_rect.x0, "right": roi_rect.x1,
                      "top": roi_rect.y0, "bottom": roi_rect.y1}[edge]
        edge_span = (min(x0, x1) - edge_coord <= edge_tol
                     and max(x0, x1) + edge_tol >= edge_coord) if axis == "y" else (
                     min(y0, y1) - edge_coord <= edge_tol
                     and max(y0, y1) + edge_tol >= edge_coord)
        if gap > edge_tol or not edge_span:
            continue
        dedupe_key = (edge, round(coordinate / 2.0))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        raw_ticks.append({"edge": edge, "axis": axis, "coordinate": coordinate,
                          "gap": gap, "seg": (x0, y0, x1, y1)})

    if not raw_ticks:
        return None

    opposite = {"left": "right", "right": "left",
                "top": "bottom", "bottom": "top"}
    labels = []
    for line in lines:
        if not RMA_LAT_RE.match(line["text"].strip()):
            continue
        rect = line["rect"]
        cx = (rect.x0 + rect.x1) / 2.0
        cy = (rect.y0 + rect.y1) / 2.0
        rot, residual = _snap_rotation(_angle_cw_from_up(*line["up"]))
        if abs(residual) > 8.0:
            continue
        edge_distances = {
            "left": abs(cx - roi_rect.x0), "right": abs(roi_rect.x1 - cx),
            "top": abs(cy - roi_rect.y0), "bottom": abs(roi_rect.y1 - cy),
        }
        edge = min(edge_distances, key=edge_distances.get)
        if edge_distances[edge] > max(24.0, min(width, height) * 0.045):
            continue
        axis = "y" if edge in ("left", "right") else "x"
        coordinate = cy if axis == "y" else cx
        nearby = [tick for tick in raw_ticks
                  if tick["edge"] == edge
                  and abs(tick["coordinate"] - coordinate) <= max(
                      6.0, rect.height * 0.75 if axis == "y" else rect.width * 0.75)]
        if not nearby:
            continue
        tick = min(nearby, key=lambda item: item["gap"] +
                   abs(item["coordinate"] - coordinate))
        labels.append({"rot": rot, "rect": rect, "tick": tick})

    # A physical tick may be associated with exactly one coordinate label.
    by_tick: dict[tuple[str, int], list[dict]] = {}
    for label in labels:
        tick = label["tick"]
        key = (tick["edge"], round(tick["coordinate"] / 2.0))
        by_tick.setdefault(key, []).append(label)
    labels = [items[0] for items in by_tick.values() if len(items) == 1]
    # A left/right (or top/bottom) pair may have one label total, never two.
    unique_pair_labels = []
    for label in labels:
        tick = label["tick"]
        same_pair = [other for other in labels
                     if other["tick"]["axis"] == tick["axis"]
                     and abs(other["tick"]["coordinate"] - tick["coordinate"]) <= pair_tol]
        if len(same_pair) == 1:
            unique_pair_labels.append(label)
    labels = unique_pair_labels
    if not labels:
        # Preserve rejected tick marks in diagnostics; they cannot vote.
        rejected_boxes = []
        for tick in raw_ticks:
            x0, y0, x1, y1 = tick["seg"]
            rejected_boxes.append(ReferenceBox(min(x0, x1), min(y0, y1),
                max(x0, x1), max(y0, y1), 0, 1.0, "lat_tick_rejected", "REJECTED TICK"))
        return {"rot": None, "n": 0, "boxes": rejected_boxes, "ignored": True}

    # Build one candidate per labelled coordinate. If there are several
    # coordinates on an axis, each must have its opposite-edge counterpart.
    candidates = []
    for label in labels:
        tick = label["tick"]
        matching = [other for other in raw_ticks
                    if other["edge"] == opposite[tick["edge"]]
                    and other["axis"] == tick["axis"]
                    and abs(other["coordinate"] - tick["coordinate"]) <= pair_tol]
        axis_ticks = [other for other in raw_ticks if other["axis"] == tick["axis"]]
        if matching:
            opposite_tick = min(matching, key=lambda item: abs(item["coordinate"] - tick["coordinate"]))
        elif len(axis_ticks) == 1:
            # A lone tick is accepted only because this candidate has a label.
            opposite_tick = None
        else:
            continue
        candidates.append({"rot": label["rot"], "rect": label["rect"],
                           "tick": tick["seg"],
                           "opposite_tick": opposite_tick["seg"] if opposite_tick else None,
                           "axis": tick["axis"], "coordinate": tick["coordinate"]})

    if not candidates:
        return None

    # Validate spacing against the complete detected pair series, not only
    # the subset that happens to carry labels.
    invalid_axes = set()
    for axis in ("x", "y"):
        paired_coordinates = []
        for tick in raw_ticks:
            if tick["axis"] != axis:
                continue
            if any(other["edge"] == opposite[tick["edge"]]
                   and other["axis"] == axis
                   and abs(other["coordinate"] - tick["coordinate"]) <= pair_tol
                   for other in raw_ticks):
                if not any(abs(tick["coordinate"] - old) <= pair_tol for old in paired_coordinates):
                    paired_coordinates.append(tick["coordinate"])
        paired_coordinates.sort()
        if len(paired_coordinates) >= 3:
            intervals = [paired_coordinates[i + 1] - paired_coordinates[i]
                         for i in range(len(paired_coordinates) - 1)]
            mean_interval = sum(intervals) / len(intervals)
            tolerance = max(2.5, mean_interval * 0.10)
            if mean_interval <= 0 or any(abs(value - mean_interval) > tolerance
                                         for value in intervals):
                invalid_axes.add(axis)

    valid_candidates = [item for item in candidates if item["axis"] not in invalid_axes]
    rotations = {item["rot"] for item in valid_candidates}
    boxes = []
    for item in valid_candidates:
        r = item["rect"]
        boxes.append(ReferenceBox(r.x0, r.y0, r.x1, r.y1, item["rot"], 5.0, "lat_label", "RMA LAT"))
        for kind, seg in (("lat_tick", item["tick"]),
                          ("lat_opposite_tick", item["opposite_tick"])):
            if seg is not None:
                x0, y0, x1, y1 = seg
                boxes.append(ReferenceBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
                                          item["rot"], 3.0, kind, "LAT TICK"))
    accepted_segments = set()
    for item in valid_candidates:
        accepted_segments.add(tuple(item["tick"]))
        if item["opposite_tick"] is not None:
            accepted_segments.add(tuple(item["opposite_tick"]))
    for tick in raw_ticks:
        if tuple(tick["seg"]) in accepted_segments:
            continue
        x0, y0, x1, y1 = tick["seg"]
        boxes.append(ReferenceBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
                                  0, 1.0, "lat_tick_rejected", "REJECTED TICK"))
    if not valid_candidates or len(rotations) != 1:
        return {"rot": None, "n": len(valid_candidates), "boxes": boxes, "ignored": True}
    return {"rot": rotations.pop(), "n": len(valid_candidates), "boxes": boxes, "ignored": False}

def _detect_rma_orientation(page: pymupdf.Page, roi_rect: pymupdf.Rect) -> OrientationResult:
    lines = _collect_text_lines(page, roi_rect)
    vote = _find_rma_latitude_vote(page, lines, roi_rect)
    if vote is None:
        return OrientationResult(0, "none", [], var_check="RMA_LAT_NOT_FOUND", text_lines=len(lines))
    if vote["ignored"]:
        return OrientationResult(0, "none", vote["boxes"], var_check="RMA_LAT_CONFLICT", text_lines=len(lines))
    return OrientationResult(
        vote["rot"], "lat_labels", vote["boxes"],
        basis=f"rma_lat_ticks(n={vote['n']})",
        var_check=f"RMA_LAT_OK(n={vote['n']})", text_lines=len(lines),
    )


def _detect_from_indicator(
    page: pymupdf.Page, roi_rect: pymupdf.Rect, allow_unlabeled_pair: bool = True
) -> OrientationResult:
    lines = _collect_text_lines(page, roi_rect)
    ind = _find_indicator_text(lines)
    n_letters = [l for l in lines if l["text"].upper() == "N"]
    if ind is not None:
        c = ind["rect"]
        region = pymupdf.Rect(c.x0 - 90, c.y0 - 90, c.x1 + 90, c.y1 + 90) & roi_rect
        n_near = [n for n in n_letters if _overlap(n["rect"], region)]
    else:
        region, n_near = roi_rect, n_letters
    pair = _find_arrow_pair(page, region, n_near, wide=ind is None) if (ind is not None or allow_unlabeled_pair) else None
    lat = _lat_label_vote(lines)
    lat_boxes = [ReferenceBox(r.x0, r.y0, r.x1, r.y1, 0, 5.0, "lat_label", "LAT") for r in (lat["boxes"] if lat else [])]

    # --- VAR gostergesi ---
    ind_rot = None
    refs: list[ReferenceBox] = []
    theta = resid = None
    basis = ""
    text = year = ""
    if pair is not None or ind is not None:
        if pair is not None:
            nvec, basis = pair["vec"], pair["basis"]
            for a in pair["pair"]:
                sg = a["seg"]
                refs.append(ReferenceBox(min(sg[0], sg[2]), min(sg[1], sg[3]), max(sg[0], sg[2]), max(sg[1], sg[3]),
                                         0, 1.0, "arrow_axis", "ARROW", (*a["tail"], *a["head"])))
        else:
            nvec, basis = ind["anchor"]["up"], "indicator_text"
        theta = _angle_cw_from_up(*nvec)
        ind_rot, resid = _snap_rotation(theta)
        if ind is not None:
            r = ind["rect"]
            label = f"VAR {ind['value']:+.1f}" if ind["value"] is not None else "VAR ?"
            if ind["year"]:
                label += f" {ind['year']}"
            text, year = ind["text"], ind["year"]
        else:
            pts = [p for a in pair["pair"] for p in (a["tail"], a["head"])]
            r = pymupdf.Rect(min(p[0] for p in pts) - 6, min(p[1] for p in pts) - 6,
                             max(p[0] for p in pts) + 6, max(p[1] for p in pts) + 6)
            label = "ARROWS (VAR text unreadable)"
        refs.insert(0, ReferenceBox(r.x0, r.y0, r.x1, r.y1, ind_rot, 60.0, "north_arrow", label))
    strong = ind is not None  # VAR + deger/yil metni okundu

    lat_rot = lat["rot"] if lat and not lat["ignored"] else None
    n_lines = len(lines)

    if ind_rot is None:
        if lat_rot is None:
            chk = "LAT_IGNORED(conflict)" if lat and lat["ignored"] else "n/a"
            return OrientationResult(0, "none", lat_boxes if lat else [], var_check=chk, text_lines=n_lines)
        return OrientationResult(lat_rot, "lat_labels", lat_boxes, basis=f"lat_labels(n={lat['n']})",
                                 var_check="n/a", text_lines=n_lines)

    final, method, check = ind_rot, "variation_indicator", "n/a"
    if lat is not None and lat["ignored"]:
        check = "LAT_IGNORED(conflict)"
    elif lat_rot is not None:
        if lat_rot == ind_rot:
            check = f"LAT_AGREE(n={lat['n']})"
        elif strong:
            check = f"LAT_CONFLICT(lat={lat_rot}) -> VAR wins"
        else:  # gosterge zayif (VAR metni okunamadi) ve LAT tutarli -> LAT'a guven
            final, method, basis, check = lat_rot, "lat_labels", f"lat_labels(n={lat['n']})", f"IND_WEAK(ind={ind_rot}) -> LAT wins"
            refs = []
    return OrientationResult(final, method, refs + lat_boxes, var_text=text, year=year, north_angle_deg=theta,
                             residual_deg=resid, var_check=check, basis=basis, text_lines=n_lines)


def detect_page_orientation(page: pymupdf.Page, roi, category: str = "", procedure: str = "") -> OrientationResult:
    """Use VAR-North for all supported charts; RMA additionally uses edge LAT ticks."""
    roi_rect = _resolve_roi_rect(page, roi)
    is_arrival = (
        str(category or "").strip().upper() == "ARRIVAL"
        or "ARRIVAL" in (procedure or "").upper()
        or str(category or "").strip().upper() == "STAR"
    )
    is_departure = (
        str(category or "").strip().upper() in ("DEPARTURE", "SID")
        or "DEPARTURE" in (procedure or "").upper()
        or "SID" in (procedure or "").upper()
    )
    is_approach = (
        str(category or "").strip().upper() in ("APPROACH", "IAP")
        or "APPROACH" in (procedure or "").upper()
        or "INSTRUMENT APPROACH" in (procedure or "").upper()
    )
    if is_approach:
        return _approach_orientation(page, roi_rect)
    result = _detect_from_indicator(
        page, roi_rect, allow_unlabeled_pair=not (is_arrival or is_departure)
    )
    is_rma = (
        "RADAR MINIMUM ALTITUDE" in (procedure or "").upper()
        or "RADAR MINIMUM ALTITUDE" in (category or "").upper()
    )
    if is_arrival:
        arrival = _detect_arrival_orientation(page, roi_rect)
        # Keep VAR-North as a first-class reusable reference and the strongest
        # authority when present. Arrival's MSA/scale detectors remain independent.
        if result.method == "variation_indicator":
            refs = result.references + arrival.references
            check = result.var_check
            if arrival.method == "arrival_references":
                if arrival.rotation_cw == result.rotation_cw:
                    check += f"; {arrival.var_check}"
                else:
                    check += f"; {arrival.var_check}; VAR_WINS"
            return OrientationResult(
                result.rotation_cw, result.method, refs, var_text=result.var_text, year=result.year,
                north_angle_deg=result.north_angle_deg, residual_deg=result.residual_deg,
                var_check=check, basis=result.basis, text_lines=max(result.text_lines, arrival.text_lines),
                apply_rotation=False,
            )
        return arrival
    if is_rma:
        lines = _collect_text_lines(page, roi_rect)
        vote = _find_rma_latitude_vote(page, lines, roi_rect)
        scale = _detect_color_altitude_scale(page, lines, roi_rect)
        scale_refs = _reference_boxes_for_scale(scale) if scale else []
        scale_rot = scale["rot"] if scale else None
        if result.method == "none":
            primary = vote if vote and not vote["ignored"] else None
            if primary is None and scale_rot is None:
                refs = vote["boxes"] if vote else []
                return OrientationResult(0, "none", refs + scale_refs,
                    var_check="RMA_LAT_AND_SCALE_NOT_FOUND" if vote is None else "RMA_REFERENCES_CONFLICT_OR_SCALE_NOT_FOUND",
                    basis="rma_lat+color_altitude", text_lines=len(lines))
            if primary is not None and scale_rot is not None and primary["rot"] != scale_rot:
                return OrientationResult(0, "none", primary["boxes"] + scale_refs,
                    var_check=f"RMA_REFERENCE_CONFLICT(LAT={primary['rot']},SCALE={scale_rot})",
                    basis="rma_lat+color_altitude", text_lines=len(lines))
            chosen = primary["rot"] if primary is not None else scale_rot
            refs = (primary["boxes"] if primary is not None else (vote["boxes"] if vote else [])) + scale_refs
            basis = "rma_lat+color_altitude" if primary is not None and scale_rot is not None else ("rma_lat_ticks" if primary is not None else "color_altitude")
            return OrientationResult(chosen, "lat_labels" if primary is not None else "rma_references",
                refs, basis=basis, var_check="RMA_REFERENCES_AGREE" if primary is not None and scale_rot is not None else "RMA_SINGLE_REFERENCE",
                text_lines=len(lines))
        refs = list(result.references) + (vote["boxes"] if vote else []) + scale_refs
        check = result.var_check
        if vote is not None:
            if vote["ignored"]:
                check += "; RMA_LAT_CONFLICT"
            elif vote["rot"] == result.rotation_cw:
                check += f"; RMA_LAT_AGREE(n={vote['n']})"
            else:
                check += f"; RMA_LAT_CONFLICT(lat={vote['rot']}) -> VAR wins"
        if scale_rot is not None:
            if scale_rot == result.rotation_cw:
                check += "; RMA_SCALE_AGREE"
            else:
                check += f"; RMA_SCALE_CONFLICT(scale={scale_rot}) -> VAR wins"
        result = OrientationResult(
            result.rotation_cw, result.method, refs,
            var_text=result.var_text, year=result.year,
            north_angle_deg=result.north_angle_deg, residual_deg=result.residual_deg,
            var_check=check, basis=result.basis, text_lines=max(result.text_lines, len(lines)),
        )
    if is_departure:
        return _departure_orientation(page, roi_rect)
    return result


# ------------------------------------------------------------------ donusum
def _transform_point_cw(x, y, page_w, page_h, rot_cw):
    rot_cw %= 360
    if rot_cw == 90:
        return page_h - y, x
    if rot_cw == 180:
        return page_w - x, page_h - y
    if rot_cw == 270:
        return y, page_w - x
    return x, y


def _transform_rect_cw(x0, y0, x1, y1, page_w, page_h, rot_cw):
    pts = [_transform_point_cw(x, y, page_w, page_h, rot_cw) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _transform_roi_after_cw_rotation(roi, page_w, page_h, rot_cw):
    rot_cw %= 360
    if rot_cw == 0:
        return dict(roi), page_w, page_h
    nx0, ny0, nx1, ny1 = _transform_rect_cw(roi["x0"], roi["y0"], roi["x1"], roi["y1"], page_w, page_h, rot_cw)
    new_w, new_h = (page_h, page_w) if rot_cw in (90, 270) else (page_w, page_h)
    return {"x0": nx0, "y0": ny0, "x1": nx1, "y1": ny1}, new_w, new_h


# ------------------------------------------------------------------ diagnostik
def _save_turn_diagnostic(target_pdf, orig_roi, orig_page_w, orig_page_h, result, out_jpg, dpi=180) -> None:
    """Sonuc PDF'i render eder; ROI (mavi), gosterge (kirmizi) ve ok eksenleri (macenta) cizer."""
    scale = dpi / 72.0
    rot_cw = result.rotation_cw % 360 if result.apply_rotation else 0
    doc = None
    if isinstance(target_pdf, pymupdf.Page):
        page = target_pdf
    else:
        doc = pymupdf.open(target_pdf)
        page = doc[0]
    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]

        def T(x, y):
            tx, ty = _transform_point_cw(x, y, orig_page_w, orig_page_h, rot_cw)
            return int(tx * scale), int(ty * scale)

        def TR(x0, y0, x1, y1):
            a = _transform_rect_cw(x0, y0, x1, y1, orig_page_w, orig_page_h, rot_cw)
            return int(a[0] * scale), int(a[1] * scale), int(a[2] * scale), int(a[3] * scale)

        x0, y0, x1, y1 = TR(orig_roi.x0, orig_roi.y0, orig_roi.x1, orig_roi.y1)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (255, 110, 0), 3)

        found = result.method in ("variation_indicator", "lat_labels", "arrival_references", "rma_references", "departure_references", "approach_references")
        for ref in result.references:
            if ref.kind == "lat_label":
                lx0, ly0, lx1, ly1 = TR(ref.x0, ref.y0, ref.x1, ref.y1)
                cv2.rectangle(bgr, (lx0 - 4, ly0 - 4), (lx1 + 4, ly1 + 4), (0, 170, 0), 2)
                continue
            if ref.kind in ("lat_tick", "lat_opposite_tick", "lat_tick_rejected"):
                tx0, ty0, tx1, ty1 = TR(ref.x0, ref.y0, ref.x1, ref.y1)
                color = ((0, 0, 255) if ref.kind == "lat_tick_rejected"
                         else (0, 165, 255) if ref.kind == "lat_tick" else (255, 0, 255))
                cv2.line(bgr, (tx0, ty0), (tx1, ty1), color, 4, cv2.LINE_AA)
                continue
            if ref.kind == "arrow_axis" and ref.line:
                p, q = T(ref.line[0], ref.line[1]), T(ref.line[2], ref.line[3])
                cv2.arrowedLine(bgr, p, q, (200, 0, 200), 3, cv2.LINE_AA, tipLength=0.3)
                continue
            if ref.kind == "msa_circle":
                bx0, by0, bx1, by1 = TR(ref.x0, ref.y0, ref.x1, ref.y1)
                cv2.rectangle(bgr, (bx0, by0), (bx1, by1), (0, 210, 255), 3)
                cv2.putText(bgr, "MSA REFERENCE", (bx0, max(18, by0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2, cv2.LINE_AA)
                continue
            if ref.kind == "msa_label":
                bx0, by0, bx1, by1 = TR(ref.x0, ref.y0, ref.x1, ref.y1)
                cv2.rectangle(bgr, (bx0 - 3, by0 - 3), (bx1 + 3, by1 + 3), (0, 210, 255), 2)
                continue
            if ref.kind == "altitude_scale":
                bx0, by0, bx1, by1 = TR(ref.x0, ref.y0, ref.x1, ref.y1)
                cv2.rectangle(bgr, (bx0, by0), (bx1, by1), (255, 180, 0), 3)
                cv2.putText(bgr, "COLOR / ALTITUDE SCALE", (bx0, max(18, by0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA)
                continue
            if ref.kind == "altitude_scale_label":
                bx0, by0, bx1, by1 = TR(ref.x0, ref.y0, ref.x1, ref.y1)
                cv2.rectangle(bgr, (bx0 - 3, by0 - 3), (bx1 + 3, by1 + 3), (255, 180, 0), 2)
                continue
            if ref.kind == "altitude_swatch":
                bx0, by0, bx1, by1 = TR(ref.x0, ref.y0, ref.x1, ref.y1)
                cv2.rectangle(bgr, (bx0, by0), (bx1, by1), (255, 0, 180), 2)
                continue
            if ref.kind == "edge_tick" and ref.line:
                p, q = T(ref.line[0], ref.line[1]), T(ref.line[2], ref.line[3])
                cv2.line(bgr, p, q, (0, 165, 255), 4, cv2.LINE_AA)
                continue
            if ref.kind in ("approach_emg", "approach_box_text", "approach_iaf", "approach_terrain",
                            "approach_airport_info", "approach_holding"):
                bx0, by0, bx1, by1 = TR(ref.x0, ref.y0, ref.x1, ref.y1)
                color = {
                    "approach_emg": (0, 0, 255),
                    "approach_box_text": (255, 0, 220),
                    "approach_iaf": (0, 210, 0),
                    "approach_terrain": (0, 165, 255),
                    "approach_airport_info": (255, 180, 0),
                    "approach_holding": (180, 0, 255),
                }[ref.kind]
                cv2.rectangle(bgr, (bx0 - 3, by0 - 3), (bx1 + 3, by1 + 3), color, 2)
                continue
            if ref.kind == "departure_text":
                bx0, by0, bx1, by1 = TR(ref.x0, ref.y0, ref.x1, ref.y1)
                cv2.rectangle(bgr, (bx0 - 3, by0 - 3), (bx1 + 3, by1 + 3), (0, 220, 0), 2)
                continue
            bx0, by0, bx1, by1 = TR(ref.x0, ref.y0, ref.x1, ref.y1)
            bx0, by0 = max(0, bx0 - 6), max(0, by0 - 6)
            bx1, by1 = min(W - 1, bx1 + 6), min(H - 1, by1 + 6)
            cv2.rectangle(bgr, (bx0, by0), (bx1, by1), (0, 0, 235), 3)
            cv2.putText(bgr, f"NORTH INDICATOR  {ref.label}", (bx0, max(18, by0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 235), 2, cv2.LINE_AA)

        if found:
            ang = "" if result.north_angle_deg is None else f" | NORTH={result.north_angle_deg:.0f}deg resid={result.residual_deg:+.0f}"
            ang += f" | TXT={result.text_lines}"
            if result.method == "variation_indicator":
                ind = "VAR_IND=FOUND"
            elif result.method == "arrival_references":
                ind = "ARRIVAL_REFERENCES"
            elif result.method == "departure_references":
                ind = "DEPARTURE_REFERENCES"
            elif result.method == "approach_references":
                ind = "APPROACH_REFERENCES"
            elif result.method == "rma_references":
                ind = "RMA_REFERENCES"
            else:
                ind = "VAR_IND=NOT USED, LAT"
            if result.method in ("arrival_references", "departure_references", "approach_references") or not result.apply_rotation:
                stage_name = ("DEPARTURE" if result.method == "departure_references" else
                              "APPROACH" if result.method == "approach_references" else "ARRIVAL")
                status = f"{stage_name} PREVIEW ONLY | PROPOSED_CW={result.rotation_cw} | {ind}({result.basis}) | CHECK={result.var_check}{ang}"
            else:
                status = f"ROT_CW={rot_cw} | {ind}({result.basis}) | CHECK={result.var_check}{ang}"
            color = (0, 0, 220) if rot_cw else (0, 140, 0)
        else:
            if result.var_check.startswith("ARRIVAL_REFERENCE_CONFLICT"):
                status = f"ROT_CW=0 | ARRIVAL REFERENCE CONFLICT -> REVIEW | {result.var_check}"
                color = (0, 80, 255)
            elif result.var_check == "ARRIVAL_REFERENCES_NOT_FOUND":
                status = f"ROT_CW=0 | ARRIVAL REFERENCES NOT FOUND | TXT={result.text_lines}"
                color = (0, 120, 255)
            elif result.var_check.startswith("DEPARTURE_REFERENCE_CONFLICT"):
                status = f"ROT_CW=0 | DEPARTURE REFERENCE CONFLICT -> REVIEW | {result.var_check}"
                color = (0, 80, 255)
            elif result.var_check == "DEPARTURE_REFERENCES_NOT_FOUND":
                status = f"ROT_CW=0 | DEPARTURE REFERENCES NOT FOUND | TXT={result.text_lines}"
                color = (0, 120, 255)
            elif result.var_check == "DEPARTURE_TEXT_CONFLICT":
                status = f"ROT_CW=0 | DEPARTURE TEXT VOTE CONFLICT -> REVIEW"
                color = (0, 80, 255)
            elif result.var_check.startswith("APPROACH_REFERENCE_CONFLICT"):
                status = f"ROT_CW=0 | APPROACH REFERENCE CONFLICT -> REVIEW | {result.var_check}"
                color = (0, 80, 255)
            elif result.var_check == "APPROACH_REFERENCES_NOT_FOUND":
                status = f"ROT_CW=0 | APPROACH REFERENCES NOT FOUND | TXT={result.text_lines}"
                color = (0, 120, 255)
            else:
                status = f"ROT_CW=0 | VAR_IND=NOT FOUND -> assumed NORTH=0 | CHECK={result.var_check} | TXT={result.text_lines}"
                color = (0, 120, 255)
        cv2.putText(bgr, status, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_jpg), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    finally:
        if doc is not None:
            doc.close()


def _is_aerodrome_diagram_item(filename: str, meta: dict) -> bool:
    category = str(meta.get("category", "")).strip().lower()
    procedure = str(meta.get("procedure", "")).strip().upper()
    return (
        procedure == "AERODROME DIAGRAM"
        or "AERODROME DIAGRAM" in filename.upper()
        or (category == "diagram" and "RADAR MINIMUM ALTITUDES" not in procedure)
    )


def _is_radar_minimum_altitudes_item(filename: str, meta: dict) -> bool:
    procedure = str(meta.get("procedure", "")).strip().upper()
    return "RADAR MINIMUM ALTITUDE" in procedure or "RADAR MINIMUM ALTITUDE" in filename.upper()



def _is_arrival_item(filename: str, meta: dict) -> bool:
    category = str(meta.get("category", "")).strip().upper()
    procedure = str(meta.get("procedure", "")).strip().upper()
    return category in ("ARRIVAL", "STAR") or "ARRIVAL" in procedure or "STAR" in procedure


def _is_approach_item(filename: str, meta: dict) -> bool:
    category = str(meta.get("category", "")).strip().upper()
    procedure = str(meta.get("procedure", "")).strip().upper()
    name = filename.upper()
    return (
        category in ("APPROACH", "IAP")
        or "APPROACH" in procedure
        or "INSTRUMENT APPROACH" in procedure
        or "APPROACH" in name
    )


def _is_departure_item(filename: str, meta: dict) -> bool:
    category = str(meta.get("category", "")).strip().upper()
    procedure = str(meta.get("procedure", "")).strip().upper()
    return category in ("DEPARTURE", "SID") or "DEPARTURE" in procedure or "SID" in procedure


def run_second_stage_orientation_turn(
    first_dir: Path,
    second_dir: Path,
    diagnostic: bool = False,
    diagnostic_dir: Path | None = None,
    only_aerodrome_diagrams: bool = True,
):
    manifest_path = first_dir / "roi_manifest.json"
    if not manifest_path.is_file():
        return [], [], []

    records: dict[str, dict] = json.loads(manifest_path.read_text(encoding="utf-8"))
    second_dir.mkdir(parents=True, exist_ok=True)
    if diagnostic and diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    moved: list[Path] = []
    diagnostics: list[Path] = []
    rotated_items: list[tuple[str, int, str]] = []
    updated_records: dict[str, dict] = {}
    remaining: dict[str, dict] = dict(records)

    for filename, meta in records.items():
        if only_aerodrome_diagrams and not (
            _is_aerodrome_diagram_item(filename, meta)
            or _is_radar_minimum_altitudes_item(filename, meta)
            or _is_arrival_item(filename, meta)
            or _is_departure_item(filename, meta)
            or _is_approach_item(filename, meta)
        ):
            continue
        src_pdf = first_dir / filename
        if not src_pdf.is_file():
            continue

        rd = meta.get("roi") or {}
        roi = PlanviewRoi(x0=float(rd.get("x0", 0.0)), y0=float(rd.get("y0", 0.0)),
                          x1=float(rd.get("x1", 0.0)), y1=float(rd.get("y1", 0.0)))
        dst_pdf = second_dir / filename

        doc = pymupdf.open(src_pdf)
        try:
            page = doc[0]
            page_w, page_h = float(page.rect.width), float(page.rect.height)
            result = detect_page_orientation(page, roi, str(meta.get("category", "")), str(meta.get("procedure", "")))
            effective_rotation = result.rotation_cw if result.apply_rotation else 0
            if effective_rotation != 0:
                page.set_rotation((int(page.rotation) + effective_rotation) % 360)
                doc.save(str(dst_pdf))
                rotated = True
            else:
                rotated = False
        finally:
            doc.close()

        if rotated:
            src_pdf.unlink()
            rotated_items.append((filename, effective_rotation, f"{result.method}/{result.basis}/{result.var_check}"))
        else:
            if dst_pdf.exists():
                dst_pdf.unlink()
            shutil.move(str(src_pdf), str(dst_pdf))

        if diagnostic and diagnostic_dir is not None:
            diag_path = diagnostic_dir / f"{Path(filename).stem}.jpg"
            _save_turn_diagnostic(dst_pdf, roi, page_w, page_h, result, diag_path)
            if diag_path.is_file():
                diagnostics.append(diag_path)

        moved.append(dst_pdf)
        remaining.pop(filename, None)

        new_roi, new_w, new_h = _transform_roi_after_cw_rotation(
            {"x0": roi.x0, "y0": roi.y0, "x1": roi.x1, "y1": roi.y1}, page_w, page_h, effective_rotation)
        nm = dict(meta)
        nm.update(page_width=new_w, page_height=new_h, roi=new_roi)
        nm["orientation"] = {
            "rotation_cw": result.rotation_cw if result.apply_rotation else 0,
            "proposed_rotation_cw": result.rotation_cw,
            "applied": result.apply_rotation,
            "method": result.method, "basis": result.basis,
            "var_text": result.var_text, "year": result.year, "var_check": result.var_check,
            "north_angle_deg": result.north_angle_deg, "residual_deg": result.residual_deg, "text_lines_in_roi": result.text_lines,
        }
        updated_records[filename] = nm

    if updated_records:
        (second_dir / "roi_manifest.json").write_text(
            json.dumps(updated_records, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps(remaining, indent=2, ensure_ascii=False), encoding="utf-8")
    return moved, diagnostics, rotated_items