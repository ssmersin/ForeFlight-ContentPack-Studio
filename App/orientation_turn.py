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


def _detect_from_indicator(page: pymupdf.Page, roi_rect: pymupdf.Rect) -> OrientationResult:
    lines = _collect_text_lines(page, roi_rect)
    ind = _find_indicator_text(lines)
    n_letters = [l for l in lines if l["text"].upper() == "N"]
    if ind is not None:
        c = ind["rect"]
        region = pymupdf.Rect(c.x0 - 90, c.y0 - 90, c.x1 + 90, c.y1 + 90) & roi_rect
        n_near = [n for n in n_letters if _overlap(n["rect"], region)]
    else:
        region, n_near = roi_rect, n_letters
    pair = _find_arrow_pair(page, region, n_near, wide=ind is None)
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
    """Yalnizca ROI icindeki variation-north gostergesine bakar; yoksa North=0."""
    del category, procedure
    return _detect_from_indicator(page, _resolve_roi_rect(page, roi))


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
    rot_cw = result.rotation_cw % 360
    doc = pymupdf.open(target_pdf)
    try:
        page = doc[0]
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

        found = result.method in ("variation_indicator", "lat_labels")
        for ref in result.references:
            if ref.kind == "lat_label":
                lx0, ly0, lx1, ly1 = TR(ref.x0, ref.y0, ref.x1, ref.y1)
                cv2.rectangle(bgr, (lx0 - 4, ly0 - 4), (lx1 + 4, ly1 + 4), (0, 170, 0), 2)
                continue
            if ref.kind == "arrow_axis" and ref.line:
                p, q = T(ref.line[0], ref.line[1]), T(ref.line[2], ref.line[3])
                cv2.arrowedLine(bgr, p, q, (200, 0, 200), 3, cv2.LINE_AA, tipLength=0.3)
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
            ind = "VAR_IND=FOUND" if result.method == "variation_indicator" else "VAR_IND=NOT USED, LAT"
            status = f"ROT_CW={rot_cw} | {ind}({result.basis}) | CHECK={result.var_check}{ang}"
            color = (0, 0, 220) if rot_cw else (0, 140, 0)
        else:
            status = f"ROT_CW=0 | VAR_IND=NOT FOUND -> assumed NORTH=0 | TXT_LINES_IN_ROI={result.text_lines}"
            color = (0, 120, 255)
        cv2.putText(bgr, status, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_jpg), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    finally:
        doc.close()


def _is_aerodrome_diagram_item(filename: str, meta: dict) -> bool:
    category = str(meta.get("category", "")).strip().lower()
    procedure = str(meta.get("procedure", "")).strip().upper()
    return (
        procedure == "AERODROME DIAGRAM"
        or "AERODROME DIAGRAM" in filename.upper()
        or (category == "diagram" and "RADAR MINIMUM ALTITUDES" not in procedure)
    )


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
        if only_aerodrome_diagrams and not _is_aerodrome_diagram_item(filename, meta):
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
            if result.rotation_cw != 0:
                page.set_rotation((int(page.rotation) + result.rotation_cw) % 360)
                doc.save(str(dst_pdf))
                rotated = True
            else:
                rotated = False
        finally:
            doc.close()

        if rotated:
            src_pdf.unlink()
            rotated_items.append((filename, result.rotation_cw, f"{result.method}/{result.basis}/{result.var_check}"))
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
            {"x0": roi.x0, "y0": roi.y0, "x1": roi.x1, "y1": roi.y1}, page_w, page_h, result.rotation_cw)
        nm = dict(meta)
        nm.update(page_width=new_w, page_height=new_h, roi=new_roi)
        nm["orientation"] = {
            "rotation_cw": result.rotation_cw, "method": result.method, "basis": result.basis,
            "var_text": result.var_text, "year": result.year, "var_check": result.var_check,
            "north_angle_deg": result.north_angle_deg, "residual_deg": result.residual_deg, "text_lines_in_roi": result.text_lines,
        }
        updated_records[filename] = nm

    if updated_records:
        (second_dir / "roi_manifest.json").write_text(
            json.dumps(updated_records, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps(remaining, indent=2, ensure_ascii=False), encoding="utf-8")
    return moved, diagnostics, rotated_items