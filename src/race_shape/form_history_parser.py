"""Parser for archived Racing Post horse profile form tables.

Targets the hp-formTable structure rendered by Racing Post's profile pages.
Uses BeautifulSoup for robust HTML traversal; falls back to a lightweight
stdlib parser when bs4 is unavailable.

Shadow/archive-only. Never imported by live scoring code.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import List, Optional

from .models import RaceShapeRun

# ── Optional BeautifulSoup import ────────────────────────────────────────────
try:
    from bs4 import BeautifulSoup as _BS4
    _BS4_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BS4_AVAILABLE = False

# ── Regex helpers ─────────────────────────────────────────────────────────────
_DATE_ABBR_RE = re.compile(r"\b(\d{1,2})([A-Za-z]{3})(\d{2})\b")   # 23Apr26
_DATE_ISO_RE  = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")        # 2026-04-23
_RESULT_RE    = re.compile(r"(\d{1,2})\s*/\s*(\d{1,2})")
_FRAC_ODDS_RE = re.compile(r"\b(\d{1,3})\s*/\s*(\d{1,3})\b")
_JOCKEY_UID_RE = re.compile(r"/profile/jockey/(\d+)/")

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_UNICODE_FRACS = {"¼": 0.25, "½": 0.5, "¾": 0.75}
_MARGIN_WORDS  = {"nk": 0.25, "nse": 0.05, "nose": 0.05, "hd": 0.2,
                  "head": 0.2, "sh": 0.1, "short head": 0.1}


def _parse_date(text: str) -> Optional[date]:
    m = _DATE_ABBR_RE.search(text)
    if m:
        day, mon, yr = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        month = _MONTH_MAP.get(mon)
        if month:
            try:
                return date(2000 + yr, month, day)
            except ValueError:
                pass
    m = _DATE_ISO_RE.search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _parse_result(text: str):
    matches = list(_RESULT_RE.finditer(text))
    if not matches:
        return None, None
    # First match is the position/field pair; winner SP uses a dash not slash
    m = matches[0]
    return int(m.group(1)), int(m.group(2))


def _parse_sp(text: str) -> Optional[float]:
    text = text.strip().replace("EVS", "1/1").replace("Evs", "1/1")
    m = _FRAC_ODDS_RE.search(text)
    if m:
        n, d = int(m.group(1)), int(m.group(2))
        if d and 1 <= n <= 200 and 1 <= d <= 20:
            return round(1 + n / d, 4)
    return None


def _parse_margin(text: str) -> Optional[float]:
    low = text.lower()
    for word, val in _MARGIN_WORDS.items():
        if word in low:
            return val
    total = 0.0
    for g, v in _UNICODE_FRACS.items():
        if g in text:
            total += v
            text = text.replace(g, "")
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if nums:
        total += float(nums[0])
    return round(total, 4) if total else None


def _cell_text(cell) -> str:
    if _BS4_AVAILABLE:
        return cell.get_text(" ", strip=True)
    return re.sub(r"<[^>]+>", " ", str(cell)).strip()


class RpFormHistoryParser:
    """Convert archived Racing Post profile HTML into RaceShapeRun records.

    Targets the hp-formTable rows identified by .hp-formTable__dateWrapper
    when BeautifulSoup is available; falls back to regex-based row detection.
    """

    def parse_html(
        self, html_text: str, *, horse: str, source_path: Optional[str] = None
    ) -> List[RaceShapeRun]:
        if _BS4_AVAILABLE:
            return self._parse_with_bs4(html_text, horse=horse, source_path=source_path)
        return self._parse_fallback(html_text, horse=horse, source_path=source_path)

    def parse_file(self, path, *, horse: Optional[str] = None) -> List[RaceShapeRun]:
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        inferred = horse or self._horse_from_path(file_path)
        return self.parse_html(text, horse=inferred, source_path=str(file_path))

    # ── BeautifulSoup path ────────────────────────────────────────────────────

    def _parse_with_bs4(self, html_text: str, *, horse: str, source_path) -> List[RaceShapeRun]:
        soup = _BS4(html_text, "lxml")
        date_wrappers = soup.select(".hp-formTable__dateWrapper")

        # Fallback: if CSS selector finds nothing (different page version),
        # use all table rows that contain a date.
        if not date_wrappers:
            return self._parse_fallback(html_text, horse=horse, source_path=source_path)

        runs: List[RaceShapeRun] = []
        for dw in date_wrappers:
            row = dw.find_parent("tr")
            if not row:
                continue
            cells = row.find_all("td")
            if len(cells) < 8:
                continue
            runs.append(self._row_to_run(cells, horse, source_path))
        return runs

    def _row_to_run(self, cells, horse: str, source_path) -> RaceShapeRun:
        # Cell layout (RP profile form tab):
        # [0] date + video/result links
        # [1] course name + surface + class info
        # [2] distance
        # [3] going abbreviation
        # [4] weight (+ headgear)
        # [5] position / field + margin + winner
        # [6] horse's own SP
        # [7] jockey (link)
        # [8] TS  [9] RPR  [10] OR  [11] OR+

        def txt(i):
            return _cell_text(cells[i]) if i < len(cells) else ""

        # Date
        race_date = _parse_date(txt(0))

        # Course — clean the messy cell[1] text
        course = self._extract_course(cells[1])

        # Distance
        dist_raw = txt(2)

        # Going
        going_raw = txt(3)

        # Position / field
        pos, field = _parse_result(txt(5))

        # Horse's own SP is cell[6] — a clean fractional string
        sp = _parse_sp(txt(6))

        # Margin from cell[5]
        margin = _parse_margin(txt(5))

        # Jockey — extract from link href for uid, from link text for name
        jockey_name, jockey_uid = self._extract_jockey(cells[7])

        # Result type
        if pos == 1:
            result_type = "WIN"
        elif pos is not None and pos <= 3:
            result_type = "PLACED"
        elif pos is not None:
            result_type = "LOSS"
        else:
            result_type = None

        return RaceShapeRun(
            horse=horse,
            race_date=race_date,
            course=course,
            race_type=None,
            distance=dist_raw or None,
            going=going_raw or None,
            finishing_position=pos,
            field_size=field,
            beaten_margin_lengths=margin,
            sp_decimal=sp,
            jockey=jockey_name,
            trainer=None,
            raw_cells=[_cell_text(c) for c in cells],
            source_path=source_path,
        )

    @staticmethod
    def _extract_course(cell) -> Optional[str]:
        """Extract course name from the messy second cell."""
        link = cell.find("a") if _BS4_AVAILABLE else None
        if link:
            href = link.get("href", "")
            # /profile/course/394/southwell-aw → "southwell-aw" → "Southwell Aw"
            m = re.search(r"/profile/course/\d+/([^/\s]+)", href)
            if m:
                slug = m.group(1).replace("-", " ")
                # Drop "(aw)" / "(aw)" redundancy already in name
                return slug.title()
        text = _cell_text(cell)
        # Take first 1-3 capitalized words before first digit or "right"
        m = re.match(r"^([A-Za-z\s\(\)]+?)(?:\s+(?:\d|\bSth\b|\bRhs\b|\bright\b))", text)
        return m.group(1).strip().title() if m else None

    @staticmethod
    def _extract_jockey(cell) -> tuple[Optional[str], Optional[int]]:
        if not _BS4_AVAILABLE:
            return None, None
        link = cell.find("a")
        if link:
            href = link.get("href", "")
            uid_m = _JOCKEY_UID_RE.search(href)
            uid = int(uid_m.group(1)) if uid_m else None
            name = link.get_text(" ", strip=True)
            # strip trailing icon text ("right" from SVG titles)
            name = re.sub(r"\s*(right|left)\s*$", "", name, flags=re.I).strip()
            return name or None, uid
        return None, None

    # ── Fallback path (no bs4, or non-RP HTML like tests) ─────────────────────

    def _parse_fallback(self, html_text: str, *, horse: str, source_path) -> List[RaceShapeRun]:
        """Generic row scanner used in tests with synthetic HTML and as bs4 fallback."""
        import html as _html
        from html.parser import HTMLParser

        class _TP(HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True)
                self._in_row = self._in_cell = False
                self._parts: List[str] = []
                self._row: List[str] = []
                self.rows: List[List[str]] = []
            def handle_starttag(self, tag, attrs):
                if tag == "tr": self._in_row, self._row = True, []
                elif self._in_row and tag in ("td", "th"): self._in_cell, self._parts = True, []
            def handle_endtag(self, tag):
                if tag in ("td", "th") and self._in_cell:
                    t = " ".join("".join(self._parts).split())
                    if t: self._row.append(_html.unescape(t))
                    self._in_cell = False
                elif tag == "tr" and self._in_row:
                    if self._row: self.rows.append(self._row)
                    self._in_row = False
            def handle_data(self, data):
                if self._in_cell: self._parts.append(data)

        tp = _TP()
        tp.feed(html_text)
        runs = []
        for cells in tp.rows:
            joined = " | ".join(cells)
            race_date = _parse_date(joined)
            pos, field = _parse_result(joined)
            if race_date is None or pos is None or field is None:
                continue
            # Jockey: look for "Jockey: Name" pattern in test HTML
            jockey = None
            for c in cells:
                m = re.search(r"Jockey:\s*(.+)", c, re.I)
                if m:
                    jockey = m.group(1).strip()
                    break
            # SP: last unambiguous fractional (cell with only N/M)
            sp = None
            for c in reversed(cells):
                s = _parse_sp(c.strip())
                if s and 1.01 <= s <= 100.0:
                    sp = s
                    break
            runs.append(RaceShapeRun(
                horse=horse, race_date=race_date,
                course=None, race_type=None, distance=None, going=None,
                finishing_position=pos, field_size=field,
                beaten_margin_lengths=_parse_margin(joined),
                sp_decimal=sp, jockey=jockey, trainer=None,
                raw_cells=cells, source_path=source_path,
            ))
        return runs

    @staticmethod
    def _horse_from_path(path: Path) -> str:
        stem = path.stem
        parts = stem.split("_horse_")
        if len(parts) == 2:
            slug = parts[1].split("_form_")[0]
            slug_parts = slug.split("_")[1:] or slug.split("_")
            return " ".join(p.capitalize() for p in slug_parts if p)
        return stem.replace("_", " ").title()


def parse_rp_profile_html(
    html_text: str, *, horse: str, source_path: Optional[str] = None
) -> List[RaceShapeRun]:
    return RpFormHistoryParser().parse_html(html_text, horse=horse, source_path=source_path)
