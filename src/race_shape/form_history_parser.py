"""Parser for archived Racing Post-style horse profile form tables.

The parser is intentionally dependency-light. It extracts table rows from saved
HTML and converts likely form rows into RaceShapeRun objects. The output is for
shadow validation and archive analysis only.
"""

from __future__ import annotations

import html
import re
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional

from .models import RaceShapeRun

_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_RESULT_RE = re.compile(r"\b(?P<pos>\d{1,2})\s*/\s*(?P<field>\d{1,2})\b")
_FRACTIONAL_ODDS_RE = re.compile(r"\b(?P<num>\d+)\s*/\s*(?P<den>\d+)\b")
_DECIMAL_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

_MARGIN_MAP = {
    "nk": 0.25,
    "nse": 0.05,
    "nose": 0.05,
    "sh": 0.1,
    "short head": 0.1,
    "hd": 0.2,
    "head": 0.2,
}

_UNICODE_FRACTIONS = {
    "¼": 0.25,
    "½": 0.5,
    "¾": 0.75,
    "⅓": 1 / 3,
    "⅔": 2 / 3,
}


class _TableTextParser(HTMLParser):
    """Small HTML table extractor tuned for archived profile pages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_row = False
        self._in_cell = False
        self._cell_parts: List[str] = []
        self._row: List[str] = []
        self.rows: List[List[str]] = []

    def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
        if tag == "tr":
            self._in_row = True
            self._row = []
        elif self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"td", "th"} and self._in_cell:
            text = " ".join("".join(self._cell_parts).split())
            if text:
                self._row.append(html.unescape(text))
            self._in_cell = False
            self._cell_parts = []
        elif tag == "tr" and self._in_row:
            if self._row:
                self.rows.append(self._row)
            self._in_row = False
            self._row = []

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._in_cell:
            self._cell_parts.append(data)


def _parse_date(text: str) -> Optional[date]:
    match = _DATE_RE.search(text)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_result(text: str) -> tuple[Optional[int], Optional[int]]:
    match = _RESULT_RE.search(text)
    if not match:
        return None, None
    return int(match.group("pos")), int(match.group("field"))


def _parse_fractional_odds(text: str) -> Optional[float]:
    normalized = text.replace("EVS", "1/1").replace("Evs", "1/1")
    match = _FRACTIONAL_ODDS_RE.search(normalized)
    if not match:
        return None
    numerator = int(match.group("num"))
    denominator = int(match.group("den"))
    if denominator == 0:
        return None
    return round(1 + numerator / denominator, 4)


def _parse_margin(text: str) -> Optional[float]:
    lowered = text.lower()
    for key, value in _MARGIN_MAP.items():
        if key in lowered:
            return value

    total = 0.0
    for glyph, value in _UNICODE_FRACTIONS.items():
        if glyph in text:
            total += value
            text = text.replace(glyph, "")

    numbers = [float(match.group(0)) for match in _DECIMAL_RE.finditer(text)]
    if numbers:
        total += numbers[0]

    return round(total, 4) if total else None


def _first_non_empty(values: Iterable[Optional[str]]) -> Optional[str]:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


class RpFormHistoryParser:
    """Convert archived horse profile HTML into RaceShapeRun records."""

    def parse_html(self, html_text: str, *, horse: str, source_path: Optional[str] = None) -> List[RaceShapeRun]:
        table_parser = _TableTextParser()
        table_parser.feed(html_text)
        runs: List[RaceShapeRun] = []

        for cells in table_parser.rows:
            joined = " | ".join(cells)
            race_date = _parse_date(joined)
            finishing_position, field_size = _parse_result(joined)
            if race_date is None or finishing_position is None or field_size is None:
                continue

            runs.append(
                RaceShapeRun(
                    horse=horse,
                    race_date=race_date,
                    course=self._guess_course(cells),
                    race_type=self._guess_race_type(cells),
                    distance=self._guess_distance(cells),
                    going=self._guess_going(cells),
                    finishing_position=finishing_position,
                    field_size=field_size,
                    beaten_margin_lengths=_parse_margin(joined),
                    sp_decimal=self._guess_sp(cells),
                    jockey=self._guess_person(cells, keywords=("jockey",)),
                    trainer=self._guess_person(cells, keywords=("trainer",)),
                    raw_cells=cells,
                    source_path=source_path,
                )
            )

        return runs

    def parse_file(self, path: str | Path, *, horse: Optional[str] = None) -> List[RaceShapeRun]:
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        inferred_horse = horse or self._infer_horse_from_filename(file_path)
        return self.parse_html(text, horse=inferred_horse, source_path=str(file_path))

    @staticmethod
    def _infer_horse_from_filename(path: Path) -> str:
        stem = path.stem
        parts = stem.split("_horse_")
        if len(parts) == 2:
            slug = parts[1].split("_form_")[0]
            slug_parts = slug.split("_")[1:] or slug.split("_")
            return " ".join(piece.capitalize() for piece in slug_parts if piece)
        return stem.replace("_", " ").title()

    @staticmethod
    def _guess_course(cells: List[str]) -> Optional[str]:
        for cell in cells:
            if _DATE_RE.search(cell):
                continue
            tokens = cell.split()
            if 1 <= len(tokens) <= 4 and cell.isupper():
                return cell.title()
        return None

    @staticmethod
    def _guess_race_type(cells: List[str]) -> Optional[str]:
        for cell in cells:
            lowered = cell.lower()
            if any(token in lowered for token in ("handicap", "stakes", "maiden", "novice", "classified")):
                return cell
        return None

    @staticmethod
    def _guess_distance(cells: List[str]) -> Optional[str]:
        for cell in cells:
            if re.search(r"\b\d{1,2}f\b|\b\d{1,2}m\b", cell.lower()):
                return cell
        return None

    @staticmethod
    def _guess_going(cells: List[str]) -> Optional[str]:
        going_terms = ("good", "soft", "heavy", "firm", "standard", "yielding")
        for cell in cells:
            lowered = cell.lower()
            if any(term in lowered for term in going_terms) and len(cell) <= 40:
                return cell
        return None

    @staticmethod
    def _guess_sp(cells: List[str]) -> Optional[float]:
        candidates = [_parse_fractional_odds(cell) for cell in cells]
        numeric = [value for value in candidates if value is not None]
        return numeric[-1] if numeric else None

    @staticmethod
    def _guess_person(cells: List[str], *, keywords: tuple[str, ...]) -> Optional[str]:
        # Archived rows often carry compact cells without labels; keep this
        # conservative rather than inventing identities from ambiguous cells.
        labelled = []
        for cell in cells:
            lowered = cell.lower()
            if any(keyword in lowered for keyword in keywords):
                labelled.append(re.sub(r"(?i)\b(jockey|trainer)\b", "", cell).strip(" :-"))
        return _first_non_empty(labelled)


def parse_rp_profile_html(html_text: str, *, horse: str, source_path: Optional[str] = None) -> List[RaceShapeRun]:
    return RpFormHistoryParser().parse_html(html_text, horse=horse, source_path=source_path)
