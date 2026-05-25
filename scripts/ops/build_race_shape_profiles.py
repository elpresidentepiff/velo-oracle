#!/usr/bin/env python3
"""Build Race Shape sidecar artifacts from archived horse profile HTML.

This command is intentionally shadow-only. It reads saved profile HTML, writes
JSON artifacts, and never imports live scoring code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.race_shape.form_history_parser import RpFormHistoryParser
from src.race_shape.ride_profile import JockeyRideProfileBuilder


def _iter_html_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.rglob("*.html"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Race Shape shadow artifacts")
    parser.add_argument("--input-dir", required=True, help="Directory containing archived RP profile HTML files")
    parser.add_argument("--output-dir", default="data/race_shape", help="Directory for generated JSON artifacts")
    parser.add_argument("--min-jockey-rides", type=int, default=3, help="Minimum parsed rides before writing a jockey profile")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    form_parser = RpFormHistoryParser()
    runs = []
    failed_files = []

    for html_file in _iter_html_files(input_dir):
        try:
            runs.extend(form_parser.parse_file(html_file))
        except Exception as exc:  # keep batch jobs resilient; report failures in metadata
            failed_files.append({"path": str(html_file), "error": str(exc)})

    runs_payload = [run.to_dict() for run in runs]
    (output_dir / "race_shape_runs.json").write_text(
        json.dumps(runs_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    profiles = JockeyRideProfileBuilder().build(runs, min_rides=args.min_jockey_rides)
    JockeyRideProfileBuilder().write_json(profiles, output_dir / "jockey_ride_profiles.json")

    metadata = {
        "input_dir": str(input_dir),
        "html_files_seen": len(_iter_html_files(input_dir)),
        "runs_parsed": len(runs),
        "jockey_profiles": len(profiles),
        "failed_files": failed_files,
        "shadow_only": True,
        "live_scoring_touched": False,
    }
    (output_dir / "build_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
