"""Jockey ride profile aggregation for the Race Shape sidecar."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .models import JockeyRideProfile, RaceShapeRun


class JockeyRideProfileBuilder:
    """Build archive-only execution variance profiles by jockey."""

    def build(self, runs: Iterable[RaceShapeRun], *, min_rides: int = 3) -> List[JockeyRideProfile]:
        grouped: Dict[str, List[RaceShapeRun]] = defaultdict(list)
        for run in runs:
            if run.jockey:
                grouped[run.jockey].append(run)

        profiles = [
            JockeyRideProfile.from_runs(jockey, jockey_runs)
            for jockey, jockey_runs in grouped.items()
            if len(jockey_runs) >= min_rides
        ]
        return sorted(
            profiles,
            key=lambda profile: (profile.anomaly_rate, profile.well_fancied_failures, profile.rides),
            reverse=True,
        )

    def write_json(self, profiles: Sequence[JockeyRideProfile], path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [profile.to_dict() for profile in profiles]
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
