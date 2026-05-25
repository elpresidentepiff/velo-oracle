"""Race Shape sidecar domain models.

These dataclasses keep execution-variance analytics portable and isolated from
production scoring. They are intended for archive validation only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RaceShapeRun:
    """One historical run parsed from an archived horse profile form table."""

    horse: str
    race_date: Optional[date]
    course: Optional[str]
    race_type: Optional[str]
    distance: Optional[str]
    going: Optional[str]
    finishing_position: Optional[int]
    field_size: Optional[int]
    beaten_margin_lengths: Optional[float]
    sp_decimal: Optional[float]
    jockey: Optional[str]
    trainer: Optional[str]
    raw_cells: List[str] = field(default_factory=list)
    source_path: Optional[str] = None

    @property
    def finished_in_frame(self) -> bool:
        return self.finishing_position is not None and self.finishing_position <= 3

    @property
    def was_well_fancied(self) -> bool:
        return self.sp_decimal is not None and self.sp_decimal <= 4.0

    @property
    def normalized_finish(self) -> Optional[float]:
        if not self.finishing_position or not self.field_size or self.field_size <= 1:
            return None
        return (self.finishing_position - 1) / (self.field_size - 1)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["race_date"] = self.race_date.isoformat() if self.race_date else None
        payload["finished_in_frame"] = self.finished_in_frame
        payload["was_well_fancied"] = self.was_well_fancied
        payload["normalized_finish"] = self.normalized_finish
        return payload


@dataclass(frozen=True)
class RideAnomaly:
    """A run where outcome diverged sharply from market expectation."""

    jockey: str
    horse: str
    race_date: Optional[date]
    course: Optional[str]
    sp_decimal: float
    finishing_position: int
    field_size: int
    severity: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["race_date"] = self.race_date.isoformat() if self.race_date else None
        return payload


@dataclass(frozen=True)
class JockeyRideProfile:
    """Aggregate shadow profile for a jockey's archived rides."""

    jockey: str
    rides: int
    wins: int
    places: int
    well_fancied_rides: int
    well_fancied_failures: int
    average_normalized_finish: Optional[float]
    normalized_finish_stdev: Optional[float]
    anomaly_rate: float
    anomalies: List[RideAnomaly] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.rides if self.rides else 0.0

    @property
    def place_rate(self) -> float:
        return self.places / self.rides if self.rides else 0.0

    @classmethod
    def from_runs(cls, jockey: str, runs: List[RaceShapeRun]) -> "JockeyRideProfile":
        valid_finishes = [r for r in runs if r.finishing_position and r.field_size]
        normalized = [r.normalized_finish for r in valid_finishes if r.normalized_finish is not None]
        anomalies: List[RideAnomaly] = []

        for run in valid_finishes:
            if not run.was_well_fancied:
                continue
            bad_finish = run.finishing_position > max(3, int(run.field_size * 0.5))
            if bad_finish and run.sp_decimal is not None:
                severity = "HIGH" if run.finishing_position > int(run.field_size * 0.75) else "MEDIUM"
                anomalies.append(
                    RideAnomaly(
                        jockey=jockey,
                        horse=run.horse,
                        race_date=run.race_date,
                        course=run.course,
                        sp_decimal=run.sp_decimal,
                        finishing_position=run.finishing_position,
                        field_size=run.field_size,
                        severity=severity,
                        reason="Well-fancied runner finished outside expected contention band",
                    )
                )

        well_fancied = [r for r in valid_finishes if r.was_well_fancied]
        wins = sum(1 for r in valid_finishes if r.finishing_position == 1)
        places = sum(1 for r in valid_finishes if r.finished_in_frame)

        return cls(
            jockey=jockey,
            rides=len(valid_finishes),
            wins=wins,
            places=places,
            well_fancied_rides=len(well_fancied),
            well_fancied_failures=len(anomalies),
            average_normalized_finish=mean(normalized) if normalized else None,
            normalized_finish_stdev=pstdev(normalized) if len(normalized) > 1 else None,
            anomaly_rate=(len(anomalies) / len(well_fancied)) if well_fancied else 0.0,
            anomalies=anomalies,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["win_rate"] = self.win_rate
        payload["place_rate"] = self.place_rate
        payload["anomalies"] = [a.to_dict() for a in self.anomalies]
        return payload
