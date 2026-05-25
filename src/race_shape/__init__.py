"""Shadow-only Race Shape intelligence sidecar.

This package is intentionally isolated from live VÉLØ scoring. It parses and
aggregates archive data so race-execution signals can be validated before any
promotion into production prediction paths.
"""

from .form_history_parser import RpFormHistoryParser, parse_rp_profile_html
from .models import JockeyRideProfile, RaceShapeRun, RideAnomaly
from .ride_profile import JockeyRideProfileBuilder

__all__ = [
    "JockeyRideProfile",
    "JockeyRideProfileBuilder",
    "RaceShapeRun",
    "RideAnomaly",
    "RpFormHistoryParser",
    "parse_rp_profile_html",
]
