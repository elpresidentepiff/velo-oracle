"""Finite state machine definitions for the Cognitive Mesh orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConvictionState(str, Enum):
    """State designations for conviction progression."""

    SCAN = "S0"
    LATENT = "S1"
    EMERGING = "S2"
    STRIKE = "S3"
    OFF_RISK = "S-1"


@dataclass(slots=True)
class OrchestratorConfig:
    """Configuration governing state transition thresholds."""

    s0_to_s1: float = 0.75
    s1_to_s2: float = 0.85
    s2_to_s3: float = 0.9
    strike_hysteresis: float = 0.7
    strike_sustain_ticks: int = 2
