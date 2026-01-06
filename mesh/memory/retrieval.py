"""Memory Vault retrieval utilities."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(slots=True)
class MemoryEvent:
    """Represents an episodic decision event suitable for audits."""

    trace_id: str
    race_id: str
    runner_id: str
    timestamp: str
    state: str
    previous_state: str
    conviction: float
    model_prob: float
    doctrine_scores: Dict[str, float]
    doctrine_veto: bool
    world_robustness: Optional[float]
    risk_throttle: str
    market_snapshot: Mapping[str, float]
    explain: str
    explain_code: str
    state_notes: Sequence[str]
    outcome: Optional[float] = None
    finish_position: Optional[int] = None
    payout: Optional[float] = None


class EpisodicMemory:
    """Volatile in-memory store used for Phase 1 scaffolding."""

    def __init__(self) -> None:
        self._events: List[MemoryEvent] = []

    def record(self, event: MemoryEvent) -> None:
        self._events.append(event)

    def latest(self, race_id: str, runner_id: str) -> Optional[MemoryEvent]:
        for event in reversed(self._events):
            if event.race_id == race_id and event.runner_id == runner_id:
                return event
        return None

    def update_outcome(
        self,
        race_id: str,
        runner_id: str,
        *,
        outcome: float,
        finish_position: Optional[int],
        payout: Optional[float],
    ) -> MemoryEvent:
        event = self.latest(race_id, runner_id)
        if event is None:
            raise KeyError(f"No memory event for race '{race_id}' runner '{runner_id}'")
        event.outcome = outcome
        event.finish_position = finish_position
        event.payout = payout
        return event

    def events_for_race(self, race_id: str) -> Iterable[MemoryEvent]:
        return (event for event in self._events if event.race_id == race_id)

    def query(self, state: str) -> Iterable[MemoryEvent]:
        return (event for event in self._events if event.state == state)

    def clear(self) -> None:
        self._events.clear()

    def entropy(
        self,
        *,
        field: str = "explain_code",
        window: int = 50,
        min_events: int = 10,
    ) -> float:
        """Return normalised Shannon entropy for recent events."""

        if not self._events:
            return 1.0

        window = max(1, window)
        values = [
            getattr(event, field, None)
            for event in self._events[-window:]
            if getattr(event, field, None) is not None
        ]

        if len(values) < max(2, min_events):
            return 1.0

        counts = Counter(values)
        total = sum(counts.values())
        if total <= 1:
            return 1.0

        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log(probability, 2)

        max_entropy = math.log(len(counts), 2) if len(counts) > 1 else 0.0
        if max_entropy == 0.0:
            return 0.0
        return float(min(1.0, entropy / max_entropy))
