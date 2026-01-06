"""Race shape simulator placeholder."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass(slots=True)
class RaceShapeSimulator:
    """Stochastic simulator producing hypothetical finishing orders."""

    seed: int = 42
    fatigue_decay: float = 0.02
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def simulate(
        self,
        runners: Sequence[str],
        favourite: Optional[str],
        pace_shift: float,
        trials: int = 100,
        seed: Optional[int] = None,
        timeout_ms: int = 200,
    ) -> List[List[str]]:
        """Run Monte Carlo simulations of race outcomes."""

        rng_seed = self.seed if seed is None else seed + int(pace_shift * 1000)
        rng = random.Random(rng_seed)
        outcomes: List[List[str]] = []
        start_ns = time.perf_counter_ns()
        budget_ns = timeout_ms * 1_000_000

        favourite_bias = 0.0
        if favourite and favourite in runners:
            favourite_bias = max(-0.25, min(0.25, pace_shift * 0.8))

        for _ in range(trials):
            ordering = list(runners)
            rng.shuffle(ordering)
            if favourite and favourite in ordering:
                base_prob = 1.0 / max(1, len(runners))
                strike = rng.random()
                win_threshold = min(0.9, base_prob + favourite_bias + 0.1)
                fade_threshold = max(0.05, base_prob - favourite_bias)
                if strike <= win_threshold:
                    ordering.remove(favourite)
                    ordering.insert(0, favourite)
                elif strike >= 1.0 - fade_threshold:
                    ordering.remove(favourite)
                    ordering.append(favourite)
            outcomes.append(ordering)
            if time.perf_counter_ns() - start_ns > budget_ns:
                raise TimeoutError("race shape simulation exceeded budget")
        return outcomes
