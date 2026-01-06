"""Market microstructure simulator placeholder."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(slots=True)
class MarketFlowSimulator:
    """Simulate price and volume drift for a runner."""

    seed: int = 7
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def simulate(
        self,
        start_price: float,
        steps: int = 12,
        seed: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Generate a synthetic market path."""

        rng = self._rng if seed is None else random.Random(seed)
        price = start_price
        prices: List[float] = [price]
        volumes: List[float] = [1.0]
        for _ in range(steps):
            change = rng.uniform(-0.1, 0.1)
            price = max(0.01, price * (1 + change))
            prices.append(round(price, 4))
            volumes.append(round(max(0.1, volumes[-1] + rng.uniform(-0.2, 0.2)), 4))
        return {"prices": prices, "volumes": volumes}

    @staticmethod
    def detect_manipulation(
        prices: Sequence[float],
        volumes: Sequence[float],
    ) -> Tuple[float, bool]:
        """Return a spoofing score derived from price reversals."""

        if len(prices) < 3:
            return 0.0, False

        max_score = 0.0
        for idx in range(1, len(prices) - 1):
            previous = prices[idx - 1]
            current = prices[idx]
            if previous <= 0:
                continue
            drop = (previous - current) / previous
            if drop < 0.06:
                continue
            lookahead = min(len(prices), idx + 4)
            for jdx in range(idx + 1, lookahead):
                rebound_base = prices[jdx]
                rebound = (rebound_base - current) / max(current, 1e-6)
                if rebound < 0.05:
                    continue
                vol_prev = volumes[idx - 1] if idx - 1 < len(volumes) else 1.0
                vol_future = volumes[jdx] if jdx < len(volumes) else vol_prev
                volume_ratio = max(0.0, vol_prev - vol_future)
                score = min(1.0, drop + rebound + 0.1 * volume_ratio)
                max_score = max(max_score, score)
        return max_score, max_score >= 0.35
