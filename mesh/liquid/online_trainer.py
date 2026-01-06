"""Online trainer orchestrating the Liquid Loop Adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .liquid_cell import LiquidCell


@dataclass(slots=True)
class OnlineTrainer:
    """Maintains a bounded adaptive head for online updates."""

    cell: LiquidCell
    calibration_tolerance: float = 0.02
    reinforcement_rate: float = 0.1
    _weights: np.ndarray = field(init=False, repr=False)
    _history: List[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._weights = np.zeros(4, dtype=np.float32)
        self._history: List[float] = []

    @property
    def weights(self) -> np.ndarray:
        """Current meta-weights."""

        return self._weights

    def update(self, gradient: np.ndarray, calibration_delta: float) -> np.ndarray:
        """Update weights if calibration remains within tolerance."""

        if abs(calibration_delta) > self.calibration_tolerance:
            return self._weights
        self._weights = self.cell.apply(self._weights, gradient)
        self._history.append(float(calibration_delta))
        return self._weights

    def reinforce(self, conviction: float, reward: float) -> np.ndarray:
        """Nudge weights based on realised reward."""

        delta = reward - conviction
        adjustment = np.full_like(self._weights, self.reinforcement_rate * delta)
        self._weights = self.cell.apply(self._weights, adjustment)
        self._history.append(float(delta))
        return self._weights

    def reset(self) -> None:
        """Reset weights to a neutral state."""

        self._weights[:] = 0.0
        self._history.clear()
