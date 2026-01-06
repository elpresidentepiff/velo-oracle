"""Liquid Loop Adapter core cell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class LiquidCell:
    """Applies bounded exponential moving average updates to weights."""

    learning_rate: float = 0.001
    max_adjustment: float = 0.1

    def apply(self, weights: np.ndarray, gradients: Sequence[float]) -> np.ndarray:
        """Return updated weights constrained by ``max_adjustment``."""

        gradients_array = np.asarray(list(gradients), dtype=np.float32)
        if gradients_array.shape != weights.shape:
            raise ValueError("gradients must match weights shape")

        delta = self.learning_rate * gradients_array
        delta = np.clip(delta, -self.max_adjustment, self.max_adjustment)
        return weights + delta
