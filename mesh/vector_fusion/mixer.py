"""Mixer module for the Vector Fusion Core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class StateMixer:
    """Deterministic placeholder for an attention-free mixer.

    The mixer simply averages the provided state vectors and applies a
    lightweight gated residual update. Later phases can replace this with a
    Hyena/Mamba style block without altering callers.
    """

    depth: int = 4
    gate: float = 0.5

    def mix(self, states: Sequence[np.ndarray]) -> np.ndarray:
        """Combine a sequence of state vectors into a single representation."""

        if not states:
            raise ValueError("at least one state vector is required")

        base = np.mean(np.stack(states, axis=0), axis=0)
        residual = states[-1]
        return (1.0 - self.gate) * base + self.gate * residual
