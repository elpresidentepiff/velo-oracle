"""Idea state transition helpers for the Vector Fusion Core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .mixer import StateMixer


@dataclass(slots=True)
class IdeaStepper:
    """Applies a deterministic update rule over the latent state vector."""

    mixer: StateMixer
    decay: float = 0.8

    def step(self, previous: Optional[np.ndarray], observation: np.ndarray) -> np.ndarray:
        """Blend the previous idea state with the current observation."""

        if previous is None:
            return observation
        mixed = self.mixer.mix((previous, observation))
        return self.decay * previous + (1.0 - self.decay) * mixed
