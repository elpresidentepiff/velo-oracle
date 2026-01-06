"""Vector Fusion Core encoders.

This module defines light-weight encoder primitives used to transform
hand-crafted racing features into fixed-width continuous representations.
Phase 1 intentionally keeps the logic deterministic and inexpensive; later
phases can swap the encoder implementations without changing the public
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np


class FeatureEncoder(Protocol):
    """Protocol implemented by all feature encoders.

    Encoders convert a slice of the raw feature dictionary into a
    fixed-width numpy vector. Implementations are free to use any encoding
    strategy (e.g. normalization, learned embeddings) as long as they honour
    the interface.
    """

    name: str
    dimension: int

    def encode(self, features: Mapping[str, float]) -> np.ndarray:
        """Return an encoded representation for the given feature mapping."""


@dataclass(slots=True)
class ScalarEncoder:
    """Simple baseline encoder that maps one scalar feature into ℝ^d.

    The encoder writes the raw feature value to the first coordinate and
    leaves the remaining dimensions at zero. Downstream modules can later
    replace this behaviour with learned projections without changing the
    surrounding plumbing.
    """

    name: str
    feature_key: str
    dimension: int = 64
    default_value: float = 0.0

    def encode(self, features: Mapping[str, float]) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[0] = float(features.get(self.feature_key, self.default_value))
        return vector


@dataclass(slots=True)
class MomentumTrendEncoder:
    """Encoder that captures short-term momentum and recency features."""

    name: str = "momentum"
    keys: Sequence[str] = ("form_momentum", "form_ewma", "form_recency")
    dimension: int = 64

    def encode(self, features: Mapping[str, float]) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for index, key in enumerate(self.keys):
            if index >= self.dimension:
                break
            vector[index] = float(features.get(key, 0.0))
        return vector


@dataclass(slots=True)
class VectorFusionCore:
    """Aggregates multiple encoders into a single fused representation."""

    encoders: Sequence[FeatureEncoder]

    def encode(self, features: Mapping[str, float]) -> np.ndarray:
        """Fuse encoder outputs into a single state vector.

        Args:
            features: Raw feature mapping for a runner.

        Returns:
            A continuous representation obtained by concatenating all encoder
            outputs along the feature axis.
        """

        vectors: Iterable[np.ndarray] = (encoder.encode(features) for encoder in self.encoders)
        return np.concatenate(tuple(vectors), axis=0)


DEFAULT_ENCODERS: Sequence[FeatureEncoder] = (
    MomentumTrendEncoder(),
    ScalarEncoder(name="rpr_latest", feature_key="rpr_latest"),
    ScalarEncoder(name="rpr_peak", feature_key="rpr_peak"),
    ScalarEncoder(name="ts_latest", feature_key="ts_latest"),
)


def build_default_vector_fusion_core() -> VectorFusionCore:
    """Factory that instantiates the default encoder stack."""

    return VectorFusionCore(encoders=DEFAULT_ENCODERS)
