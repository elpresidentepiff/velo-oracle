"""Post-mortem analysis for Cognitive Mesh decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from mesh.memory import MemoryEvent


@dataclass(slots=True)
class PostMortemInsight:
    """Structured insight for a single runner outcome."""

    runner_id: str
    conviction: float
    model_prob: float
    outcome: float
    finish_position: int | None
    doctrine_alignment: Dict[str, float]
    world_robustness: float | None
    risk_throttle: str
    explain: str
    explain_code: str
    adjustments: List[str]


class PostMortemEngine:
    """Derives governance-grade summaries after outcomes land."""

    def build(self, events: Iterable[MemoryEvent]) -> Dict[str, PostMortemInsight]:
        insights: Dict[str, PostMortemInsight] = {}
        for event in events:
            if event.outcome is None:
                continue
            doctrine_alignment = {
                key: score for key, score in event.doctrine_scores.items() if score >= 0.5
            }
            adjustments: List[str] = []
            delta = event.outcome - event.conviction
            if delta > 0.1:
                adjustments.append("reward_reinforced")
            elif delta < -0.1:
                adjustments.append("conviction_decay")
            if event.world_robustness is not None and event.world_robustness < 0.4:
                adjustments.append("world_warned")
            if not adjustments:
                adjustments.append("stable")
            insights[event.runner_id] = PostMortemInsight(
                runner_id=event.runner_id,
                conviction=event.conviction,
                model_prob=event.model_prob,
                outcome=event.outcome,
                finish_position=event.finish_position,
                doctrine_alignment=doctrine_alignment,
                world_robustness=event.world_robustness,
                risk_throttle=event.risk_throttle,
                explain=event.explain,
                explain_code=event.explain_code,
                adjustments=adjustments,
            )
        return insights
