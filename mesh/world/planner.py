"""World Model Sandbox planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .market_flow import MarketFlowSimulator
from .race_shape import RaceShapeSimulator


@dataclass(slots=True)
class WorldPlanner:
    """Coordinates race and market simulators to assess robustness."""

    race_shape: RaceShapeSimulator
    market_flow: MarketFlowSimulator

    def assess(
        self,
        runners: Sequence[str],
        favourite: str,
        start_price: float,
        trials: int = 100,
        seed: Optional[int] = None,
        timeout_ms: int = 200,
    ) -> Dict[str, object]:
        """Return robustness statistics for the nominated favourite."""

        outcomes = self.race_shape.simulate(
            runners,
            favourite,
            pace_shift=0.0,
            trials=trials,
            seed=seed,
            timeout_ms=timeout_ms,
        )
        wins = sum(1 for outcome in outcomes if outcome[0] == favourite)
        win_rate = wins / max(1, len(outcomes))

        scenarios: List[Dict[str, float]] = [
            {"pace_shift": 0.0, "win_rate": float(win_rate), "trials": float(len(outcomes))}
        ]
        scenario_trials = max(10, trials // 2)
        for idx, shift in enumerate((-0.12, 0.12), start=1):
            scenario_outcomes = self.race_shape.simulate(
                runners,
                favourite,
                pace_shift=shift,
                trials=scenario_trials,
                seed=None if seed is None else seed + 31 * idx,
                timeout_ms=timeout_ms,
            )
            scenario_wins = sum(1 for outcome in scenario_outcomes if outcome[0] == favourite)
            scenarios.append(
                {
                    "pace_shift": float(shift),
                    "win_rate": float(scenario_wins / max(1, len(scenario_outcomes))),
                    "trials": float(len(scenario_outcomes)),
                }
            )

        market_path = self.market_flow.simulate(
            start_price,
            steps=12,
            seed=None if seed is None else seed + 1,
        )
        manipulation_score, spoof_flag = self.market_flow.detect_manipulation(
            market_path["prices"], market_path["volumes"]
        )
        return {
            "win_rate": win_rate,
            "prices": market_path["prices"],
            "volumes": market_path["volumes"],
            "trials": len(outcomes),
            "scenarios": scenarios,
            "manipulation_score": manipulation_score,
            "spoof_flag": spoof_flag,
        }
