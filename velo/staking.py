"""
VELO v11 Staking Engine
Combines GA + RL with hard exposure caps
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class StakingConfig:
    starting_bank: float = 1000.0
    max_daily_loss_frac: float = 0.25
    max_race_exposure_frac: float = 0.10


class StakingEngine:
    """
    Combines GA-optimized staking with RL caps and hard exposure limits.
    """
    
    def __init__(self,
                 ga_theta: Dict[str, float],
                 rl_agent,  # RLStakingAgent
                 cfg: StakingConfig):
        self.theta = ga_theta
        self.rl = rl_agent
        self.cfg = cfg
    
    def stake_race(
        self,
        decisions: List,  # List[RunnerDecision]
        odds_lookup: Dict[str, float],
        bank: float,
    ) -> List[Tuple[object, float]]:
        """
        Compute stakes for all runners in a race.
        
        Returns:
            List of (RunnerDecision, stake) tuples
        """
        # Compute raw stakes
        stakes_raw = []
        for d in decisions:
            odds = float(odds_lookup.get(d.runner_id, 0.0))
            
            # Skip invalid odds or fade bucket
            if odds <= 1.01 or d.bucket == "fade":
                stakes_raw.append((d, 0.0))
                continue
            
            # GA base stake
            stake_base = self._ga_stake(
                d.bucket, d.win_proba, d.place_proba, odds, bank
            )
            
            # RL cap
            stake_rl, _ = self.rl.stake_for_runner(
                d.bucket, odds, bank, train=False
            )
            
            # Take minimum
            stake = min(stake_base, stake_rl)
            stakes_raw.append((d, stake))
        
        # Apply race-level exposure cap
        total = sum(s for _, s in stakes_raw)
        cap = self.cfg.max_race_exposure_frac * bank
        scale = 1.0 if total <= cap else cap / total
        
        return [(d, s * scale) for d, s in stakes_raw]
    
    def _ga_stake(self, bucket: str, p_win: float, p_place: float, 
                  odds: float, bank: float) -> float:
        """
        Compute GA-optimized stake for a runner.
        """
        # Extract bucket-specific params
        k_core = self.theta.get("k_core", 0.5)
        k_value = self.theta.get("k_value", 0.5)
        k_chaos = self.theta.get("k_chaos", 0.3)
        alpha_edge = self.theta.get("alpha_edge", 1.0)
        max_frac = self.theta.get("max_frac_bank", 0.10)
        
        # Select multiplier based on bucket
        if bucket == "core_strike":
            k = k_core
        elif bucket == "value_place":
            k = k_value
        elif bucket == "chaos_longshot":
            k = k_chaos
        else:
            return 0.0
        
        # Compute edge
        implied_prob = 1.0 / odds if odds > 1.0 else 0.0
        edge = max(0.0, p_win - implied_prob)
        
        # Kelly-style stake
        if edge > 0 and odds > 1.0:
            kelly_frac = edge / (odds - 1.0)
            stake_frac = k * (edge ** alpha_edge) * kelly_frac
            stake_frac = min(stake_frac, max_frac)
            return stake_frac * bank
        
        return 0.0
