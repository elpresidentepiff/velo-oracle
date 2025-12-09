"""
VELO v11 Intent Feature Builder
Builds X_intent features for IntentModel (Gaussian Mixture)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def build_intent_features(
    df: pd.DataFrame,
    expected_rpr: np.ndarray = None
) -> np.ndarray:
    """
    Build intent features from historical race data.
    
    Features:
        - perf_delta: actual_rpr - expected_rpr
        - market_delta: sp_implied_prob - bfsp_implied_prob
        - trainer_hot_z: trainer recent form z-score
        - jockey_hot_z: jockey recent form z-score
    
    Args:
        df: Historical dataframe with columns:
            - rpr (actual performance)
            - or (official rating, used as baseline expected)
            - sp (starting price)
            - bfsp (betfair starting price, optional)
            - trainer, jockey
        expected_rpr: Optional array of expected RPR from model
    
    Returns:
        X_intent: [n_samples, n_features_intent]
    """
    n = len(df)
    
    # 1. Performance delta
    if expected_rpr is not None:
        perf_delta = df['rpr'].values - expected_rpr
    else:
        # Use OR as baseline expected
        perf_delta = df['rpr'].values - df['or'].values
    
    # 2. Market delta
    sp_implied = 1.0 / df['sp'].values
    if 'bfsp' in df.columns:
        bfsp_implied = 1.0 / df['bfsp'].values
        market_delta = sp_implied - bfsp_implied
    else:
        market_delta = np.zeros(n)
    
    # 3. Trainer hot/cold z-score
    trainer_hot_z = _compute_hot_z(df, 'trainer', window=30)
    
    # 4. Jockey hot/cold z-score
    jockey_hot_z = _compute_hot_z(df, 'jockey', window=30)
    
    # Stack features
    X_intent = np.column_stack([
        perf_delta,
        market_delta,
        trainer_hot_z,
        jockey_hot_z
    ])
    
    return X_intent


def _compute_hot_z(df: pd.DataFrame, entity_col: str, window: int = 30) -> np.ndarray:
    """
    Compute hot/cold z-score for trainer or jockey.
    
    Z-score = (recent_strike_rate - long_term_mean) / long_term_std
    """
    n = len(df)
    z_scores = np.zeros(n)
    
    # Compute rolling strike rates
    df = df.copy()
    df['win_flag'] = (df['finishing_pos'] == 1).astype(int)
    
    for entity in df[entity_col].unique():
        mask = df[entity_col] == entity
        entity_df = df[mask].sort_values('race_date')
        
        # Recent strike rate (last window days)
        recent_sr = entity_df['win_flag'].rolling(window, min_periods=5).mean()
        
        # Long-term mean and std
        long_term_mean = entity_df['win_flag'].expanding(min_periods=20).mean()
        long_term_std = entity_df['win_flag'].expanding(min_periods=20).std()
        
        # Z-score
        z = (recent_sr - long_term_mean) / (long_term_std + 1e-6)
        
        # Assign back
        z_scores[mask] = z.fillna(0).values
    
    return z_scores


def build_intent_for_runner(
    runner_row: pd.Series,
    runner_history: pd.DataFrame,
    expected_rpr: float = None
) -> np.ndarray:
    """
    Build intent features for a single runner (race-day inference).
    
    Args:
        runner_row: Current race row for the runner
        runner_history: Historical runs for context
        expected_rpr: Expected RPR from model
    
    Returns:
        X_intent: [1, n_features_intent]
    """
    # Use most recent run for intent signals
    if len(runner_history) == 0:
        return np.zeros((1, 4))
    
    last_run = runner_history.iloc[-1]
    
    # Performance delta
    if expected_rpr is not None:
        perf_delta = last_run['rpr'] - expected_rpr
    else:
        perf_delta = last_run['rpr'] - last_run['or']
    
    # Market delta
    sp_implied = 1.0 / last_run['sp'] if last_run['sp'] > 0 else 0.0
    bfsp_implied = 1.0 / last_run.get('bfsp', last_run['sp']) if last_run.get('bfsp', last_run['sp']) > 0 else 0.0
    market_delta = sp_implied - bfsp_implied
    
    # Trainer/jockey hot z-scores (simplified)
    trainer_recent_sr = runner_history['win_flag'].tail(10).mean() if 'win_flag' in runner_history else 0.0
    jockey_recent_sr = runner_history['win_flag'].tail(10).mean() if 'win_flag' in runner_history else 0.0
    
    trainer_hot_z = (trainer_recent_sr - 0.15) / 0.1  # Rough normalization
    jockey_hot_z = (jockey_recent_sr - 0.15) / 0.1
    
    X_intent = np.array([[
        perf_delta,
        market_delta,
        trainer_hot_z,
        jockey_hot_z
    ]])
    
    return X_intent
