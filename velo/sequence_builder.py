"""
VELO v11 Sequence Builder
Builds X_seq tensors for FormSequenceModel (LSTM)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


def build_sequences(
    df: pd.DataFrame,
    seq_len: int = 5,
    feature_cols: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequence tensors from historical race data.
    
    Args:
        df: Historical dataframe with columns:
            - runner_id
            - race_date
            - win (1/0)
            - place (1/0)
            - feature columns (rpr, ts, or, days_since_run, etc.)
        seq_len: Number of historical runs to include
        feature_cols: List of feature column names to use
    
    Returns:
        X_seq: [n_samples, seq_len, n_features]
        y_win: [n_samples]
        y_place: [n_samples]
        runner_ids: [n_samples]
    """
    if feature_cols is None:
        feature_cols = [
            'rpr', 'ts', 'or', 'days_since_run', 
            'distance_diff', 'going_match', 'finishing_pos_norm'
        ]
    
    # Sort by runner and date
    df = df.sort_values(['runner_id', 'race_date']).reset_index(drop=True)
    
    sequences = []
    targets_win = []
    targets_place = []
    runner_ids = []
    
    # Group by runner
    for runner_id, group in df.groupby('runner_id'):
        if len(group) < seq_len + 1:
            continue
        
        # Slide window over historical runs
        for i in range(len(group) - seq_len):
            # Get sequence of past runs
            seq_data = group.iloc[i:i+seq_len][feature_cols].values
            
            # Get next run as target
            next_run = group.iloc[i+seq_len]
            
            sequences.append(seq_data)
            targets_win.append(next_run['win'])
            targets_place.append(next_run['place'])
            runner_ids.append(runner_id)
    
    X_seq = np.array(sequences, dtype=np.float32)
    y_win = np.array(targets_win, dtype=np.float32)
    y_place = np.array(targets_place, dtype=np.float32)
    runner_ids = np.array(runner_ids)
    
    return X_seq, y_win, y_place, runner_ids


def build_sequence_for_runner(
    runner_history: pd.DataFrame,
    seq_len: int = 5,
    feature_cols: List[str] = None
) -> np.ndarray:
    """
    Build sequence tensor for a single runner (race-day inference).
    
    Args:
        runner_history: Historical runs for one runner, sorted by date
        seq_len: Number of runs to include
        feature_cols: Feature columns to use
    
    Returns:
        X_seq: [1, seq_len, n_features] or None if insufficient history
    """
    if feature_cols is None:
        feature_cols = [
            'rpr', 'ts', 'or', 'days_since_run',
            'distance_diff', 'going_match', 'finishing_pos_norm'
        ]
    
    if len(runner_history) < seq_len:
        return None
    
    # Take last seq_len runs
    seq_data = runner_history.iloc[-seq_len:][feature_cols].values
    
    return seq_data.reshape(1, seq_len, -1).astype(np.float32)
