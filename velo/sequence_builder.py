# velo/sequence_builder.py

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build LSTM sequences from historical flat table.

    Returns:
        X_seq:      [n_seq, seq_len, n_features]
        y_win:      [n_seq]
        y_place:    [n_seq]
        runner_ids: [n_seq]
    """
    df = df.sort_values(["runner_id", "race_date"])
    X_list, y_win_list, y_place_list, rid_list = [], [], [], []

    for rid, g in df.groupby("runner_id"):
        g = g.reset_index(drop=True)
        feats = g[feature_cols].values
        win = g["win"].values
        place = g["place"].values

        if len(g) <= seq_len:
            continue

        for i in range(seq_len, len(g)):
            X_list.append(feats[i - seq_len:i])
            y_win_list.append(win[i])
            y_place_list.append(place[i])
            rid_list.append(rid)

    if not X_list:
        raise ValueError("No sequences built; check seq_len or data coverage.")

    X_seq = np.stack(X_list, axis=0)
    y_seq_win = np.array(y_win_list, dtype=float)
    y_seq_place = np.array(y_place_list, dtype=float)
    runner_ids = np.array(rid_list)

    return X_seq, y_seq_win, y_seq_place, runner_ids


def build_sequence_for_runner(
    runner_history: pd.DataFrame,
    seq_len: int,
    feature_cols: List[str],
) -> Optional[np.ndarray]:
    """
    Build sequence for a single runner at race-day.
    
    Returns:
        X_seq: [1, seq_len, n_features] or None if insufficient history
    """
    runner_history = runner_history.sort_values("race_date")
    if len(runner_history) < seq_len:
        return None
    feats = runner_history[feature_cols].values
    return feats[-seq_len:].reshape(1, seq_len, len(feature_cols))
