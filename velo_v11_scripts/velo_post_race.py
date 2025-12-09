#!/usr/bin/env python3
"""
VELO v11 POST-RACE MODE - Results Logging and RL Update Script
Logs results, updates RL Q-table, computes calibration metrics.
"""

import sys
import os
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from velo.advanced_components import RLStakingAgent


def main():
    parser = argparse.ArgumentParser(description='VELO Post-Race Learning')
    parser.add_argument('--models-dir', default='./models', help='Directory with trained models')
    parser.add_argument('--predictions', required=True, help='Path to predictions JSON')
    parser.add_argument('--results', required=True, help='Path to race results CSV')
    parser.add_argument('--history', default='./results_history.csv', help='Results history file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("VELO v11 POST-RACE LEARNING")
    print("=" * 80)
    
    models_dir = Path(args.models_dir)
    
    # 1. Load predictions and results
    print("\n[1/4] Loading predictions and results...")
    
    import json
    with open(args.predictions, 'r') as f:
        predictions_data = json.load(f)
    
    predictions = pd.DataFrame(predictions_data['predictions'])
    results = pd.read_csv(args.results)
    
    print(f"  {len(predictions)} predictions")
    print(f"  {len(results)} results")
    
    # 2. Match predictions with results
    print("\n[2/4] Computing P&L...")
    
    merged = predictions.merge(
        results[['race_id', 'runner_id', 'finishing_pos', 'win', 'place']],
        on=['race_id', 'runner_id'],
        how='left'
    )
    
    # Compute P&L
    merged['win_return'] = merged.apply(
        lambda row: row['stake'] * row['odds'] if row['win'] == 1 else 0,
        axis=1
    )
    merged['loss'] = merged.apply(
        lambda row: row['stake'] if row['win'] == 0 else 0,
        axis=1
    )
    merged['pnl'] = merged['win_return'] - merged['stake']
    
    total_staked = merged['stake'].sum()
    total_pnl = merged['pnl'].sum()
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
    
    print(f"  Total staked: £{total_staked:.2f}")
    print(f"  Total P&L: £{total_pnl:.2f}")
    print(f"  ROI: {roi:.1f}%")
    
    # 3. Update RL agent
    print("\n[3/4] Updating RL agent...")
    
    with open(models_dir / 'rl_agent.pkl', 'rb') as f:
        rl_agent = pickle.load(f)
    
    bank = predictions_data['bank']
    
    for _, row in merged.iterrows():
        if pd.notna(row['finishing_pos']):
            # Compute reward
            reward = row['pnl'] / bank
            
            # Update Q-table
            rl_agent.update(
                bucket=row['bucket'],
                odds=row['odds'],
                stake_frac=row['stake'] / bank,
                reward=reward
            )
    
    # Save updated agent
    with open(models_dir / 'rl_agent.pkl', 'wb') as f:
        pickle.dump(rl_agent, f)
    
    print("  ✓ RL Q-table updated")
    
    # 4. Append to history
    print("\n[4/4] Updating results history...")
    
    history_file = Path(args.history)
    
    if history_file.exists():
        history = pd.read_csv(history_file)
        history = pd.concat([history, merged], ignore_index=True)
    else:
        history = merged
    
    history.to_csv(history_file, index=False)
    print(f"  ✓ History saved to {history_file}")
    
    # Calibration metrics
    print("\n[CALIBRATION]")
    
    # Brier score
    brier = np.mean((merged['win_proba'] - merged['win']) ** 2)
    print(f"  Brier score: {brier:.4f}")
    
    # Calibration by probability bin
    merged['prob_bin'] = pd.cut(merged['win_proba'], bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0])
    calib = merged.groupby('prob_bin').agg({
        'win_proba': 'mean',
        'win': 'mean',
        'runner_id': 'count'
    }).rename(columns={'runner_id': 'count'})
    
    print("\n  Calibration by bin:")
    print(calib)
    
    print("\nPOST-RACE LEARNING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
