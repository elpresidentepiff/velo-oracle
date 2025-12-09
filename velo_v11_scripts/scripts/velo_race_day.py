#!/usr/bin/env python3
"""
VELO v11 RACE MODE - Race Day Inference Script
Loads models, generates predictions, computes stakes, outputs decisions.
"""

import sys
import os
import argparse
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from velo.core_stack import RaceSample, RunnerFeatures, RaceDecisionModel
from velo.advanced_components import FormSequenceModel
from velo.staking import StakingEngine, StakingConfig
from velo.sequence_builder import build_sequence_for_runner
from velo.intent_builder import build_intent_for_runner


def main():
    parser = argparse.ArgumentParser(description='VELO Race Day Inference')
    parser.add_argument('--models-dir', default='./models', help='Directory with trained models')
    parser.add_argument('--race-data', required=True, help='Path to today\'s race data CSV')
    parser.add_argument('--race-id', help='Specific race ID to analyze (optional)')
    parser.add_argument('--bank', type=float, default=1000.0, help='Current bankroll')
    parser.add_argument('--output', default='predictions.json', help='Output file for predictions')
    args = parser.parse_args()
    
    print("=" * 80)
    print("VELO v11 RACE DAY INFERENCE")
    print("=" * 80)
    
    models_dir = Path(args.models_dir)
    
    # 1. Load all artifacts
    print("\n[1/5] Loading models...")
    
    with open(models_dir / 'feature_fabric.pkl', 'rb') as f:
        fabric = pickle.load(f)
    
    with open(models_dir / 'baseline_glm.pkl', 'rb') as f:
        baseline = pickle.load(f)
    
    with open(models_dir / 'trees.pkl', 'rb') as f:
        trees = pickle.load(f)
    
    with open(models_dir / 'bayes_knn.pkl', 'rb') as f:
        bayes_knn = pickle.load(f)
    
    with open(models_dir / 'clustering.pkl', 'rb') as f:
        clustering = pickle.load(f)
    
    with open(models_dir / 'sqpe.pkl', 'rb') as f:
        sqpe = pickle.load(f)
    
    # Load sequence model
    with open(models_dir / 'form_sequence_config.pkl', 'rb') as f:
        seq_config = pickle.load(f)
    
    form_seq_model = FormSequenceModel(**seq_config)
    import torch
    form_seq_model.net.load_state_dict(
        torch.load(models_dir / 'form_sequence.pt', map_location='cpu')
    )
    
    with open(models_dir / 'intent_model.pkl', 'rb') as f:
        intent_model = pickle.load(f)
    
    with open(models_dir / 'ga_theta.pkl', 'rb') as f:
        ga_theta = pickle.load(f)
    
    with open(models_dir / 'rl_agent.pkl', 'rb') as f:
        rl_agent = pickle.load(f)
    
    print("  ✓ All models loaded")
    
    # 2. Load race data
    print("\n[2/5] Loading race data...")
    df = pd.read_csv(args.race_data)
    
    if args.race_id:
        df = df[df['race_id'] == args.race_id]
    
    print(f"  Loaded {len(df)} runners across {df['race_id'].nunique()} races")
    
    # 3. Process each race
    print("\n[3/5] Generating predictions...")
    
    all_predictions = []
    
    for race_id, race_df in df.groupby('race_id'):
        print(f"\n  Race: {race_id}")
        
        # Build features
        X, runners = fabric.transform(race_df)
        
        # Build sequences (if history available)
        seq_win_proba = None
        seq_place_proba = None
        # TODO: Load runner history and build sequences
        
        # Build intent features
        intent_scalar = None
        # TODO: Build intent features from recent runs
        
        # Get race cluster
        race_cluster = clustering.predict_race_cluster(race_df)
        horse_clusters = clustering.predict_horse_clusters(X)
        
        # Build RaceSample
        race_sample = RaceSample(
            race_id=str(race_id),
            runners=runners,
        )
        
        # Get SQPE predictions (seq/intent as TODO for now)
        sqpe_outputs = sqpe.predict_for_race(
            race_sample,
            race_cluster,
            horse_clusters,
            seq_win_proba=None,
            seq_place_proba=None,
            intent_scalar_for_runners=None,
        )
        
        # Get decisions
        decision_model = RaceDecisionModel(sqpe.cfg)
        decisions = decision_model.suggest_bets(race_sample, sqpe_outputs)
        
        # Compute stakes
        staking_cfg = StakingConfig(max_race_exposure_frac=0.10)
        staking_engine = StakingEngine(ga_theta, rl_agent, staking_cfg)
        
        odds_lookup = {
            str(row['runner_id']): row['sp'] 
            for _, row in race_df.iterrows()
        }
        
        stakes = staking_engine.stake_race(decisions, odds_lookup, args.bank)
        
        # Format output
        for (decision, stake) in stakes:
            if stake > 0:
                runner_data = race_df[race_df['runner_id'] == decision.runner_id].iloc[0]
                
                all_predictions.append({
                    'race_id': race_id,
                    'runner_id': decision.runner_id,
                    'runner_name': runner_data.get('runner_name', 'Unknown'),
                    'bucket': decision.bucket,
                    'win_proba': float(decision.win_proba),
                    'place_proba': float(decision.place_proba),
                    'odds': float(odds_lookup.get(str(decision.runner_id), 0)),
                    'stake': float(stake),
                    'edge': float(decision.win_proba - 1.0/odds_lookup.get(str(decision.runner_id), 1))
                })
        
        print(f"    {len([s for _, s in stakes if s > 0])} bets recommended")
    
    # 4. Save output
    print(f"\n[4/5] Saving predictions to {args.output}...")
    
    output_data = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'bank': args.bank,
        'total_races': df['race_id'].nunique(),
        'total_bets': len(all_predictions),
        'total_stake': sum(p['stake'] for p in all_predictions),
        'predictions': all_predictions
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  ✓ Saved {len(all_predictions)} predictions")
    
    # 5. Summary
    print("\n[5/5] Summary:")
    print(f"  Total stake: £{output_data['total_stake']:.2f}")
    print(f"  Exposure: {output_data['total_stake']/args.bank*100:.1f}% of bank")
    
    print("\nRACE DAY INFERENCE COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
