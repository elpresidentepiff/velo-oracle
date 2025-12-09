# VELO v11 - Complete Integration

## Structure

```
velo/
├── __init__.py
├── core_stack.py          # Base models, SQPE, clustering, decisions
├── advanced_components.py # LSTM, Intent GMM, GA, RL
├── staking.py             # Staking engine with GA + RL
├── sequence_builder.py    # Build X_seq tensors for LSTM
└── intent_builder.py      # Build X_intent features for GMM

scripts/
├── velo_offline_train.py  # OFFLINE MODE - Train all models
├── velo_race_day.py       # RACE MODE - Generate predictions
└── velo_post_race.py      # POST-RACE MODE - Log results, update RL
```

## Usage

### 1. OFFLINE MODE - Training

Train all models on historical data:

```bash
python scripts/velo_offline_train.py \
  --data /path/to/historical_races.csv \
  --output-dir ./models \
  --seq-len 5
```

**This will:**
- Build features using FeatureFabric
- Train base models (GLM, Trees, Bayes/KNN)
- Train FormSequenceModel (LSTM)
- Train IntentModel (GMM)
- Train SQPE meta-brain
- Optimize staking with GA
- Train RL agent
- Save all artifacts to `./models/`

**Artifacts saved:**
- `feature_fabric.pkl`
- `baseline_glm.pkl`, `trees.pkl`, `bayes_knn.pkl`
- `clustering.pkl`
- `sqpe.pkl`
- `form_sequence.pt` (PyTorch)
- `intent_model.pkl`
- `ga_theta.pkl`
- `rl_agent.pkl`

---

### 2. RACE MODE - Inference

Generate predictions for today's races:

```bash
python scripts/velo_race_day.py \
  --models-dir ./models \
  --race-data /path/to/todays_races.csv \
  --bank 1000.0 \
  --output predictions.json
```

**This will:**
- Load all trained models
- Build features for today's runners
- Generate SQPE predictions
- Classify into buckets (core_strike, value_place, chaos_longshot, fade)
- Compute stakes using GA + RL with exposure caps
- Output JSON with predictions and stakes

**Output format:**
```json
{
  "timestamp": "2025-12-08T23:00:00",
  "bank": 1000.0,
  "total_races": 5,
  "total_bets": 12,
  "total_stake": 87.50,
  "predictions": [
    {
      "race_id": "newcastle_1518",
      "runner_id": "12345",
      "runner_name": "Fast Horse",
      "bucket": "core_strike",
      "win_proba": 0.35,
      "place_proba": 0.62,
      "odds": 3.5,
      "stake": 15.20,
      "edge": 0.064
    }
  ]
}
```

---

### 3. POST-RACE MODE - Learning

Log results and update RL agent:

```bash
python scripts/velo_post_race.py \
  --models-dir ./models \
  --predictions predictions.json \
  --results /path/to/race_results.csv \
  --history ./results_history.csv
```

**This will:**
- Match predictions with actual results
- Compute P&L and ROI
- Update RL Q-table based on realized rewards
- Append to results history
- Print calibration metrics (Brier score, calibration by bin)

---

## Data Requirements

### Historical Data (for training)

CSV/Parquet with columns:
- `race_id`, `runner_id`, `race_date`
- `win` (1/0), `place` (1/0)
- `rpr`, `ts`, `or` (ratings)
- `sp`, `bfsp` (odds)
- `trainer`, `jockey`, `track`, `going`, `distance_band`
- `age`, `weight`, `draw`, `days_since_run`, `distance`, `field_size`
- `finishing_pos`, `finishing_pos_norm`
- `distance_diff`, `going_match` (for sequences)

### Race Day Data

Same format as historical, but without `win`, `place`, `finishing_pos` (unknown)

### Results Data

CSV with columns:
- `race_id`, `runner_id`
- `finishing_pos`, `win`, `place`

---

## Governance (Manus Operating Manual)

See `MANUS_OPERATING_MANUAL.md` for complete rules.

**Key principles:**
- **2 MODES ONLY**: OFFLINE (training) and RACE (inference)
- **Single-pass operations**: Load once, train once, save once, exit
- **No improvisation**: Execute task, return result, stop
- **Credit protection**: No nested LLM loops, no self-diagnosis rambles

**Workflow:**
1. **OFFLINE**: `velo_offline_train.py` → Train → Save → EXIT
2. **RACE**: `velo_race_day.py` → Load → Predict → Output → EXIT
3. **POST-RACE**: `velo_post_race.py` → Log → Update RL → EXIT

---

## Dependencies

```bash
pip install numpy pandas scikit-learn xgboost torch
```

---

## Integration Status

✅ Core stack implemented
✅ Advanced components implemented  
✅ SQPE wiring for LSTM + Intent
✅ Sequence builder
✅ Intent builder
✅ Staking engine (GA + RL)
✅ 3 operational scripts
✅ Proper module structure

**READY TO USE**
