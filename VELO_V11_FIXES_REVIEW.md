# VELO v11 Fixes & Improvements Review

This document outlines the proposed changes to the VELO v11 codebase based on the specification provided in `pasted_content_9.txt`. The goal is to create a production-ready, deterministic, and efficient system.

---

## 1. `velo/core_stack.py` - Core Engine

### 1.1. Bayes Model Naming
- **Change**: Add `BayesKNNModel = BayesKnnModel` alias.
- **Reason**: Ensures backwards compatibility for imports.

### 1.2. ClusteringEngines Refactor
- **Change**: Replace the entire `ClusteringEngines` class.
- **New Implementation**:
    - Consistent `fit(df, X)` method.
    - Separate `race_clusters(df)` and `horse_clusters(X)` methods.
    - New race-day helpers: `predict_race_cluster(race_df)` and `predict_horse_clusters(X)`.
- **Reason**: Improves consistency, simplifies offline vs. race-day logic.

### 1.3. SQPEMetaBrain - Full Integration
- **Change**: Major refactor of `__init__`, `fit`, and `predict_for_race`.
- **New `__init__`**:
    - Accepts optional, pre-fitted `baseline`, `trees`, and `bayes_knn` models.
    - Initializes models if not provided.
- **New `fit`**:
    - Accepts optional `seq_win_proba`, `seq_place_proba`, `intent_scalar`.
    - Optionally refits base models (`refit_base=False` by default).
    - Builds meta-feature matrix by stacking base model predictions, sequence/intent features, and cluster vectors.
- **New `predict_for_race`**:
    - Accepts optional sequence and intent features for the race.
    - Builds meta-feature matrix for the race in the same way as `fit`.
- **Reason**: Creates a single, unified meta-brain that can leverage all v11+ components (LSTM, Intent) when available, while gracefully degrading to a simpler model if not.

### 1.4. RaceDecisionModel API Clean-up
- **Change**: Use `suggest_bets(race, sqpe_outputs)` as the primary public method in scripts.
- **Reason**: Clarifies the API for external callers.

---

## 2. `velo/sequence_builder.py` - LSTM Tensors

- **Change**: Replace the existing `build_sequences` function.
- **New Implementation**:
    - More robust sliding window logic.
    - Handles cases with insufficient history.
    - Returns `X_seq`, `y_win`, `y_place`, and `runner_ids`.
- **Addition**: New `build_sequence_for_runner` helper for race-day inference.
- **Reason**: Correctly and efficiently builds the 3D tensors required for the LSTM model.

---

## 3. `velo/intent_builder.py` - Intent Features

- **Change**: Create this new file to centralize intent feature logic.
- **Implementation**:
    - `build_intent_features(df, expected_rpr)`: Computes performance delta, market delta, and trainer/jockey hot z-scores.
    - `_compute_hot_z(...)`: Helper for calculating z-scores.
- **Reason**: Isolates the logic for creating intent features, making it reusable and easier to maintain.

---

## 4. `velo/staking.py` - Staking Engine

- **Change**: No major changes. The existing implementation is considered fine.
- **Review**: Ensure `StakingConfig` defaults are correct.
- **Reason**: The GA + RL staking engine is already well-defined.

---

## 5. `scripts/` - Operational Scripts

### 5.1. `velo_offline_train.py` (TRAIN MODE)
- **Change**: Update the main training pipeline.
- **New Logic**:
    1. Fit base models (`baseline`, `trees`, `bayes_knn`) separately first.
    2. Instantiate `SQPEMetaBrain` with the pre-fitted models.
    3. Call `sqpe.fit` with `refit_base=False`.
- **Reason**: Ensures base models are trained only once and then passed to the meta-learner, which is more efficient and logically sound.

### 5.2. `velo_race_day.py` (RACE MODE)
- **Change**: Update the race-day prediction loop.
- **New Logic**:
    1. Use the refactored `clustering.predict_race_cluster(race_df)`.
    2. Pass `None` for sequence and intent features to `sqpe.predict_for_race` (as a TODO for full data plumbing).
    3. Call `decision_model.suggest_bets(...)`.
- **Reason**: Aligns the script with the new, more robust APIs in the core modules.

### 5.3. `velo_post_race.py` (POST-RACE MODE)
- **Change**: No major changes.
- **Review**: Ensure column names from `predictions.json` and `results.csv` match.
- **Reason**: The existing logic for P&L calculation, RL updates, and calibration is sound.

---

## 6. Production Polish

- **`README.md`**: Update to reflect the new 3-mode script-based workflow.
- **`MANUS_OPERATING_MANUAL.md`**: Clarify the strict separation of OFFLINE, RACE, and POST-RACE modes.
- **`tests/`**: (Optional) Add basic sanity tests to verify data shapes and prevent regressions.

---

## Summary of Impact

These changes will transform the VELO v11 codebase from a collection of components into a cohesive, production-grade system. The key benefits are:

- **Unified Architecture**: A single, powerful `SQPEMetaBrain` that integrates all models.
- **Clear Separation of Concerns**: Dedicated modules for sequences, intent, and staking.
- **Deterministic Execution**: Three distinct, single-purpose scripts for training, inference, and learning.
- **Improved Efficiency**: Base models are trained once, reducing redundant computation.
- **Production Ready**: The system is more robust, maintainable, and easier to deploy.

**This is the blueprint for a VC-ready engine.**
