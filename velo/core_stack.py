"""
VÉLØ v11 – Core Algorithmic Stack
---------------------------------
This file defines:
- Data contracts
- Feature fabric
- Model zoo (baseline + ensembles + clustering)
- SQPE meta-brain (stacking & regime-aware weighting)
- Decision & staking skeleton
- Post-race learning hooks

Dependencies (minimal):
    pip install numpy pandas scikit-learn xgboost
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import brier_score_loss, log_loss

# Optional, but you *will* want it
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------------------------------------------------------------------------
# 1. Data contracts
# ---------------------------------------------------------------------------

@dataclass
class RunnerFeatures:
    """All numeric features for a single runner in a single race."""
    runner_id: str
    race_id: str
    features: np.ndarray  # final feature vector after fabrication
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RaceSample:
    """Collection of runners for a race, with labels if known."""
    race_id: str
    runners: List[RunnerFeatures]
    y_win: Optional[np.ndarray] = None          # 1 if win, else 0
    y_place: Optional[np.ndarray] = None        # 1 if in TBP / Top4, etc.
    y_rpr: Optional[np.ndarray] = None          # numeric performance target


@dataclass
class VeloConfig:
    """Global knobs. Adjust, don’t whine."""
    n_pca_components: int = 20
    n_knn_neighbors: int = 25
    n_clusters_races: int = 6
    n_clusters_horses: int = 8
    random_state: int = 1337

    # Model weights by regime (example defaults)
    weight_linear: float = 0.7
    weight_bayes: float = 0.6
    weight_rf: float = 1.0
    weight_gbm: float = 1.2
    weight_knn: float = 0.6

    # Thresholds
    chaos_cluster_ids: Tuple[int, ...] = (3, 4, 5)  # race clusters treated as "chaos"


# ---------------------------------------------------------------------------
# 2. Feature Fabric
# ---------------------------------------------------------------------------

class FeatureFabric:
    """
    Responsible for:
    - Taking raw race dataframe(s)
    - Building cleaned numeric matrices
    - Managing fit/transform state (encoders, scalers, PCA, etc.)

    Expect raw_df to have columns for:
        - race_id, runner_id
        - numeric stats (RPR history, TS history, OR, days since run, etc.)
        - market stats (odds, volume, slopes)
        - categorical keys (trainer, jockey, owner, track, going, distance_band...)
    """

    def __init__(self, config: VeloConfig):
        self.cfg = config
        self.numeric_cols: List[str] = []
        self.cat_cols: List[str] = []
        self._fitted: bool = False
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
        self._pca: Optional[PCA] = None
        # Simple one-hot encoding index
        self._cat_levels: Dict[str, List[Any]] = {}

    def fit(self, df: pd.DataFrame,
            numeric_cols: List[str],
            cat_cols: List[str]) -> "FeatureFabric":
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols

        # Fit categorical levels
        for col in cat_cols:
            self._cat_levels[col] = sorted(df[col].dropna().unique().tolist())

        # Build raw numeric matrix
        X_num = df[numeric_cols].astype(float).values
        self._means = X_num.mean(axis=0)
        self._stds = X_num.std(axis=0) + 1e-6
        X_scaled = (X_num - self._means) / self._stds

        # Optional PCA to compress correlated blocks
        self._pca = PCA(
            n_components=min(self.cfg.n_pca_components, X_scaled.shape[1]),
            random_state=self.cfg.random_state
        )
        self._pca.fit(X_scaled)
        self._fitted = True
        return self

    def _encode_cats(self, df: pd.DataFrame) -> np.ndarray:
        if not self.cat_cols:
            return np.zeros((len(df), 0))

        mats = []
        for col in self.cat_cols:
            levels = self._cat_levels.get(col, [])
            idx = {val: i for i, val in enumerate(levels)}
            onehot = np.zeros((len(df), len(levels)), dtype=float)
            values = df[col].values
            for r, v in enumerate(values):
                if v in idx:
                    onehot[r, idx[v]] = 1.0
            mats.append(onehot)
        return np.hstack(mats) if mats else np.zeros((len(df), 0))

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[RunnerFeatures]]:
        assert self._fitted, "FeatureFabric must be fitted first."

        X_num = df[self.numeric_cols].astype(float).values
        X_scaled = (X_num - self._means) / self._stds
        X_pca = self._pca.transform(X_scaled) if self._pca is not None else X_scaled

        X_cat = self._encode_cats(df)

        X = np.hstack([X_pca, X_cat])

        runners = []
        for i in range(len(df)):
            runners.append(
                RunnerFeatures(
                    runner_id=str(df.iloc[i]["runner_id"]),
                    race_id=str(df.iloc[i]["race_id"]),
                    features=X[i],
                    metadata={
                        "raw_index": int(df.index[i]),
                        "runner_name": df.iloc[i].get("runner_name", None),
                        "trainer": df.iloc[i].get("trainer", None),
                        "jockey": df.iloc[i].get("jockey", None),
                    },
                )
            )
        return X, runners


# ---------------------------------------------------------------------------
# 3. Model Zoo – core engines
# ---------------------------------------------------------------------------

@dataclass
class ModelOutputs:
    """Predictions from one model for one race."""
    win_proba: np.ndarray
    place_proba: np.ndarray
    expected_rpr: Optional[np.ndarray] = None


class BaseVeloModel:
    """Parent interface; every engine implements this."""

    def fit(self, X: np.ndarray, y_win: np.ndarray,
            y_place: Optional[np.ndarray] = None,
            y_rpr: Optional[np.ndarray] = None):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> ModelOutputs:
        raise NotImplementedError


class BaselineGLM(BaseVeloModel):
    """
    Linear + Logistic regression baseline.
    Gives you honest, boring probabilities: the adult supervision.
    """

    def __init__(self):
        self.clf_win = LogisticRegression(max_iter=1000)
        self.clf_place = LogisticRegression(max_iter=1000)
        self.reg_rpr = LinearRegression()

    def fit(self, X, y_win, y_place=None, y_rpr=None):
        self.clf_win.fit(X, y_win)
        if y_place is not None:
            self.clf_place.fit(X, y_place)
        if y_rpr is not None:
            self.reg_rpr.fit(X, y_rpr)
        return self

    def predict(self, X):
        win_proba = self.clf_win.predict_proba(X)[:, 1]
        place_proba = (
            self.clf_place.predict_proba(X)[:, 1]
            if hasattr(self.clf_place, "coef_")
            else win_proba
        )
        expected_rpr = (
            self.reg_rpr.predict(X)
            if hasattr(self.reg_rpr, "coef_")
            else None
        )
        return ModelOutputs(win_proba, place_proba, expected_rpr)


class TreeEnsembleModel(BaseVeloModel):
    """
    Random Forest + Gradient Boosting (or XGBoost if available).
    This is your main non-linear workhorse.
    """

    def __init__(self, use_xgb: bool = True, random_state: int = 1337):
        self.use_xgb = use_xgb and HAS_XGB
        self.random_state = random_state

        self.rf_win = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
        self.rf_place = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
        self.rf_rpr = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )

        if self.use_xgb:
            self.gbm_win = XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=random_state,
            )
            self.gbm_place = XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=random_state,
            )
            self.gbm_rpr = XGBRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=random_state,
            )
        else:
            self.gbm_win = GradientBoostingClassifier(random_state=random_state)
            self.gbm_place = GradientBoostingClassifier(random_state=random_state)
            self.gbm_rpr = GradientBoostingRegressor(random_state=random_state)

    def fit(self, X, y_win, y_place=None, y_rpr=None):
        self.rf_win.fit(X, y_win)
        if y_place is not None:
            self.rf_place.fit(X, y_place)
        if y_rpr is not None:
            self.rf_rpr.fit(X, y_rpr)

        self.gbm_win.fit(X, y_win)
        if y_place is not None:
            self.gbm_place.fit(X, y_place)
        if y_rpr is not None:
            self.gbm_rpr.fit(X, y_rpr)
        return self

    def predict(self, X):
        win_rf = self.rf_win.predict_proba(X)[:, 1]
        win_gbm = self.gbm_win.predict_proba(X)[:, 1]
        win_proba = 0.4 * win_rf + 0.6 * win_gbm

        if hasattr(self.rf_place, "estimators_"):
            place_rf = self.rf_place.predict_proba(X)[:, 1]
            place_gbm = self.gbm_place.predict_proba(X)[:, 1]
            place_proba = 0.4 * place_rf + 0.6 * place_gbm
        else:
            place_proba = win_proba

        if hasattr(self.rf_rpr, "estimators_"):
            rpr_rf = self.rf_rpr.predict(X)
            rpr_gbm = self.gbm_rpr.predict(X)
            expected_rpr = 0.4 * rpr_rf + 0.6 * rpr_gbm
        else:
            expected_rpr = None

        return ModelOutputs(win_proba, place_proba, expected_rpr)


class BayesKnnModel(BaseVeloModel):
    """
    Naive Bayes + KNN ensemble:
    - Bayes gives quick calibrated-ish probabilities.
    - KNN gives analogue-based empirical frequencies.
    """

    def __init__(self, n_neighbors: int = 25):
        self.nb = GaussianNB()
        self.knn_win = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        self.knn_place = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        self._fitted_place = False

    def fit(self, X, y_win, y_place=None, y_rpr=None):
        self.nb.fit(X, y_win)
        self.knn_win.fit(X, y_win)
        if y_place is not None:
            self.knn_place.fit(X, y_place)
            self._fitted_place = True
        return self

    def predict(self, X):
        win_nb = self.nb.predict_proba(X)[:, 1]
        win_knn = self.knn_win.predict_proba(X)[:, 1]
        win_proba = 0.5 * win_nb + 0.5 * win_knn

        if self._fitted_place:
            place_knn = self.knn_place.predict_proba(X)[:, 1]
            # Use NB as rough proxy for place too
            place_proba = 0.3 * win_nb + 0.7 * place_knn
        else:
            place_proba = win_proba

        return ModelOutputs(win_proba, place_proba, None)


# Backwards-compatible alias
BayesKNNModel = BayesKnnModel


class ClusteringEngines:
    """
    K-Means clustering for race regimes and horse types.
    This is NOT about prediction directly. It shapes how SQPE weights things.
    """

    def __init__(self, cfg: VeloConfig):
        self.cfg = cfg
        self.kmeans_races = KMeans(
            n_clusters=cfg.n_clusters_races,
            random_state=cfg.random_state,
            n_init=10,
        )
        self.kmeans_horses = KMeans(
            n_clusters=cfg.n_clusters_horses,
            random_state=cfg.random_state,
            n_init=10,
        )
        self._fitted = False

    def _race_agg(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """Helper to compute race-level aggregates."""
        agg = race_df.groupby("race_id").agg(
            field_size=("runner_id", "count"),
            avg_or=("or", "mean"),
            std_or=("or", "std"),
            avg_rpr=("last_rpr", "mean"),
            chaos_proxy=("win_odds", "std"),
        ).fillna(0.0)
        return agg

    def fit(self, race_df: pd.DataFrame, X: np.ndarray):
        """
        race_df: 1 row per runner (historical)
        X: feature matrix aligned with df rows
        """
        agg = self._race_agg(race_df)
        self.kmeans_races.fit(agg.values)
        self.kmeans_horses.fit(X)
        self._fitted = True
        return self

    def race_clusters(self, race_df: pd.DataFrame) -> Dict[str, int]:
        assert self._fitted
        agg = self._race_agg(race_df)
        labels = self.kmeans_races.predict(agg.values)
        return {rid: int(lbl) for rid, lbl in zip(agg.index.astype(str), labels)}

    def horse_clusters(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted
        return self.kmeans_horses.predict(X)

    def predict_race_cluster(self, race_df: pd.DataFrame) -> int:
        """
        Predict cluster for a single race (race_day usage).
        race_df: all runners for ONE race (race_id constant).
        """
        assert self._fitted
        agg = self._race_agg(race_df)
        return int(self.kmeans_races.predict(agg.values)[0])

    def predict_horse_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Wrapper for race-day; same as horse_clusters.
        """
        return self.horse_clusters(X)


# ---------------------------------------------------------------------------
# 4. SQPE Meta-Brain v11
# ---------------------------------------------------------------------------

@dataclass
class SQPERunnerOutput:
    """Final fused output for one runner."""
    win_proba: float
    place_proba: float
    expected_rpr: Optional[float]
    chaos_flag: bool
    race_cluster: int
    horse_cluster: int
    model_contributions: Dict[str, float]


class SQPEMetaBrain:
    """
    Stacking + regime-aware weighting of:
        - BaselineGLM
        - TreeEnsembleModel
        - BayesKnnModel
    plus clustering context and chaos logic.
    """

    def __init__(
        self,
        cfg: VeloConfig,
        clustering: ClusteringEngines,
        baseline: Optional[BaselineGLM] = None,
        trees: Optional[TreeEnsembleModel] = None,
        bayes_knn: Optional[BayesKnnModel] = None,
    ):
        self.cfg = cfg
        self.clustering = clustering

        self.baseline = baseline or BaselineGLM()
        self.trees = trees or TreeEnsembleModel(random_state=cfg.random_state)
        self.bayes_knn = bayes_knn or BayesKnnModel(n_neighbors=cfg.n_knn_neighbors)

        # Meta calibrator (simple logistic stacker)
        self.meta_win = LogisticRegression(max_iter=1000)
        self.meta_place = LogisticRegression(max_iter=1000)
        self._meta_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_win: np.ndarray,
        y_place: np.ndarray,
        race_df: pd.DataFrame,
        seq_win_proba: Optional[np.ndarray] = None,
        seq_place_proba: Optional[np.ndarray] = None,
        intent_scalar: Optional[np.ndarray] = None,
        refit_base: bool = True,
    ):
        # Optionally refit base models
        if refit_base:
            self.baseline.fit(X, y_win, y_place)
            self.trees.fit(X, y_win, y_place)
            self.bayes_knn.fit(X, y_win, y_place)

        # Clustering
        self.clustering.fit(race_df, X)
        race_clusters = self.clustering.race_clusters(race_df)
        race_ids = race_df["race_id"].astype(str).values
        race_cluster_vec = np.array([race_clusters[rid] for rid in race_ids])
        horse_cluster_vec = self.clustering.horse_clusters(X)

        # Base predictions
        b = self.baseline.predict(X)
        t = self.trees.predict(X)
        k = self.bayes_knn.predict(X)

        # WIN meta features
        win_feats = [
            b.win_proba,
            t.win_proba,
            k.win_proba,
        ]
        if seq_win_proba is not None:
            win_feats.append(seq_win_proba)
        if intent_scalar is not None:
            win_feats.append(intent_scalar)

        win_feats.extend([race_cluster_vec, horse_cluster_vec])
        meta_X = np.vstack(win_feats).T

        # PLACE meta features
        place_feats = [
            b.place_proba,
            t.place_proba,
            k.place_proba,
        ]
        if seq_place_proba is not None:
            place_feats.append(seq_place_proba)
        if intent_scalar is not None:
            place_feats.append(intent_scalar)

        place_feats.extend([race_cluster_vec, horse_cluster_vec])
        meta_X_place = np.vstack(place_feats).T

        self.meta_win.fit(meta_X, y_win)
        self.meta_place.fit(meta_X_place, y_place)
        self._meta_fitted = True
        return self

    def predict_for_race(
        self,
        race: RaceSample,
        race_cluster: int,
        horse_clusters_for_runners: np.ndarray,
        seq_win_proba: Optional[np.ndarray] = None,
        seq_place_proba: Optional[np.ndarray] = None,
        intent_scalar_for_runners: Optional[np.ndarray] = None,
    ) -> List[SQPERunnerOutput]:
        assert self._meta_fitted, "SQPE must be fitted."

        X = np.vstack([r.features for r in race.runners])

        b = self.baseline.predict(X)
        t = self.trees.predict(X)
        k = self.bayes_knn.predict(X)

        n = len(race.runners)
        rc_vec = np.full(n, race_cluster, dtype=float)
        hc_vec = horse_clusters_for_runners.astype(float)

        # WIN
        win_feats = [
            b.win_proba,
            t.win_proba,
            k.win_proba,
        ]
        if seq_win_proba is not None:
            win_feats.append(seq_win_proba)
        if intent_scalar_for_runners is not None:
            win_feats.append(intent_scalar_for_runners)

        win_feats.extend([rc_vec, hc_vec])
        meta_X = np.vstack(win_feats).T

        # PLACE
        place_feats = [
            b.place_proba,
            t.place_proba,
            k.place_proba,
        ]
        if seq_place_proba is not None:
            place_feats.append(seq_place_proba)
        if intent_scalar_for_runners is not None:
            place_feats.append(intent_scalar_for_runners)

        place_feats.extend([rc_vec, hc_vec])
        meta_X_place = np.vstack(place_feats).T

        fused_win = self.meta_win.predict_proba(meta_X)[:, 1]
        fused_place = self.meta_place.predict_proba(meta_X_place)[:, 1]

        chaos_flag = race_cluster in self.cfg.chaos_cluster_ids

        outputs: List[SQPERunnerOutput] = []
        for i, runner in enumerate(race.runners):
            contributions = {
                "baseline_win": float(b.win_proba[i]),
                "trees_win": float(t.win_proba[i]),
                "bayes_knn_win": float(k.win_proba[i]),
            }
            expected_rpr = None
            if b.expected_rpr is not None or t.expected_rpr is not None:
                if t.expected_rpr is not None:
                    expected_rpr = float(t.expected_rpr[i])
                elif b.expected_rpr is not None:
                    expected_rpr = float(b.expected_rpr[i])

            outputs.append(
                SQPERunnerOutput(
                    win_proba=float(fused_win[i]),
                    place_proba=float(fused_place[i]),
                    expected_rpr=expected_rpr,
                    chaos_flag=chaos_flag,
                    race_cluster=int(race_cluster),
                    horse_cluster=int(horse_clusters_for_runners[i]),
                    model_contributions=contributions,
                )
            )
        return outputs


# ---------------------------------------------------------------------------
# 5. Decision & Staking Layer (skeleton)
# ---------------------------------------------------------------------------

@dataclass
class RunnerDecision:
    runner_id: str
    race_id: str
    runner_name: Optional[str]
    win_proba: float
    place_proba: float
    chaos_flag: bool
    bucket: str           # 'core_strike', 'value_place', 'chaos_longshot', 'fade'
    suggested_bet: str    # description only for now
    notes: str


class RaceDecisionModel:
    """
    Converts SQPE outputs into human / staking decisions.
    This is where your Fade v2 framework, ChaosMode, etc, plug in.
    """

    def __init__(self, cfg: VeloConfig):
        self.cfg = cfg

    def classify_runners(self, sqpe_outputs: List[SQPERunnerOutput]) -> List[str]:
        """
        Very simple rule-based bucket allocation.
        You’ll extend this with proper heuristics + GA / RL later.
        """
        win_probs = np.array([o.win_proba for o in sqpe_outputs])
        place_probs = np.array([o.place_proba for o in sqpe_outputs])

        buckets = []
        for i, o in enumerate(sqpe_outputs):
            p_win = win_probs[i]
            p_place = place_probs[i]

            if o.chaos_flag:
                # In chaos, we focus on place/value, not glory.
                if p_place > 0.35 and p_win < 0.22:
                    buckets.append("value_place")
                elif p_win > 0.20:
                    buckets.append("chaos_longshot")
                else:
                    buckets.append("fade")
            else:
                # Structured races: top probability cluster gets core strike.
                if p_win >= win_probs.max() * 0.9 and p_win > 0.22:
                    buckets.append("core_strike")
                elif p_place > 0.30:
                    buckets.append("value_place")
                else:
                    buckets.append("fade")
        return buckets

    def suggest_bets(self, race: RaceSample,
                     sqpe_outputs: List[SQPERunnerOutput]) -> List[RunnerDecision]:
        buckets = self.classify_runners(sqpe_outputs)

        decisions: List[RunnerDecision] = []
        for runner, out, bucket in zip(race.runners, sqpe_outputs, buckets):
            # Skeleton staking logic, you’ll bolt GA / RL on top of this.
            if bucket == "core_strike":
                suggested = "Win + TBP (core)"
            elif bucket == "value_place":
                suggested = "TBP / Top 4 only"
            elif bucket == "chaos_longshot":
                suggested = "Small Win + TBP (chaos play)"
            else:
                suggested = "No bet / hard fade"

            note_parts = [f"cluster_race={out.race_cluster}",
                          f"cluster_horse={out.horse_cluster}",
                          f"win={out.win_proba:.3f}",
                          f"place={out.place_proba:.3f}",
                          f"chaos={out.chaos_flag}"]

            decisions.append(
                RunnerDecision(
                    runner_id=runner.runner_id,
                    race_id=runner.race_id,
                    runner_name=runner.metadata.get("runner_name"),
                    win_proba=out.win_proba,
                    place_proba=out.place_proba,
                    chaos_flag=out.chaos_flag,
                    bucket=bucket,
                    suggested_bet=suggested,
                    notes=" | ".join(note_parts),
                )
            )
        return decisions


# ---------------------------------------------------------------------------
# 6. Post-Race Learning Hooks (skeleton)
# ---------------------------------------------------------------------------

class PostRaceLearner:
    """
    After results, we log calibration and update diagnostics.
    Full EM / RL update lives here in future.
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def log_race(self, race: RaceSample,
                 sqpe_outputs: List[SQPERunnerOutput]):
        if race.y_win is None or race.y_place is None:
            return

        p_win = np.array([o.win_proba for o in sqpe_outputs])
        p_place = np.array([o.place_proba for o in sqpe_outputs])

        win_brier = brier_score_loss(race.y_win, p_win)
        place_brier = brier_score_loss(race.y_place, p_place)

        try:
            win_logloss = log_loss(race.y_win, p_win, eps=1e-6)
        except ValueError:
            win_logloss = None

        self.history.append(
            {
                "race_id": race.race_id,
                "win_brier": float(win_brier),
                "place_brier": float(place_brier),
                "win_logloss": float(win_logloss) if win_logloss is not None else None,
            }
        )

    def summary(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)
