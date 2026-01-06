"""FastAPI server exposing the Cognitive Mesh decision endpoint."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from mesh.governance import PostMortemEngine
from mesh.liquid import LiquidCell, OnlineTrainer
from mesh.memory import EpisodicMemory, MemoryEvent
from mesh.neosym import DoctrineExecutor, load_default_doctrines
from mesh.orchestrator import ConvictionState, OrchestratorConfig, RiskBudget, StateOrchestrator
from mesh.vector_fusion import (
    IdeaStepper,
    StateMixer,
    VectorFusionCore,
    build_default_vector_fusion_core,
)
from mesh.world import MarketFlowSimulator, RaceShapeSimulator, WorldPlanner


LOGGER = logging.getLogger("mesh.api")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_SCHEMA_VERSION = "1.0.0"
TRACE_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"
LOG_DIR = Path("logs/decisions")
PARQUET_DIR = Path("memory/episodic")
LOG_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VÉLØ Cognitive Mesh", version="1.0.0")


class MarketSnapshot(BaseModel):
    """Market context shared across runners."""

    prices: Dict[str, float] = Field(..., description="Runner price map")
    liquidity: float = Field(..., ge=0.0)
    timestamp: str


class RunnerInput(BaseModel):
    """Runner payload validated by FastAPI."""

    runner_id: str
    features: Dict[str, float]

    @field_validator("features", mode="before")
    def _ensure_float_values(cls, value: Mapping[str, object]) -> Mapping[str, float]:
        try:
            return {key: float(val) for key, val in value.items()}
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("All feature values must be numeric") from exc


class RequestOptions(BaseModel):
    """Optional request tunables."""

    rollouts: int = Field(256, ge=0, le=512)
    seed: Optional[int] = None
    ablate: Sequence[str] = Field(default_factory=tuple)
    governance: bool = False

    @field_validator("ablate")
    def _validate_ablation(cls, value: Sequence[str]) -> Sequence[str]:
        allowed = {"vector", "neosym", "world"}
        invalid = [flag for flag in value if flag not in allowed]
        if invalid:
            raise ValueError(f"Unknown ablation flags: {', '.join(invalid)}")
        # ensure deterministic ordering without duplicates
        seen = []
        for flag in value:
            if flag not in seen:
                seen.append(flag)
        return tuple(seen)


class DecisionRequest(BaseModel):
    """Contract enforced for the /decide endpoint."""

    schema_version: str = Field(DEFAULT_SCHEMA_VERSION, description="Request schema version")
    race_id: str
    runners: List[RunnerInput]
    market: MarketSnapshot
    options: RequestOptions = Field(default_factory=RequestOptions)

    @model_validator(mode="after")
    def _validate_payload(self) -> "DecisionRequest":
        if self.market is None:
            raise ValueError("market snapshot is required")
        prices: Mapping[str, float] = self.market.prices
        for runner in self.runners:
            if runner.runner_id not in prices:
                raise ValueError(f"Missing market price for runner '{runner.runner_id}'")
        return self


class RiskTrace(BaseModel):
    throttle: str
    notes: List[str]


class WorldTrace(BaseModel):
    robustness: Optional[float]
    rollouts: int
    budget_ms: int
    status: str
    prices: List[float] = Field(default_factory=list)
    volumes: List[float] = Field(default_factory=list)
    scenarios: List[Dict[str, float]] = Field(default_factory=list)
    manipulation_score: float = 0.0
    spoof_flag: bool = False


class CounterfactualTrace(BaseModel):
    """Counterfactual metrics if a strike was withheld."""

    state: str
    conviction: float
    s3_positions: int
    throttle: str


class ConvictionTrace(BaseModel):
    runner_id: str
    state: str
    conviction: float
    model_prob: float
    doctrine_scores: Dict[str, float]
    doctrine_veto: bool
    world: WorldTrace
    risk: RiskTrace
    explain: str
    explain_code: str
    governance: Optional["GovernanceNote"] = None
    counterfactual: CounterfactualTrace


class RunLineage(BaseModel):
    model_sha: str
    feature_schema_hash: str
    doctrine_pack_version: str
    vector_fusion_version: str


class DecisionResponse(BaseModel):
    schema_version: str
    trace_id: str
    run: RunLineage
    convictions: List[ConvictionTrace]
    risk_throttle: str
    artifacts: Dict[str, str]


class GovernanceNote(BaseModel):
    audit_ref: str
    state_transition: Dict[str, str]
    doctrine_alignment: Dict[str, float]
    risk_throttle: str
    world_status: str
    notes: List[str]


class OutcomeRunner(BaseModel):
    runner_id: str
    finish_position: Optional[int] = None
    win: bool = False
    payout: Optional[float] = None


class OutcomeIngestion(BaseModel):
    schema_version: str = Field(DEFAULT_SCHEMA_VERSION, description="Request schema version")
    race_id: str
    timestamp: str
    results: List[OutcomeRunner]

    @model_validator(mode="after")
    def _ensure_results(self) -> "OutcomeIngestion":
        if not self.results:
            raise ValueError("results payload cannot be empty")
        return self


class PostMortemRunner(BaseModel):
    runner_id: str
    conviction: float
    model_prob: float
    outcome: float
    finish_position: Optional[int]
    doctrine_alignment: Dict[str, float]
    world_robustness: Optional[float]
    risk_throttle: str
    explain: str
    explain_code: str
    adjustments: List[str]


class OutcomeResponse(BaseModel):
    schema_version: str
    race_id: str
    processed: List[PostMortemRunner]


@dataclass(slots=True)
class RunnerDecisionState:
    """Cached per-runner state used to preserve temporal continuity."""

    state_vector: np.ndarray
    state: ConvictionState
    strike_ready_ticks: int
    last_conviction: float
    last_reconfirmed_tick: int
    bias_heat: float
    baseline_score: float


@dataclass(slots=True)
class EngineLineage:
    model_sha: str
    feature_schema_hash: str
    doctrine_pack_version: str
    vector_fusion_version: str


@dataclass(slots=True)
class RiskRails:
    """Basic kill-switch configuration for Phase 1 hardening."""

    calibration_drift_limit: float = 0.02
    max_drawdown_ratio: float = 0.2
    liquidity_floor: float = 50000.0

    calibration_drift: float = 0.0
    max_drawdown: float = 0.0
    equity: float = 0.0
    peak_equity: float = 0.0

    def evaluate(self, liquidity: float) -> Tuple[str, List[str]]:
        """Return throttle state and supporting notes."""

        notes: List[str] = []
        throttle = "ON"
        if liquidity < self.liquidity_floor:
            throttle = "OFF"
            notes.append("liq_floor_breach")
        else:
            notes.append("liq_ok")

        if self.calibration_drift > self.calibration_drift_limit:
            throttle = "OFF"
            notes.append("calibration_drift")
        else:
            notes.append("cal_ok")

        if self.max_drawdown > self.max_drawdown_ratio and throttle != "OFF":
            throttle = "REDUCE"
            notes.append("dd_high")
        elif throttle != "OFF":
            notes.append("dd_ok")

        return throttle, notes

    def register_outcome(self, *, reward: float, conviction: float, model_prob: float) -> None:
        """Update calibration and drawdown trackers."""

        error = abs(reward - model_prob)
        self.calibration_drift = 0.9 * self.calibration_drift + 0.1 * error

        pnl = reward - conviction
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        peak = max(self.peak_equity, 1e-6)
        drawdown = (self.peak_equity - self.equity) / peak
        self.max_drawdown = max(self.max_drawdown, drawdown)


@dataclass(slots=True)
class DecisionMetrics:
    """Tracks Prometheus-style counters for observability."""

    decisions_total: int = 0
    s3_decisions: int = 0
    doctrine_fires: MutableMapping[str, int] = field(default_factory=dict)
    rollout_runtime_ms: List[int] = field(default_factory=list)
    reversal_events: int = 0
    manipulation_flags: int = 0
    entropy_clamps: int = 0

    def increment_doctrine(self, doctrine_id: str, fired: bool) -> None:
        if doctrine_id not in self.doctrine_fires:
            self.doctrine_fires[doctrine_id] = 0
        if fired:
            self.doctrine_fires[doctrine_id] += 1


class DecisionLogger:
    """Writes decision traces to NDJSON and parquet mirrors."""

    def __init__(self, log_dir: Path, parquet_dir: Path) -> None:
        self._log_dir = log_dir
        self._parquet_dir = parquet_dir

    def write(self, record: Mapping[str, object]) -> None:
        trace_path = self._log_dir / "decisions.ndjson"
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + os.linesep)

        parquet_path = self._parquet_dir / "decisions.parquet"
        frame = pd.DataFrame([record])
        try:
            frame.to_parquet(parquet_path, engine="pyarrow", index=False)
        except Exception as exc:  # pragma: no cover - optional dependency guard
            LOGGER.warning("parquet_write_failed", extra={"error": str(exc)})


@dataclass(slots=True)
class DecisionEngine:
    """Coordinates Cognitive Mesh components to serve decisions."""

    vector_core: VectorFusionCore
    stepper: IdeaStepper
    doctrine_executor: DoctrineExecutor
    orchestrator: StateOrchestrator
    world_planner: WorldPlanner
    memory: EpisodicMemory
    trainer: OnlineTrainer
    risk_budget: RiskBudget
    lineage: EngineLineage
    logger: DecisionLogger
    risk_rails: RiskRails
    metrics: DecisionMetrics
    post_mortem: PostMortemEngine
    _state_cache: MutableMapping[str, RunnerDecisionState] = field(default_factory=dict)
    _zero_vector: np.ndarray = field(init=False, repr=False)
    _strike_ready_cache: MutableMapping[str, int] = field(init=False, repr=False)
    _tick_counter: int = 0

    def __post_init__(self) -> None:
        self._zero_vector = self.vector_core.encode({}) * 0.0
        self._strike_ready_cache = {}

    def reset(self) -> None:
        """Clear cached state for deterministic testing and replays."""

        self._state_cache.clear()
        self.memory.clear()
        self.trainer.reset()
        self.risk_budget.set_active(0)
        self.metrics = DecisionMetrics()
        self._strike_ready_cache.clear()
        self.risk_rails.calibration_drift = 0.0
        self.risk_rails.max_drawdown = 0.0
        self.risk_rails.equity = 0.0
        self.risk_rails.peak_equity = 0.0
        self._tick_counter = 0

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Evaluate the request and return conviction payloads."""

        if not request.runners:
            raise HTTPException(status_code=400, detail="runners payload cannot be empty")

        base_throttle, base_notes = self.risk_rails.evaluate(request.market.liquidity)
        active_positions = sum(
            1 for state in self._state_cache.values() if state.state == ConvictionState.STRIKE
        )
        self.risk_budget.set_active(active_positions)

        self._tick_counter += 1
        ablate = set(request.options.ablate)
        runner_ids = [runner.runner_id for runner in request.runners]
        convictions: List[ConvictionTrace] = []
        start_time = time.perf_counter_ns()
        trace_id = self._build_trace_id()

        governance_mode = request.options.governance
        aggregate_throttle = base_throttle

        for runner in request.runners:
            previous = self._state_cache.get(runner.runner_id)

            fused = self._encode_features(runner.features, ablate)
            previous_vector = previous.state_vector if previous else None
            state_vector = self.stepper.step(previous_vector, fused)

            intensity, baseline_score, baseline_prob = self._compute_baseline_metrics(state_vector)
            doctrine_scores, veto = self._evaluate_doctrines(
                runner_features=runner.features,
                baseline_prob=baseline_prob,
                implied_prob=self._implied_probability(request.market.prices[runner.runner_id]),
                ablate_neosym="neosym" in ablate,
            )
            combined_score = self._combine_scores(baseline_score, doctrine_scores, ablate)
            combined_score, reconfirm_tick, decay_triggered = self._apply_confidence_decay(
                conviction=combined_score,
                baseline_score=baseline_score,
                previous=previous,
            )

            previous_state = previous.state if previous else ConvictionState.SCAN
            allow_strike = self._update_strike_ready(runner.runner_id, combined_score)
            if base_throttle == "OFF":
                allow_strike = False

            state = self.orchestrator.transition(
                previous_state,
                combined_score,
                doctrine_veto=veto,
                allow_strike=allow_strike,
            )
            entered_strike = (
                previous_state != ConvictionState.STRIKE
                and state == ConvictionState.STRIKE
                and allow_strike
            )

            world_trace = self._assess_world(
                runner_id=runner.runner_id,
                runner_ids=runner_ids,
                start_price=request.market.prices[runner.runner_id],
                rollouts=request.options.rollouts,
                seed=request.options.seed,
                ablate_world="world" in ablate,
            )

            world_robustness = world_trace.robustness if world_trace.robustness is not None else 0.0
            model_prob = float(np.clip((baseline_prob + world_robustness) / 2.0, 0.01, 0.99))

            combined_score, ego_triggered = self._apply_ego_kill_switch(
                conviction=combined_score,
                baseline_score=baseline_score,
                doctrine_scores=doctrine_scores,
                world_trace=world_trace,
                previous=previous,
            )

            combined_score, bias_triggered = self._apply_bias_quarantine(
                conviction=combined_score,
                baseline_score=baseline_score,
                world_trace=world_trace,
                previous=previous,
            )

            combined_score, entropy_triggered, entropy_score = self._apply_memory_entropy_guard(
                conviction=combined_score,
                baseline_score=baseline_score,
            )

            inversion_triggered = self._scenario_inversion_guard(
                baseline_score=baseline_score,
                conviction=combined_score,
                doctrine_scores=doctrine_scores,
            )

            runner_throttle = base_throttle
            runner_notes = list(base_notes)
            if decay_triggered:
                runner_notes.append("confidence_decay")
            if ego_triggered:
                runner_notes.append("ego_clamp")
            if bias_triggered:
                runner_notes.append("bias_quarantine")
                runner_throttle = self._resolve_throttle(runner_throttle, "REDUCE")
            if entropy_triggered:
                runner_notes.append("memory_entropy_low")
                runner_throttle = self._resolve_throttle(runner_throttle, "REDUCE")
                self.metrics.entropy_clamps += 1
            if inversion_triggered:
                runner_notes.append("scenario_inversion")
                runner_throttle = self._resolve_throttle(runner_throttle, "OFF")

            runner_throttle, manipulation_notes = self._apply_market_manipulation_guard(
                throttle=runner_throttle,
                world_trace=world_trace,
                liquidity=request.market.liquidity,
            )
            if manipulation_notes:
                runner_notes.extend(manipulation_notes)
                self.metrics.manipulation_flags += 1

            aggregate_throttle = self._resolve_throttle(aggregate_throttle, runner_throttle)

            gradient = np.full_like(self.trainer.weights, combined_score, dtype=np.float32)
            self.trainer.update(gradient, calibration_delta=world_robustness - model_prob)

            counterfactual_state = self._simulate_transition(
                previous_state=previous_state,
                conviction=combined_score,
                doctrine_veto=veto,
                allow_strike=False,
            )

            explanation, explain_code = self._build_explanation(
                state,
                baseline_score,
                doctrine_scores,
                world_trace,
            )

            governance_note = (
                self._build_governance_note(
                    trace_id=trace_id,
                    runner_id=runner.runner_id,
                    previous_state=previous_state,
                    new_state=state,
                    doctrine_scores=doctrine_scores,
                    throttle=runner_throttle,
                    world_trace=world_trace,
                )
                if governance_mode
                else None
            )

            counterfactual = CounterfactualTrace(
                state=counterfactual_state.value,
                conviction=combined_score,
                s3_positions=max(
                    0,
                    self.risk_budget.active_positions - (1 if entered_strike else 0),
                ),
                throttle=base_throttle,
            )

            conviction = ConvictionTrace(
                runner_id=runner.runner_id,
                state=state.value,
                conviction=combined_score,
                model_prob=model_prob,
                doctrine_scores=doctrine_scores,
                doctrine_veto=veto,
                world=world_trace,
                risk=RiskTrace(throttle=runner_throttle, notes=runner_notes),
                explain=explanation,
                explain_code=explain_code,
                governance=governance_note,
                counterfactual=counterfactual,
            )
            convictions.append(conviction)

            bias_heat = self._update_bias_heat(
                previous_state=previous,
                conviction=combined_score,
                baseline_score=baseline_score,
            )

            self._state_cache[runner.runner_id] = RunnerDecisionState(
                state_vector=state_vector.copy(),
                state=state,
                strike_ready_ticks=self._strike_ready.get(runner.runner_id, 0),
                last_conviction=combined_score,
                last_reconfirmed_tick=reconfirm_tick,
                bias_heat=bias_heat,
                baseline_score=baseline_score,
            )

            state_notes = [
                f"allow_strike={allow_strike}",
                f"throttle={runner_throttle}",
                f"world_status={world_trace.status}",
                f"counterfactual={counterfactual_state.value}",
                f"bias_heat={bias_heat:.2f}",
                f"memory_entropy={entropy_score:.2f}",
                f"manipulation_score={world_trace.manipulation_score:.2f}",
                f"spoof_flag={world_trace.spoof_flag}",
            ]
            if decay_triggered:
                state_notes.append("confidence_decay")
            if ego_triggered:
                state_notes.append("ego_clamp")
            if bias_triggered:
                state_notes.append("bias_quarantine")
            if inversion_triggered:
                state_notes.append("scenario_inversion")
            if entropy_triggered:
                state_notes.append("memory_entropy_low")
            if manipulation_notes:
                state_notes.extend(manipulation_notes)

            self.memory.record(
                MemoryEvent(
                    trace_id=trace_id,
                    race_id=request.race_id,
                    runner_id=runner.runner_id,
                    timestamp=request.market.timestamp,
                    state=state.value,
                    previous_state=previous_state.value,
                    conviction=combined_score,
                    model_prob=model_prob,
                    doctrine_scores=doctrine_scores,
                    doctrine_veto=veto,
                    world_robustness=world_trace.robustness,
                    risk_throttle=runner_throttle,
                    market_snapshot={
                        "price": request.market.prices[runner.runner_id],
                        "liquidity": request.market.liquidity,
                    },
                    explain=explanation,
                    explain_code=explain_code,
                    state_notes=state_notes,
                )
            )

            self._log_runner_event(
                trace_id=trace_id,
                race_id=request.race_id,
                runner_id=runner.runner_id,
                baseline_score=baseline_score,
                combined_score=combined_score,
                veto=veto,
                intensity=intensity,
                state=state,
            )

            self.metrics.increment_doctrine("TOTAL", combined_score >= self.orchestrator.config.s2_to_s3)
            for doctrine_id, score in doctrine_scores.items():
                self.metrics.increment_doctrine(doctrine_id, score > 0.0)
            if state == ConvictionState.STRIKE:
                self.metrics.s3_decisions += 1

        elapsed_ms = int((time.perf_counter_ns() - start_time) / 1_000_000)
        self.metrics.decisions_total += len(convictions)

        lineage_payload = asdict(self.lineage)
        artifact_hash = hashlib.sha1(json.dumps(lineage_payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
        record = {
            "event": "decision",
            "trace_id": trace_id,
            "race_id": request.race_id,
            "schema_version": DEFAULT_SCHEMA_VERSION,
            "convictions": [conviction.model_dump() for conviction in convictions],
            "elapsed_ms": elapsed_ms,
            "lineage": lineage_payload,
            "artifacts": {"lineage_hash": artifact_hash},
            "risk_throttle": aggregate_throttle,
        }
        self.logger.write(record)

        return DecisionResponse(
            schema_version=DEFAULT_SCHEMA_VERSION,
            trace_id=trace_id,
            run=RunLineage(**lineage_payload),
            convictions=convictions,
            risk_throttle=aggregate_throttle,
            artifacts={"lineage_hash": artifact_hash},
        )

    def _encode_features(self, features: Mapping[str, float], ablate: Iterable[str]) -> np.ndarray:
        if "vector" in ablate:
            return self._zero_vector.copy()
        return self.vector_core.encode(features)

    def _compute_baseline_metrics(self, state_vector: np.ndarray) -> Tuple[float, float, float]:
        """Derive baseline conviction metrics from the fused state vector."""

        norm = float(np.linalg.norm(state_vector))
        scale = np.sqrt(state_vector.size) + 1e-6
        intensity = float(norm / scale)
        baseline_score = float(np.clip(np.tanh(intensity / 3.0), 0.0, 1.0))
        baseline_prob = float(np.clip(1.0 / (1.0 + np.exp(-intensity)), 0.01, 0.99))
        return intensity, baseline_score, baseline_prob

    def _evaluate_doctrines(
        self,
        runner_features: Mapping[str, float],
        baseline_prob: float,
        implied_prob: float,
        ablate_neosym: bool,
    ) -> Tuple[Dict[str, float], bool]:
        """Run doctrine evaluations and derive a veto signal."""

        if ablate_neosym:
            return {}, False

        context: Dict[str, float] = {
            "model_prob": baseline_prob,
            "sp_implied_prob": implied_prob,
        }
        evaluations = self.doctrine_executor.evaluate(runner_features, context)
        doctrine_scores = {
            doctrine_id: evaluation.score for doctrine_id, evaluation in evaluations.items()
        }
        veto = any(
            rule.rule.action.upper() == "STAND_DOWN" and rule.satisfied
            for evaluation in evaluations.values()
            for rule in evaluation.rules
        )
        return doctrine_scores, veto

    def _combine_scores(
        self,
        baseline_score: float,
        doctrine_scores: Mapping[str, float],
        ablate: Iterable[str],
    ) -> float:
        """Blend baseline and doctrine signals into a conviction score."""

        if "neosym" in ablate:
            doctrine_total = 0.0
        else:
            doctrine_total = float(min(1.0, sum(doctrine_scores.values())))
        blended = 0.6 * baseline_score + 0.4 * doctrine_total
        return float(np.clip(blended, 0.0, 1.0))

    def _update_strike_ready(self, runner_id: str, conviction: float) -> bool:
        threshold = self.orchestrator.config.s2_to_s3
        sustain = getattr(self.orchestrator.config, "strike_sustain_ticks", 2)
        counter = self._strike_ready.get(runner_id, 0)
        if conviction >= threshold:
            counter += 1
        else:
            counter = 0
        self._strike_ready[runner_id] = counter
        return counter >= sustain

    def _assess_world(
        self,
        runner_id: str,
        runner_ids: Sequence[str],
        start_price: float,
        rollouts: int,
        seed: Optional[int],
        ablate_world: bool,
    ) -> WorldTrace:
        if ablate_world or rollouts == 0:
            return WorldTrace(
                robustness=None,
                rollouts=0,
                budget_ms=0,
                status="ablated",
                scenarios=[],
                manipulation_score=0.0,
                spoof_flag=False,
            )

        start = time.perf_counter_ns()
        try:
            stats = self.world_planner.assess(
                runners=runner_ids,
                favourite=runner_id,
                start_price=start_price,
                trials=rollouts,
                seed=seed,
                timeout_ms=200,
            )
            elapsed = int((time.perf_counter_ns() - start) / 1_000_000)
        except TimeoutError:
            self.metrics.reversal_events += 1
            return WorldTrace(
                robustness=None,
                rollouts=0,
                budget_ms=200,
                status="timeout",
                scenarios=[],
                manipulation_score=0.0,
                spoof_flag=False,
            )

        elapsed = int((time.perf_counter_ns() - start) / 1_000_000)
        self.metrics.rollout_runtime_ms.append(elapsed)
        return WorldTrace(
            robustness=float(stats["win_rate"]),
            rollouts=stats.get("trials", rollouts),
            budget_ms=min(elapsed, 200),
            status="ok",
            prices=list(stats.get("prices", [])),
            volumes=list(stats.get("volumes", [])),
            scenarios=list(stats.get("scenarios", [])),
            manipulation_score=float(stats.get("manipulation_score", 0.0)),
            spoof_flag=bool(stats.get("spoof_flag", False)),
        )

    def _build_explanation(
        self,
        state: ConvictionState,
        baseline_score: float,
        doctrine_scores: Mapping[str, float],
        world: WorldTrace,
    ) -> Tuple[str, str]:
        doctrine_fragments = ", ".join(
            f"{key}:{score:.2f}" for key, score in sorted(doctrine_scores.items())
        ) or "none"
        robustness = "n/a" if world.robustness is None else f"{world.robustness:.2f}"
        explain_code = f"{state.value}_{'VETO' if doctrine_scores and max(doctrine_scores.values()) == 0 else 'PASS'}"
        message = (
            f"State {state.value} | baseline {baseline_score:.2f} | "
            f"Doctrines [{doctrine_fragments}] | Robustness {robustness}"
        )
        return message, explain_code

    def _log_runner_event(
        self,
        trace_id: str,
        race_id: str,
        runner_id: str,
        baseline_score: float,
        combined_score: float,
        veto: bool,
        intensity: float,
        state: ConvictionState,
    ) -> None:
        payload: Dict[str, object] = {
            "trace_id": trace_id,
            "race_id": race_id,
            "runner_id": runner_id,
            "baseline_score": round(baseline_score, 4),
            "combined_score": round(combined_score, 4),
            "veto": veto,
            "intensity": round(intensity, 4),
            "state": state.value,
        }
        _log_event("runner_decision", payload)

    def _apply_confidence_decay(
        self,
        *,
        conviction: float,
        baseline_score: float,
        previous: Optional[RunnerDecisionState],
    ) -> Tuple[float, int, bool]:
        """Decay conviction when the supporting evidence goes stale."""

        reconfirm_tick = self._tick_counter
        triggered = False
        adjusted = conviction

        if previous is None:
            return adjusted, reconfirm_tick, triggered

        reconfirm_tick = previous.last_reconfirmed_tick
        baseline_improved = baseline_score >= previous.baseline_score + 0.02
        conviction_improved = conviction >= previous.last_conviction + 0.05

        if baseline_improved or conviction_improved:
            reconfirm_tick = self._tick_counter
        else:
            ticks_since = max(0, self._tick_counter - previous.last_reconfirmed_tick)
            if ticks_since > 0:
                decay_factor = float(np.exp(-0.08 * ticks_since))
                floor = max(0.0, baseline_score * 0.7)
                adjusted = max(conviction * decay_factor, floor)
                triggered = adjusted < conviction - 1e-6

        return adjusted, reconfirm_tick, triggered

    def _apply_ego_kill_switch(
        self,
        *,
        conviction: float,
        baseline_score: float,
        doctrine_scores: Mapping[str, float],
        world_trace: WorldTrace,
        previous: Optional[RunnerDecisionState],
    ) -> Tuple[float, bool]:
        """Clamp conviction if it accelerates faster than the evidence."""

        previous_conviction = previous.last_conviction if previous else 0.0
        doctrine_mean = 0.0
        if doctrine_scores:
            doctrine_mean = float(sum(doctrine_scores.values())) / max(len(doctrine_scores), 1)
        evidence = 0.5 * baseline_score + 0.5 * doctrine_mean
        if world_trace.robustness is not None:
            evidence = (evidence + world_trace.robustness) / 2.0

        triggered = False
        if conviction - evidence >= 0.2 and conviction - previous_conviction >= 0.1:
            conviction = float(min(conviction, previous_conviction + 0.05))
            triggered = True
        return conviction, triggered

    def _apply_bias_quarantine(
        self,
        *,
        conviction: float,
        baseline_score: float,
        world_trace: WorldTrace,
        previous: Optional[RunnerDecisionState],
    ) -> Tuple[float, bool]:
        """Reduce conviction during hot streaks that lack robustness."""

        if previous is None:
            return conviction, False

        heat = previous.bias_heat
        if heat < 2.0:
            return conviction, False

        robustness = world_trace.robustness or 0.0
        required = min(0.9, 0.55 + 0.05 * max(0.0, heat - 1.5))
        if robustness < required:
            conviction = float(min(conviction, baseline_score))
            return conviction, True
        return conviction, False

    def _apply_memory_entropy_guard(
        self,
        *,
        conviction: float,
        baseline_score: float,
    ) -> Tuple[float, bool, float]:
        """Clamp conviction when memory entropy collapses."""

        entropy = self.memory.entropy()
        if entropy >= 0.45:
            return conviction, False, entropy

        softened = max(baseline_score * (0.6 + 0.4 * entropy), conviction * (0.75 + 0.25 * entropy))
        conviction = float(min(conviction, softened))
        return conviction, True, entropy

    def _apply_market_manipulation_guard(
        self,
        *,
        throttle: str,
        world_trace: WorldTrace,
        liquidity: float,
    ) -> Tuple[str, List[str]]:
        """Adjust throttle when spoofing or dark pools are suspected."""

        notes: List[str] = []
        if world_trace.spoof_flag:
            throttle = self._resolve_throttle(throttle, "OFF")
            notes.append("market_spoof")
        elif world_trace.manipulation_score >= 0.4:
            throttle = self._resolve_throttle(throttle, "REDUCE")
            notes.append("market_manipulation")

        high_liquidity = liquidity >= self.risk_rails.liquidity_floor * 4
        max_volume = max(world_trace.volumes) if world_trace.volumes else 0.0
        if high_liquidity and max_volume < 1.5:
            throttle = self._resolve_throttle(throttle, "REDUCE")
            notes.append("dark_pool")

        return throttle, notes

    def _scenario_inversion_guard(
        self,
        *,
        baseline_score: float,
        conviction: float,
        doctrine_scores: Mapping[str, float],
    ) -> bool:
        """Detect inversions where strong baselines collapse after fusion."""

        doctrine_total = float(sum(doctrine_scores.values()))
        if baseline_score > 0.75 and conviction < baseline_score - 0.25 and doctrine_total > 0.2:
            return True
        return False

    def _update_bias_heat(
        self,
        *,
        previous_state: Optional[RunnerDecisionState],
        conviction: float,
        baseline_score: float,
    ) -> float:
        if previous_state is None:
            return 0.0

        delta_conviction = conviction - previous_state.last_conviction
        delta_baseline = baseline_score - previous_state.baseline_score
        heat = previous_state.bias_heat

        if delta_conviction > 0.1 and delta_conviction > delta_baseline + 0.05:
            heat = min(5.0, heat + 1.0)
        else:
            heat = max(0.0, heat - 0.5)
        return heat

    @staticmethod
    def _resolve_throttle(current: str, candidate: str) -> str:
        order = {"OFF": 0, "REDUCE": 1, "ON": 2}
        if candidate not in order or current not in order:
            return current
        return current if order[current] <= order[candidate] else candidate

    def _simulate_transition(
        self,
        *,
        previous_state: ConvictionState,
        conviction: float,
        doctrine_veto: bool,
        allow_strike: bool,
    ) -> ConvictionState:
        config = self.orchestrator.config

        if doctrine_veto:
            return ConvictionState.OFF_RISK

        if previous_state == ConvictionState.OFF_RISK:
            if conviction < config.s0_to_s1:
                return ConvictionState.OFF_RISK
            return ConvictionState.SCAN

        if previous_state == ConvictionState.SCAN:
            if conviction >= config.s0_to_s1:
                return ConvictionState.LATENT
            return ConvictionState.SCAN

        if previous_state == ConvictionState.LATENT:
            if conviction >= config.s1_to_s2:
                return ConvictionState.EMERGING
            if conviction < config.s0_to_s1:
                return ConvictionState.SCAN
            return ConvictionState.LATENT

        if previous_state == ConvictionState.EMERGING:
            if conviction >= config.s2_to_s3 and allow_strike:
                return ConvictionState.STRIKE
            if conviction < config.s0_to_s1:
                return ConvictionState.SCAN
            return ConvictionState.EMERGING

        if previous_state == ConvictionState.STRIKE:
            if conviction < config.strike_hysteresis:
                return ConvictionState.EMERGING
            return ConvictionState.STRIKE

        return ConvictionState.SCAN

    def _build_trace_id(self) -> str:
        timestamp = time.strftime(TRACE_TIMESTAMP_FORMAT, time.gmtime())
        suffix = uuid.uuid4().hex[:4]
        return f"vx-{timestamp}-{suffix}"

    @property
    def _strike_ready(self) -> MutableMapping[str, int]:
        return self._strike_ready_cache

    def _build_governance_note(
        self,
        *,
        trace_id: str,
        runner_id: str,
        previous_state: ConvictionState,
        new_state: ConvictionState,
        doctrine_scores: Mapping[str, float],
        throttle: str,
        world_trace: WorldTrace,
    ) -> GovernanceNote:
        alignment = {
            key: round(score, 3)
            for key, score in doctrine_scores.items()
            if score >= 0.5
        }
        notes = [f"world={world_trace.status}", f"rollouts={world_trace.rollouts}"]
        if world_trace.status == "timeout":
            notes.append("world_timeout")
        return GovernanceNote(
            audit_ref=f"{trace_id}:{runner_id}",
            state_transition={"from": previous_state.value, "to": new_state.value},
            doctrine_alignment=alignment,
            risk_throttle=throttle,
            world_status=world_trace.status,
            notes=notes,
        )

    def ingest_outcome(self, payload: OutcomeIngestion) -> OutcomeResponse:
        if not payload.results:
            raise HTTPException(status_code=400, detail="results payload cannot be empty")

        processed: List[PostMortemRunner] = []
        for result in payload.results:
            outcome_value = 1.0 if result.win else 0.0
            try:
                event = self.memory.update_outcome(
                    payload.race_id,
                    result.runner_id,
                    outcome=outcome_value,
                    finish_position=result.finish_position,
                    payout=result.payout,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

            self._apply_outcome_feedback(event=event, outcome_value=outcome_value)

        insights = self.post_mortem.build(self.memory.events_for_race(payload.race_id))
        for runner_id, insight in insights.items():
            processed.append(
                PostMortemRunner(
                    runner_id=runner_id,
                    conviction=insight.conviction,
                    model_prob=insight.model_prob,
                    outcome=insight.outcome,
                    finish_position=insight.finish_position,
                    doctrine_alignment=insight.doctrine_alignment,
                    world_robustness=insight.world_robustness,
                    risk_throttle=insight.risk_throttle,
                    explain=insight.explain,
                    explain_code=insight.explain_code,
                    adjustments=insight.adjustments,
                )
            )

        log_record = {
            "event": "outcome_ingested",
            "race_id": payload.race_id,
            "schema_version": payload.schema_version,
            "timestamp": payload.timestamp,
            "processed": [runner.model_dump() for runner in processed],
        }
        self.logger.write(log_record)

        return OutcomeResponse(
            schema_version=payload.schema_version,
            race_id=payload.race_id,
            processed=processed,
        )

    def _apply_outcome_feedback(self, *, event: MemoryEvent, outcome_value: float) -> None:
        self.risk_rails.register_outcome(
            reward=outcome_value,
            conviction=event.conviction,
            model_prob=event.model_prob,
        )
        self.trainer.reinforce(event.conviction, outcome_value)

    @staticmethod
    def _implied_probability(price: float) -> float:
        if price <= 0:
            return 0.0
        return float(np.clip(1.0 / price, 0.0, 1.0))


def _compute_feature_schema_hash(core: VectorFusionCore) -> str:
    digest = hashlib.sha1()
    for encoder in core.encoders:  # type: ignore[attr-defined]
        digest.update(encoder.name.encode("utf-8"))
        digest.update(str(getattr(encoder, "dimension", 0)).encode("utf-8"))
    return digest.hexdigest()[:7]


def _compute_doctrine_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:7]


_repo_root = Path(__file__).resolve().parents[2]
_doctrine_path = _repo_root / "mesh" / "neosym" / "doctrine_lang.yaml"
_vector_core = build_default_vector_fusion_core()
_stepper = IdeaStepper(mixer=StateMixer())
_doctrine_executor = DoctrineExecutor(load_default_doctrines(_repo_root))
_risk_budget = RiskBudget(max_positions=3)
_orchestrator = StateOrchestrator(
    config=OrchestratorConfig(),
    risk_budget=_risk_budget,
)
_world_planner = WorldPlanner(
    race_shape=RaceShapeSimulator(),
    market_flow=MarketFlowSimulator(),
)
_memory = EpisodicMemory()
_online_trainer = OnlineTrainer(cell=LiquidCell())
_post_mortem = PostMortemEngine()
_lineage = EngineLineage(
    model_sha=os.getenv("VELO_MODEL_SHA", "unknown"),
    feature_schema_hash=_compute_feature_schema_hash(_vector_core),
    doctrine_pack_version=_compute_doctrine_version(_doctrine_path),
    vector_fusion_version="0.1.0",
)
_engine = DecisionEngine(
    vector_core=_vector_core,
    stepper=_stepper,
    doctrine_executor=_doctrine_executor,
    orchestrator=_orchestrator,
    world_planner=_world_planner,
    memory=_memory,
    trainer=_online_trainer,
    risk_budget=_risk_budget,
    lineage=_lineage,
    logger=DecisionLogger(LOG_DIR, PARQUET_DIR),
    risk_rails=RiskRails(),
    metrics=DecisionMetrics(),
    post_mortem=_post_mortem,
)


def get_engine() -> DecisionEngine:
    """Return the singleton decision engine instance."""

    return _engine


@app.post("/decide", response_model=DecisionResponse)
async def decide(request: DecisionRequest) -> DecisionResponse:
    """Return conviction payloads for each runner in the request."""

    try:
        return _engine.decide(request)
    except ValidationError as exc:  # pragma: no cover - FastAPI handles normally
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc


@app.post("/ingest/outcome", response_model=OutcomeResponse)
async def ingest_outcome(payload: OutcomeIngestion) -> OutcomeResponse:
    """Ingest official outcomes and trigger post-mortem analysis."""

    return _engine.ingest_outcome(payload)


def _log_event(message: str, payload: Mapping[str, object]) -> None:
    LOGGER.info(json.dumps({"message": message, **payload}))
