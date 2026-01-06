"""Contract and behaviour tests for the /decide endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from mesh.api.server import RunnerDecisionState, WorldTrace, app, get_engine
from mesh.memory import MemoryEvent
from mesh.orchestrator import ConvictionState


client = TestClient(app)


def _base_runner_features() -> Dict[str, float]:
    return {
        "form_momentum": 0.65,
        "form_ewma": 1.05,
        "form_recency": 6,
        "rpr_latest": 108,
        "rpr_peak": 112,
        "ts_latest": 94,
        "pace_pressure": 0.32,
        "draw_advantage": 1.0,
        "official_rating_drop": 4.0,
        "weight_carried": 8.9,
        "early_speed_rank": 1.0,
    }


def _build_request(*, ablate: List[str] | None = None) -> Dict[str, object]:
    prices = {"alpha": 5.0, "beta": 6.5}
    return {
        "schema_version": "1.0.0",
        "race_id": "audit_race",
        "runners": [
            {"runner_id": "alpha", "features": _base_runner_features()},
            {"runner_id": "beta", "features": {**_base_runner_features(), "form_momentum": -0.2}},
        ],
        "market": {
            "prices": prices,
            "liquidity": 125_000.0,
            "timestamp": datetime.utcnow().isoformat(),
        },
        "options": {
            "rollouts": 64,
            "seed": 1337,
            "ablate": ablate or [],
        },
    }


def test_decide_returns_trace_with_lineage() -> None:
    response = client.post("/decide", json=_build_request())
    assert response.status_code == 200
    body = response.json()

    assert body["schema_version"] == "1.0.0"
    assert body["trace_id"].startswith("vx-")
    assert "artifacts" in body
    assert "lineage_hash" in body["artifacts"]
    assert set(body["run"].keys()) == {
        "model_sha",
        "feature_schema_hash",
        "doctrine_pack_version",
        "vector_fusion_version",
    }

    convictions = body["convictions"]
    assert len(convictions) == 2
    for conviction in convictions:
        assert 0.0 <= conviction["conviction"] <= 1.0
        assert conviction["state"] in {state.value for state in ConvictionState}
        assert "world" in conviction
        assert "risk" in conviction
        assert "explain" in conviction
        assert "explain_code" in conviction
        assert "counterfactual" in conviction
        assert "scenarios" in conviction["world"]
        assert "manipulation_score" in conviction["world"]
        assert "spoof_flag" in conviction["world"]
        counterfactual = conviction["counterfactual"]
        assert set(counterfactual.keys()) == {"state", "conviction", "s3_positions", "throttle"}
    assert body["risk_throttle"] in {"ON", "OFF", "REDUCE"}


def test_missing_market_prices_rejected() -> None:
    request = _build_request()
    request["market"]["prices"].pop("alpha")
    response = client.post("/decide", json=request)
    assert response.status_code == 422


def test_state_hysteresis_prevents_flip_flop() -> None:
    request = _build_request()
    states: List[str] = []
    for _ in range(5):
        result = client.post("/decide", json=request)
        assert result.status_code == 200
        states.append(result.json()["convictions"][0]["state"])

    assert states[0] != "S3"
    assert "S3" in states
    # once S3 achieved, it should not immediately drop back to S2 on identical input
    post_strike_states = states[states.index("S3") :]
    assert all(state in {"S3", "S2"} for state in post_strike_states)
    assert post_strike_states[0] == "S3"


def test_doctrine_scores_fire_for_qualifying_runner() -> None:
    request = _build_request()
    response = client.post("/decide", json=request)
    assert response.status_code == 200
    conviction = response.json()["convictions"][0]
    doctrine_scores = conviction["doctrine_scores"]
    assert doctrine_scores["doctrine_A_form_cycle"] > 0.0
    assert doctrine_scores["doctrine_B_pace_shape"] > 0.0
    assert conviction["doctrine_veto"] is False


def test_ablation_flags_disable_modules_gracefully() -> None:
    request = _build_request(ablate=["vector", "neosym", "world"])
    response = client.post("/decide", json=request)
    assert response.status_code == 200
    conviction = response.json()["convictions"][0]
    assert conviction["world"]["status"] == "ablated"
    assert conviction["conviction"] == 0.0


def test_governance_mode_includes_notes() -> None:
    request = _build_request()
    request["options"]["governance"] = True
    response = client.post("/decide", json=request)
    assert response.status_code == 200
    conviction = response.json()["convictions"][0]
    assert "governance" in conviction
    governance = conviction["governance"]
    assert governance["audit_ref"].startswith("vx-")
    assert governance["state_transition"]["from"] in {state.value for state in ConvictionState}
    assert governance["state_transition"]["to"] in {state.value for state in ConvictionState}


def test_outcome_ingestion_returns_post_mortem() -> None:
    decide_response = client.post("/decide", json=_build_request())
    assert decide_response.status_code == 200
    outcome_payload = {
        "schema_version": "1.0.0",
        "race_id": "audit_race",
        "timestamp": datetime.utcnow().isoformat(),
        "results": [
            {"runner_id": "alpha", "win": True, "finish_position": 1},
            {"runner_id": "beta", "win": False, "finish_position": 5},
        ],
    }
    outcome_response = client.post("/ingest/outcome", json=outcome_payload)
    assert outcome_response.status_code == 200
    body = outcome_response.json()
    assert body["race_id"] == "audit_race"
    assert len(body["processed"]) == 2
    runner_record = next(item for item in body["processed"] if item["runner_id"] == "alpha")
    assert runner_record["outcome"] == 1.0
    assert "adjustments" in runner_record


def test_world_timeout_path(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = get_engine()

    def _raise_timeout(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise TimeoutError("test timeout")

    monkeypatch.setattr(type(engine.world_planner), "assess", _raise_timeout)
    response = client.post("/decide", json=_build_request())
    assert response.status_code == 200
    conviction = response.json()["convictions"][0]
    assert conviction["world"]["status"] == "timeout"
    assert conviction["world"]["robustness"] is None


def test_confidence_decay_reduces_without_reconfirmation() -> None:
    engine = get_engine()
    engine.reset()
    zero_vector = engine.vector_core.encode({}) * 0.0
    previous = RunnerDecisionState(
        state_vector=zero_vector,
        state=ConvictionState.EMERGING,
        strike_ready_ticks=0,
        last_conviction=0.9,
        last_reconfirmed_tick=0,
        bias_heat=0.0,
        baseline_score=0.82,
    )
    engine._tick_counter = 5  # type: ignore[attr-defined]
    decayed, reconfirm_tick, triggered = engine._apply_confidence_decay(  # type: ignore[attr-defined]
        conviction=0.9,
        baseline_score=0.82,
        previous=previous,
    )
    assert triggered is True
    assert decayed < 0.9
    assert reconfirm_tick == 0


def test_bias_quarantine_reduces_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = get_engine()
    engine.reset()

    def noop_decay(self, conviction, baseline_score, previous):  # type: ignore[no-untyped-def]
        return conviction, self._tick_counter, False

    def noop_ego(self, conviction, baseline_score, doctrine_scores, world_trace, previous):  # type: ignore[no-untyped-def]
        return conviction, False

    def forced_bias(self, conviction, baseline_score, world_trace, previous):  # type: ignore[no-untyped-def]
        return baseline_score, True

    monkeypatch.setattr(type(engine), "_apply_confidence_decay", noop_decay)
    monkeypatch.setattr(type(engine), "_apply_ego_kill_switch", noop_ego)
    monkeypatch.setattr(type(engine), "_apply_bias_quarantine", forced_bias)

    response = client.post("/decide", json=_build_request())
    assert response.status_code == 200
    body = response.json()
    conviction = body["convictions"][0]
    assert conviction["risk"]["throttle"] == "REDUCE"
    assert "bias_quarantine" in conviction["risk"]["notes"]
    assert body["risk_throttle"] == "REDUCE"


def test_ego_kill_switch_clamps_when_evidence_lags() -> None:
    engine = get_engine()
    previous = RunnerDecisionState(
        state_vector=engine.vector_core.encode({}) * 0.0,
        state=ConvictionState.EMERGING,
        strike_ready_ticks=0,
        last_conviction=0.6,
        last_reconfirmed_tick=0,
        bias_heat=0.0,
        baseline_score=0.55,
    )
    world_trace = WorldTrace(
        robustness=0.4,
        rollouts=32,
        budget_ms=50,
        status="ok",
        scenarios=[],
        manipulation_score=0.0,
        spoof_flag=False,
    )
    clamped, triggered = engine._apply_ego_kill_switch(  # type: ignore[attr-defined]
        conviction=0.85,
        baseline_score=0.6,
        doctrine_scores={"A": 0.3, "B": 0.25},
        world_trace=world_trace,
        previous=previous,
    )
    assert triggered is True
    assert clamped == pytest.approx(previous.last_conviction + 0.05, abs=1e-6)


def test_scenario_inversion_forces_off(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = get_engine()
    engine.reset()

    def noop_decay(self, conviction, baseline_score, previous):  # type: ignore[no-untyped-def]
        return conviction, self._tick_counter, False

    def noop_ego(self, conviction, baseline_score, doctrine_scores, world_trace, previous):  # type: ignore[no-untyped-def]
        return conviction, False

    def noop_bias(self, conviction, baseline_score, world_trace, previous):  # type: ignore[no-untyped-def]
        return conviction, False

    def forced_inversion(self, baseline_score, conviction, doctrine_scores):  # type: ignore[no-untyped-def]
        return True

    monkeypatch.setattr(type(engine), "_apply_confidence_decay", noop_decay)
    monkeypatch.setattr(type(engine), "_apply_ego_kill_switch", noop_ego)
    monkeypatch.setattr(type(engine), "_apply_bias_quarantine", noop_bias)
    monkeypatch.setattr(type(engine), "_scenario_inversion_guard", forced_inversion)

    response = client.post("/decide", json=_build_request())
    assert response.status_code == 200
    body = response.json()
    conviction = body["convictions"][0]
    assert conviction["risk"]["throttle"] == "OFF"
    assert "scenario_inversion" in conviction["risk"]["notes"]
    assert body["risk_throttle"] == "OFF"


def test_memory_entropy_guard_reduces_throttle() -> None:
    engine = get_engine()
    engine.reset()
    for idx in range(12):
        engine.memory.record(
            MemoryEvent(
                trace_id=f"seed-{idx}",
                race_id="seed_race",
                runner_id=f"seed_runner_{idx}",
                timestamp="2025-01-01T00:00:00Z",
                state="S2",
                previous_state="S1",
                conviction=0.8,
                model_prob=0.65,
                doctrine_scores={"A": 0.4},
                doctrine_veto=False,
                world_robustness=0.5,
                risk_throttle="ON",
                market_snapshot={"price": 5.0, "liquidity": 125_000.0},
                explain="seed",
                explain_code="S2_PASS",
                state_notes=[],
            )
        )

    response = client.post("/decide", json=_build_request())
    assert response.status_code == 200
    conviction = response.json()["convictions"][0]
    assert conviction["risk"]["throttle"] in {"REDUCE", "OFF"}
    assert "memory_entropy_low" in conviction["risk"]["notes"]


def test_market_spoof_triggers_off(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = get_engine()
    engine.reset()

    def flagged_assess(self, **kwargs):  # type: ignore[no-untyped-def]
        return {
            "win_rate": 0.62,
            "prices": [5.0, 4.1, 5.3],
            "volumes": [1.0, 3.5, 0.4],
            "trials": kwargs.get("trials", 64),
            "scenarios": [
                {"pace_shift": 0.0, "win_rate": 0.62, "trials": kwargs.get("trials", 64)}
            ],
            "manipulation_score": 0.7,
            "spoof_flag": True,
        }

    monkeypatch.setattr(type(engine.world_planner), "assess", flagged_assess)

    response = client.post("/decide", json=_build_request())
    assert response.status_code == 200
    body = response.json()
    conviction = body["convictions"][0]
    assert conviction["risk"]["throttle"] == "OFF"
    assert "market_spoof" in conviction["risk"]["notes"]
    assert body["risk_throttle"] == "OFF"

