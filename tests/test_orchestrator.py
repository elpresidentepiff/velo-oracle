from mesh.orchestrator import ConvictionState, OrchestratorConfig, RiskBudget, StateOrchestrator


def test_orchestrator_progression_to_strike() -> None:
    orchestrator = StateOrchestrator(OrchestratorConfig(), RiskBudget(max_positions=2))

    state = ConvictionState.SCAN
    state = orchestrator.transition(state, conviction=0.8)
    assert state == ConvictionState.LATENT
    state = orchestrator.transition(state, conviction=0.87)
    assert state == ConvictionState.EMERGING
    # need to allow strike explicitly after sustain ticks
    state = orchestrator.transition(state, conviction=0.95, allow_strike=False)
    assert state == ConvictionState.EMERGING
    state = orchestrator.transition(state, conviction=0.95, allow_strike=True)
    assert state == ConvictionState.STRIKE
    state = orchestrator.transition(state, conviction=0.6)
    assert state == ConvictionState.EMERGING
