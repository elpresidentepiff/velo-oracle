"""State orchestrator policies."""

from __future__ import annotations

from dataclasses import dataclass

from .states import ConvictionState, OrchestratorConfig


@dataclass(slots=True)
class RiskBudget:
    """Tracks the number of concurrent strike positions allowed."""

    max_positions: int
    active_positions: int = 0

    def can_add_position(self) -> bool:
        return self.active_positions < self.max_positions

    def enter(self) -> None:
        if not self.can_add_position():
            raise RuntimeError("risk budget exceeded")
        self.active_positions += 1

    def exit(self) -> None:
        if self.active_positions > 0:
            self.active_positions -= 1

    def set_active(self, count: int) -> None:
        """Synchronise the active position counter with external state."""

        if count < 0 or count > self.max_positions:
            raise ValueError("active positions must be within budget bounds")
        self.active_positions = count


@dataclass(slots=True)
class StateOrchestrator:
    """Finite state machine implementing conviction progression."""

    config: OrchestratorConfig
    risk_budget: RiskBudget

    def transition(
        self,
        current_state: ConvictionState,
        conviction: float,
        doctrine_veto: bool = False,
        allow_strike: bool = True,
    ) -> ConvictionState:
        """Return the next conviction state."""

        if doctrine_veto:
            if current_state == ConvictionState.STRIKE:
                self.risk_budget.exit()
            return ConvictionState.OFF_RISK

        if current_state == ConvictionState.OFF_RISK and conviction < self.config.s0_to_s1:
            return ConvictionState.OFF_RISK

        if current_state == ConvictionState.SCAN:
            if conviction >= self.config.s0_to_s1:
                return ConvictionState.LATENT
            return ConvictionState.SCAN

        if current_state == ConvictionState.LATENT:
            if conviction >= self.config.s1_to_s2:
                return ConvictionState.EMERGING
            if conviction < self.config.s0_to_s1:
                return ConvictionState.SCAN
            return ConvictionState.LATENT

        if current_state == ConvictionState.EMERGING:
            if (
                conviction >= self.config.s2_to_s3
                and allow_strike
                and self.risk_budget.can_add_position()
            ):
                self.risk_budget.enter()
                return ConvictionState.STRIKE
            if conviction < self.config.s0_to_s1:
                return ConvictionState.SCAN
            return ConvictionState.EMERGING

        if current_state == ConvictionState.STRIKE:
            if conviction < self.config.strike_hysteresis:
                self.risk_budget.exit()
                return ConvictionState.EMERGING
            return ConvictionState.STRIKE

        return ConvictionState.SCAN
