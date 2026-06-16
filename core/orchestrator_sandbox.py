"""
core/orchestrator.py

Coordinate-search Orchestrator for the peer-review OMAS platform.

The platform separates two weight vectors:

    w^SW   fixed social-planner valuation weights used only to compute SW
    w^ECU  variable incentive weights used only to pay agents ECUs

Implemented formulas
--------------------
    ecu_i^(t) = Σ_q w_q^ECU · (1/(n-1))Σ_{j≠i}s_ji^(t)(q)

    SW^(t) = Σ_q w_q^SW · (1/n)Σ_i (1/(n-1))Σ_{j≠i}s_ji^(t)(q)

The Orchestrator controls w^ECU, not w^SW. On update rounds, it performs
finite-difference coordinate search over w^ECU. For every quality dimension q,
it constructs two candidate vectors, w_q+ and w_q-, obtained by perturbing q by
±epsilon and renormalising the vector to preserve the ECU budget. Each candidate
is evaluated by a sandbox rerun supplied by the protocol. Sandbox reruns use the
same pre-round state and visibility rules as the real round, but their outputs are
not added to the official dialogue history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from core.ecu import EcuLedger

Direction = Literal["+", "-"]


@dataclass
class CandidateEvaluation:
    """One counterfactual coordinate perturbation evaluation."""
    cycle: int
    dimension: str
    direction: Direction
    candidate_weights: dict[str, float]
    social_welfare: float

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "dimension": self.dimension,
            "direction": self.direction,
            "candidate_weights": {k: round(v, 6) for k, v in self.candidate_weights.items()},
            "social_welfare": round(self.social_welfare, 6),
        }


@dataclass
class OrchestratorUpdate:
    """Record of one coordinate-search decision after a real round."""
    cycle: int
    observed_sw: float
    old_weights: dict[str, float]
    new_weights: dict[str, float]
    accepted: bool
    best_dimension: str | None = None
    best_direction: Direction | None = None
    best_candidate_sw: float | None = None
    candidates: list[CandidateEvaluation] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        if self.best_candidate_sw is None:
            return 0.0
        return self.best_candidate_sw - self.observed_sw

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "observed_sw": round(self.observed_sw, 6),
            "old_weights": {k: round(v, 6) for k, v in self.old_weights.items()},
            "new_weights": {k: round(v, 6) for k, v in self.new_weights.items()},
            "accepted": self.accepted,
            "best_dimension": self.best_dimension,
            "best_direction": self.best_direction,
            "best_candidate_sw": None if self.best_candidate_sw is None else round(self.best_candidate_sw, 6),
            "improvement": round(self.improvement, 6),
            "candidates": [c.to_dict() for c in self.candidates],
        }


class Orchestrator:
    """
    Updates the variable ECU incentive vector w^ECU using coordinate search.

    The fixed SW valuation vector w^SW lives in EcuLedger.sw_weights and is never
    changed by this class.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        update_every: int = 2,
        min_weight: float = 0.1,
        max_weight: float = 3.0,
        enabled: bool = True,
        min_improvement: float = 1e-9,
    ):
        self.step_size = float(step_size)
        self.update_every = max(1, int(update_every))
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.enabled = enabled
        self.min_improvement = float(min_improvement)
        self._history: list[OrchestratorUpdate] = []
        self.local_optimum_reached: bool = False

    def should_update(self, cycle: int, *, is_final_cycle: bool = False) -> bool:
        """Return True when the Orchestrator should run after this real cycle."""
        if not self.enabled or self.local_optimum_reached or is_final_cycle:
            return False
        return (cycle + 1) % self.update_every == 0

    def coordinate_search(
        self,
        cycle: int,
        ledger: EcuLedger,
        observed_sw: float,
        evaluator: Callable[[dict[str, float]], float],
    ) -> OrchestratorUpdate | None:
        """
        Run one coordinate-search update.

        Parameters
        ----------
        cycle : int
            The completed real cycle whose pre-round state will be used by the
            protocol-level sandbox evaluator.
        ledger : EcuLedger
            Official ledger. Its ECU weights are updated only if a candidate
            improves observed SW.
        observed_sw : float
            Social welfare from the completed real cycle.
        evaluator : Callable[[dict[str, float]], float]
            Protocol-supplied sandbox evaluator. It must rerun the same cycle
            from the same pre-round state under candidate ECU weights, compute
            candidate SW, and discard all sandbox outputs.
        """
        if not self.enabled or self.local_optimum_reached:
            return None

        old_weights = dict(ledger.ecu_weights)
        best_weights = dict(old_weights)
        best_sw = float(observed_sw)
        best_dim: str | None = None
        best_direction: Direction | None = None
        candidates: list[CandidateEvaluation] = []

        for dim in old_weights:
            for direction, signed_step in (("+", self.step_size), ("-", -self.step_size)):
                candidate_weights = self._normalise(
                    old_weights,
                    target_dim=dim,
                    new_val=old_weights[dim] + signed_step,
                    target_sum=sum(old_weights.values()),
                )
                candidate_sw = float(evaluator(candidate_weights))
                candidate = CandidateEvaluation(
                    cycle=cycle,
                    dimension=dim,
                    direction=direction,  # type: ignore[arg-type]
                    candidate_weights=candidate_weights,
                    social_welfare=candidate_sw,
                )
                candidates.append(candidate)

                if candidate_sw > best_sw + self.min_improvement:
                    best_sw = candidate_sw
                    best_weights = candidate_weights
                    best_dim = dim
                    best_direction = direction  # type: ignore[assignment]

        accepted = best_dim is not None
        if accepted:
            ledger.update_ecu_weights(best_weights)
        else:
            # Local optimum for the current coordinate neighbourhood.
            self.local_optimum_reached = True

        update = OrchestratorUpdate(
            cycle=cycle,
            observed_sw=float(observed_sw),
            old_weights=old_weights,
            new_weights=dict(ledger.ecu_weights),
            accepted=accepted,
            best_dimension=best_dim,
            best_direction=best_direction,
            best_candidate_sw=best_sw,
            candidates=candidates,
        )
        self._history.append(update)
        return update

    def _normalise(
        self,
        weights: dict[str, float],
        target_dim: str,
        new_val: float,
        target_sum: float,
    ) -> dict[str, float]:
        """Perturb one ECU weight and rescale the others to keep Σw^ECU fixed."""
        new_val = max(self.min_weight, min(self.max_weight, new_val))
        out = dict(weights)
        out[target_dim] = new_val

        others = [d for d in out if d != target_dim]
        remaining = max(0.0, target_sum - new_val)
        other_sum = sum(out[d] for d in others)

        if others and other_sum > 0:
            scale = remaining / other_sum
            for d in others:
                out[d] = max(self.min_weight, min(self.max_weight, out[d] * scale))

        # If clipping disturbed the total, do one final proportional correction
        # over non-target dimensions that can still move. This keeps the budget
        # close without violating min/max bounds.
        total = sum(out.values())
        if others and abs(total - target_sum) > 1e-9:
            adjustable = [d for d in others if self.min_weight < out[d] < self.max_weight]
            if adjustable:
                diff = target_sum - total
                add_each = diff / len(adjustable)
                for d in adjustable:
                    out[d] = max(self.min_weight, min(self.max_weight, out[d] + add_each))

        return out

    @property
    def history(self) -> list[OrchestratorUpdate]:
        return list(self._history)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "algorithm": "coordinate_search_counterfactual",
            "step_size": self.step_size,
            "update_every": self.update_every,
            "min_improvement": self.min_improvement,
            "local_optimum_reached": self.local_optimum_reached,
            "updates": [u.to_dict() for u in self._history],
        }
