"""
core/orchestrator.py

Orchestrator update rule for the peer-review OMAS platform.

The model now separates two weight vectors:

    w^SW   fixed social-planner valuation weights used only to compute SW
    w^ECU  variable incentive weights used only to pay agents ECUs

Implemented formulas
--------------------
    ecu_i^(t) = Σ_q w_q^ECU · (1/(n-1))Σ_{j≠i}s_ji^(t)(q)

    SW^(t) = Σ_q w_q^SW · (1/n)Σ_i (1/(n-1))Σ_{j≠i}s_ji^(t)(q)

The Orchestrator controls w^ECU, not w^SW. Since the platform observes only the
realized peer scores from the current round and cannot re-run counterfactual agent
responses for every candidate weight vector, the local search uses a simple response
approximation: increasing a dimension's ECU incentive is predicted to raise the next
round's mean score on that dimension slightly. Candidate ECU weights are therefore
evaluated by the approximate objective:

    max_{w^ECU} SW^(t+1) ≈ Σ_q w_q^SW · s̄_q^(t)(w^ECU)

where s̄_q is the mean peer score by dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.ecu import EcuLedger
    from core.state import PeerReviewOutput


@dataclass
class OrchestratorUpdate:
    """Record of one ECU-weight update decision."""
    cycle: int
    dimension: str
    old_weight: float
    new_weight: float
    sw_before: float
    sw_after: float
    direction: str  # "+" or "-"

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "dimension": self.dimension,
            "old_weight": round(self.old_weight, 4),
            "new_weight": round(self.new_weight, 4),
            "sw_before": round(self.sw_before, 4),
            "sw_after": round(self.sw_after, 4),
            "direction": self.direction,
        }


class Orchestrator:
    """
    Updates the variable ECU incentive vector w^ECU.

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
        response_strength: float = 0.05,
    ):
        self.step_size = step_size
        self.update_every = update_every
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.enabled = enabled
        self.response_strength = response_strength
        self._history: list[OrchestratorUpdate] = []

    def step(
        self,
        cycle: int,
        round_reviews: list[PeerReviewOutput],
        ledger: EcuLedger,
    ) -> dict[str, float]:
        """
        Run one Orchestrator step.

        Returns the current ECU weights. On update rounds, each dimension is
        perturbed by ±ε and the candidate with the highest approximate SW is kept.
        """
        # Always record realized SW for this round using fixed w^SW.
        ledger.record_social_welfare(cycle, round_reviews)

        if not self.enabled:
            return dict(ledger.ecu_weights)

        if (cycle + 1) % self.update_every != 0:
            return dict(ledger.ecu_weights)

        weights = dict(ledger.ecu_weights)
        target_sum = sum(weights.values())
        mean_scores = ledger.mean_peer_scores(round_reviews)
        sw_current = self._compute_sw(mean_scores, ledger.sw_weights)

        for dim in list(weights.keys()):
            old_w = weights[dim]

            w_plus = self._normalise(weights, dim, old_w + self.step_size, target_sum)
            predicted_plus = self._predict_mean_scores(mean_scores, weights, w_plus)
            sw_plus = self._compute_sw(predicted_plus, ledger.sw_weights)

            w_minus = self._normalise(weights, dim, old_w - self.step_size, target_sum)
            predicted_minus = self._predict_mean_scores(mean_scores, weights, w_minus)
            sw_minus = self._compute_sw(predicted_minus, ledger.sw_weights)

            best_sw = max(sw_plus, sw_minus)
            if best_sw > sw_current + 1e-9:
                if sw_plus >= sw_minus:
                    new_weights = w_plus
                    direction = "+"
                    new_sw = sw_plus
                else:
                    new_weights = w_minus
                    direction = "-"
                    new_sw = sw_minus

                self._history.append(OrchestratorUpdate(
                    cycle=cycle,
                    dimension=dim,
                    old_weight=old_w,
                    new_weight=new_weights[dim],
                    sw_before=sw_current,
                    sw_after=new_sw,
                    direction=direction,
                ))
                weights = new_weights
                sw_current = new_sw

        ledger.update_ecu_weights(weights)
        return weights

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
        return out

    def _predict_mean_scores(
        self,
        current_scores: dict[str, float],
        current_ecu_weights: dict[str, float],
        candidate_ecu_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Approximate s̄_q^(t)(w^ECU) for a candidate incentive vector.

        The approximation is intentionally conservative: a positive relative change
        in a dimension's ECU weight predicts a small improvement in that dimension's
        next mean score; a negative change predicts a small decrease. Scores remain
        clipped to [0, 1].
        """
        predicted: dict[str, float] = {}
        for dim, score in current_scores.items():
            old = max(current_ecu_weights.get(dim, 1.0), 1e-9)
            new = candidate_ecu_weights.get(dim, old)
            relative_change = (new - old) / old
            predicted[dim] = min(1.0, max(0.0, score + self.response_strength * relative_change))
        return predicted

    @staticmethod
    def _compute_sw(mean_scores: dict[str, float], sw_weights: dict[str, float]) -> float:
        """SW = Σ_q w_q^SW · s̄_q."""
        return sum(sw_weights.get(dim, 1.0) * mean_scores.get(dim, 0.0) for dim in sw_weights)

    @property
    def history(self) -> list[OrchestratorUpdate]:
        return list(self._history)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "step_size": self.step_size,
            "update_every": self.update_every,
            "response_strength": self.response_strength,
            "updates": [u.to_dict() for u in self._history],
        }
