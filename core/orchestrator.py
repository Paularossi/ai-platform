"""
core/orchestrator.py

Importance-vote gradient Orchestrator for the peer-review OMAS platform.

The platform separates two weight vectors:

    w^SW   fixed social-planner valuation weights, used only to compute SW
    w^ECU  variable incentive weights, used only to pay agents ECUs

This Orchestrator updates w^ECU every K rounds using importance votes collected
from agents during Phase 2 peer review. Each agent distributes 100 points across
the quality dimensions, answering the question:
"Given the topic of the debate and your role, which dimensions are most important
to you?"

Update rule (Robbins-Monro gradient ascent)
-------------------------------------------
Let v_i = agent i's importance vector (sums to 100).
Let v̄_q  = (1/n) * sum_i v_i_q             (mean vote across agents, sums to 100)
Let v̂_q  = v̄_q / 100                       (normalised, sums to 1)

New weights before renormalisation:
    w_q^ECU(t+1) = w_q^ECU(t) + (1/t) * v̂_q

Then renormalise so sum(w_q^ECU) = initial_sum, preserving the update direction
while keeping the ECU budget stable across rounds.

The 1/t learning rate satisfies the Robbins-Monro conditions (sum 1/t = inf,
sum 1/t^2 < inf), guaranteeing convergence as t -> inf. For finite T the schedule
gives larger updates early and diminishing updates later, without unbounded growth.

Advantages over the earlier sandbox coordinate-search approach:
  - Zero additional LLM calls (votes collected in regular Phase 2)
  - Participatory: agents determine the update direction from their perspective
  - No counterfactual re-runs required
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.ecu import EcuLedger


@dataclass
class OrchestratorUpdate:
    """Record of one importance-vote gradient step."""
    cycle: int
    old_weights: dict[str, float]
    raw_votes: dict[str, dict[str, float]]
    mean_votes: dict[str, float]
    normalised_votes: dict[str, float]
    learning_rate: float
    pre_normalise_weights: dict[str, float]
    new_weights: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "learning_rate": round(self.learning_rate, 6),
            "old_weights": {k: round(v, 6) for k, v in self.old_weights.items()},
            "raw_votes": {
                agent: {d: round(v, 2) for d, v in votes.items()}
                for agent, votes in self.raw_votes.items()
            },
            "mean_votes": {k: round(v, 4) for k, v in self.mean_votes.items()},
            "normalised_votes": {k: round(v, 6) for k, v in self.normalised_votes.items()},
            "pre_normalise_weights": {k: round(v, 6) for k, v in self.pre_normalise_weights.items()},
            "new_weights": {k: round(v, 6) for k, v in self.new_weights.items()},
        }


class Orchestrator:
    """
    Updates w^ECU using importance-vote gradient ascent.

    The fixed SW valuation vector w^SW lives in EcuLedger.sw_weights and is
    never changed by this class.

    Parameters
    ----------
    update_every : int
        Run the update every K rounds (default 1 = every round).
    min_weight : float
        Floor for any individual weight after renormalisation (default 0.1).
    enabled : bool
        If False, all update calls are no-ops.
    """

    def __init__(
        self,
        update_every: int = 1,
        min_weight: float = 0.1,
        enabled: bool = True,
    ):
        self.update_every = max(1, int(update_every))
        self.min_weight = float(min_weight)
        self.enabled = enabled
        self._history: list[OrchestratorUpdate] = []

    def should_update(self, cycle: int, *, is_final_cycle: bool = False) -> bool:
        """Return True when the Orchestrator should run after this real cycle."""
        if not self.enabled or is_final_cycle:
            return False
        return (cycle + 1) % self.update_every == 0

    def update(
        self,
        cycle: int,
        ledger: "EcuLedger",
        importance_votes: dict[str, dict[str, float]],
    ) -> OrchestratorUpdate | None:
        """
        Apply one importance-vote gradient step.

        Parameters
        ----------
        cycle : int
            Completed real cycle (0-indexed). Learning rate = 1 / (cycle + 1).
        ledger : EcuLedger
            Official ledger; ECU weights updated in-place.
        importance_votes : dict[str, dict[str, float]]
            {agent_name: {dimension: points}} where each agent's points sum to 100.
            Agents with invalid or zero-sum votes are silently excluded.
        """
        if not self.enabled:
            return None

        dim_names = list(ledger.ecu_weights.keys())

        # ── 1. Validate and normalise each agent's vote to sum exactly 100 ──
        valid_votes: dict[str, dict[str, float]] = {}
        for agent, votes in importance_votes.items():
            if not isinstance(votes, dict):
                continue
            filtered = {d: max(0.0, float(votes.get(d, 0.0))) for d in dim_names}
            total = sum(filtered.values())
            if total <= 0:
                continue
            valid_votes[agent] = {d: v * 100.0 / total for d, v in filtered.items()}

        if not valid_votes:
            return None

        # ── 2. Average across agents (sums to 100) ───────────────────────────
        n = len(valid_votes)
        mean_votes: dict[str, float] = {
            d: sum(v[d] for v in valid_votes.values()) / n
            for d in dim_names
        }

        # ── 3. Normalise to sum to 1 ─────────────────────────────────────────
        total = sum(mean_votes.values())
        normalised: dict[str, float] = {d: mean_votes[d] / total for d in dim_names}

        # ── 4. Gradient step: w_q += (1/t) * v̂_q ───────────────────────────
        t = cycle + 1
        lr = 1.0 / t
        old_weights = dict(ledger.ecu_weights)
        initial_sum = sum(old_weights.values())
        stepped = {d: old_weights[d] + lr * normalised[d] for d in dim_names}

        # ── 5. Renormalise to initial_sum ────────────────────────────────────
        new_weights = self._renormalise(stepped, initial_sum)
        ledger.update_ecu_weights(new_weights)

        update = OrchestratorUpdate(
            cycle=cycle,
            old_weights=old_weights,
            raw_votes=valid_votes,
            mean_votes=mean_votes,
            normalised_votes=normalised,
            learning_rate=lr,
            pre_normalise_weights=stepped,
            new_weights=new_weights,
        )
        self._history.append(update)
        return update

    def _renormalise(self, weights: dict[str, float], target_sum: float) -> dict[str, float]:
        """Scale weights to target_sum, clip at min_weight, redistribute residual."""
        total = sum(weights.values())
        if total <= 0:
            n = len(weights)
            return {d: target_sum / n for d in weights}

        out = {d: v * target_sum / total for d, v in weights.items()}

        # Clip and redistribute
        clipped = {d: max(self.min_weight, v) for d, v in out.items()}
        excess = sum(clipped.values()) - target_sum
        if abs(excess) > 1e-9:
            adjustable = [d for d in clipped if clipped[d] > self.min_weight]
            if adjustable:
                cut = excess / len(adjustable)
                for d in adjustable:
                    clipped[d] = max(self.min_weight, clipped[d] - cut)

        return clipped

    @property
    def history(self) -> list[OrchestratorUpdate]:
        return list(self._history)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "algorithm": "importance_vote_gradient",
            "update_every": self.update_every,
            "min_weight": self.min_weight,
            "updates": [u.to_dict() for u in self._history],
        }