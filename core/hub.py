"""
core/hub.py

The CommunicationHub sits at the centre of the agent network.
Agents never communicate directly with each other - they submit
outputs to the hub and receive context packets from the hub.

Responsibilities
----------------
1. Store every AgentOutput in an ordered message log
2. When an agent is about to be called, build a ContextPacket
   containing only what that agent is allowed to see (per visibility_mode)
3. Track convergence across the full agent roster
4. Expose a clean summary of the current state for logging / UI

This design means the hub is the single source of truth for an
experiment run on one item. The gossip protocol just drives the
outer loop (cycles, stopping rule) and delegates everything else here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.state import AgentOutput, PeerReviewOutput

if TYPE_CHECKING:
    from core.ecu import EcuLedger


# ---------------------------------------------------------------------------
# ContextPacket - what an agent receives from the hub before its turn
# ---------------------------------------------------------------------------

@dataclass
class ContextPacket:
    """
    Everything an agent needs to produce its next output.
    Built by the hub; consumed by the agent's prompt builder.

    Fields
    ------
    item_id : str
    item_data : dict
        The raw input (text, image path / URL, etc.)
    current_contribution : Any
        The most recent agreed contribution/labels, regardless of who set it.
        str for deliberation tasks, dict[field→verdict] for classification.
    visible_history : list[AgentOutput]
        The slice of the message log this agent is allowed to see.
        Content depends on visibility_mode.
    cycle : int
        Which cycle we are currently in (0-indexed).
    agent_name : str
        The name of the agent about to be called.
    ecu_balances : dict[str, float]
        Current ecu balances for all agents.
        Populated only under the transparent information condition;
        empty dict otherwise.
    """
    item_id: str
    item_data: dict[str, Any]
    current_contribution: Any
    visible_history: list[AgentOutput]
    cycle: int
    agent_name: str
    ecu_balances: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CommunicationHub
# ---------------------------------------------------------------------------

class CommunicationHub:
    """
    Central communication hub for one experiment item.

    Parameters
    ----------
    item_id : str
    item_data : dict
        Raw input data for this item.
    visibility_mode : str
        One of: "Current state only", "Previous agent only",
                "Full history", "Summary only"
    agent_names : list[str]
        Ordered list of agent names (used for convergence checks).
    ledger : EcuLedger | None
        If provided, ecu scoring runs after every submission.
    scorer : QualityScorer | None
        Required when ledger is provided. Scores each contribution.
    ecu_info_condition : str
        "transparent"  — agents see ecu balances in their context packet
        "opaque"       — balances are hidden (default)
    """

    def __init__(
        self,
        item_id: str,
        item_data: dict[str, Any],
        visibility_mode: str,
        agent_names: list[str],
        ledger: EcuLedger | None = None,
        ecu_info_condition: str = "opaque",
    ):
        self.item_id = item_id
        self.item_data = copy.deepcopy(item_data)
        self.visibility_mode = visibility_mode
        self.agent_names = agent_names
        self.ledger = ledger
        self.ecu_info_condition = ecu_info_condition

        self._log: list[AgentOutput] = []
        self._peer_review_log: list[PeerReviewOutput] = []
        self._current_contribution: Any = None
        self.originator_name: str | None = None
        self.originator_contribution: Any = None
        self.converged: bool = False

    # ------------------------------------------------------------------
    # Core interface used by the protocol
    # ------------------------------------------------------------------

    def build_context(self, agent_name: str, cycle: int) -> ContextPacket:
        """
        Build and return a ContextPacket for the given agent.
        Called by the protocol just before dispatching to an agent.
        """
        ecu_balances: dict[str, float] = {}
        if self.ledger and self.ecu_info_condition == "transparent":
            ecu_balances = self.ledger.balances

        return ContextPacket(
            item_id=self.item_id,
            item_data=copy.deepcopy(self.item_data),
            current_contribution=copy.deepcopy(self._current_contribution),
            visible_history=self._build_visible_history(agent_name),
            cycle=cycle,
            agent_name=agent_name,
            ecu_balances=ecu_balances,
        )

    def submit(self, output: AgentOutput) -> None:
        """Accept a Phase 1 AgentOutput and store it."""
        prev = self._current_contribution

        if isinstance(output.contribution, dict) and isinstance(prev, dict):
            output.changed = {
                fname: (output.contribution.get(fname) != prev.get(fname))
                for fname in output.contribution
            }
        elif output.contribution != prev:
            output.changed = {"__contribution__": True}
        else:
            output.changed = {}

        if isinstance(output.contribution, dict) and isinstance(self._current_contribution, dict):
            self._current_contribution.update(output.contribution)
        else:
            self._current_contribution = copy.deepcopy(output.contribution)

        if not self._log:
            self.originator_name = output.agent_name
            self.originator_contribution = copy.deepcopy(output.contribution)

        self._log.append(output)

    def submit_peer_review(self, review: PeerReviewOutput) -> None:
        """
        Accept a Phase 2 PeerReviewOutput and store it.
        ECU calculation is handled by the protocol after all reviews are collected.
        """
        self._peer_review_log.append(review)

    def compute_ecus_for_round(self, cycle: int) -> dict[str, float]:
        """
        After all peer reviews for a round are submitted, compute and record
        ECUs for every agent. Returns {agent_name: ecu_earned}.
        Called by the protocol once per round after all reviews arrive.
        """
        if not self.ledger:
            return {}

        round_reviews = [r for r in self._peer_review_log if r.cycle == cycle]
        round_contributions = {
            o.contribution for o in self._log if o.cycle == cycle
        }

        # Map agent_name → contribution for this cycle
        contrib_map: dict[str, Any] = {
            o.agent_name: o.contribution
            for o in self._log if o.cycle == cycle
        }

        earned: dict[str, float] = {}
        for agent_name in self.agent_names:
            contribution = contrib_map.get(agent_name)
            ecu = self.ledger.record_from_reviews(
                agent_name=agent_name,
                cycle=cycle,
                item_id=self.item_id,
                contribution=contribution,
                reviews=round_reviews,
            )
            earned[agent_name] = ecu

            # Attach scores back to the AgentOutput for logging
            for out in self._log:
                if out.agent_name == agent_name and out.cycle == cycle:
                    last_record = self.ledger.history[-len(self.agent_names):]
                    for rec in last_record:
                        if rec.agent_name == agent_name and rec.cycle == cycle:
                            out.ecu_scores = rec.aggregated_scores
                            out.ecu_earned = rec.ecu_earned
                            break

        return earned

    def check_convergence(self, field_names: list[str] | None = None) -> bool:
        """
        Return True if all agents' most recent outputs agree.
        For classification tasks: all agents agree on all (or specified) fields.
        For deliberation tasks: all agents' contribution strings are identical.
        Sets self.converged = True if so.
        Requires every agent to have submitted at least once.
        """
        latest = self._latest_per_agent()

        if set(latest.keys()) != set(self.agent_names):
            return False

        agents = list(latest.keys())
        ref = latest[agents[0]]

        for agent in agents[1:]:
            other = latest[agent]
            # Classification: compare field by field
            if ref.is_classification and other.is_classification:
                for fname, val in other.labels.items():
                    if field_names and fname not in field_names:
                        continue
                    if val != ref.labels.get(fname):
                        return False
            else:
                # Deliberation: exact string match (strict)
                if other.contribution != ref.contribution:
                    return False

        self.converged = True
        return True

    # ------------------------------------------------------------------
    # Accessors for logging / UI
    # ------------------------------------------------------------------

    @property
    def log(self) -> list[AgentOutput]:
        """Full message log (read-only view)."""
        return list(self._log)

    @property
    def current_contribution(self) -> Any:
        return copy.deepcopy(self._current_contribution)

    # Keep current_labels as a convenience alias for classification tasks
    @property
    def current_labels(self) -> dict[str, Any]:
        if isinstance(self._current_contribution, dict):
            return copy.deepcopy(self._current_contribution)
        return {}

    def set_current_labels(self, labels: dict[str, Any]) -> None:
        """
        Override the current contribution state directly.
        Used by aggregating protocols (e.g. Crowd) to push the round's
        aggregated result after all individual submissions are collected.
        """
        if isinstance(self._current_contribution, dict):
            self._current_contribution = copy.deepcopy(labels)
        else:
            self._current_contribution = copy.deepcopy(labels)

    @property
    def num_submissions(self) -> int:
        return len(self._log)

    def last_submission(self) -> AgentOutput | None:
        return self._log[-1] if self._log else None

    def submissions_by_agent(self, agent_name: str) -> list[AgentOutput]:
        return [o for o in self._log if o.agent_name == agent_name]

    @property
    def peer_review_log(self) -> list[PeerReviewOutput]:
        return list(self._peer_review_log)

    @property
    def ecu_balances(self) -> dict[str, float]:
        if self.ledger:
            return self.ledger.balances
        return {}

    def to_dict(self) -> dict:
        d = {
            "item_id": self.item_id,
            "current_contribution": self._current_contribution,
            "originator_name": self.originator_name,
            "originator_contribution": self.originator_contribution,
            "converged": self.converged,
            "num_submissions": self.num_submissions,
            "log": [o.to_dict() for o in self._log],
            "peer_review_log": [p.to_dict() for p in self._peer_review_log],
        }
        if self.ledger:
            d["ecu"] = self.ledger.to_dict()
        return d

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def ecu_info_str(self, cycle: int) -> str:
        """
        Build the ECU context string shown to agents based on information condition.
        Empty string under opaque condition.
        """
        if not self.ledger or self.ecu_info_condition == "opaque":
            return ""

        balances = self.ledger.balances
        weights = self.ledger.weights

        if self.ecu_info_condition == "transparent":
            w_str = ", ".join(f"{k}={v:.2f}" for k, v in weights.items())
            b_str = ", ".join(f"{k}={v:.3f}" for k, v in balances.items())
            return (
                f"[ECU update — Round {cycle + 1}] "
                f"Current weights: {w_str}. "
                f"Cumulative balances: {b_str}."
            )
        elif self.ecu_info_condition == "semi-transparent":
            import random
            noisy = {k: round(v * random.uniform(0.8, 1.2), 2) for k, v in weights.items()}
            w_str = ", ".join(f"{k}≈{v}" for k, v in noisy.items())
            b_str = ", ".join(f"{k}={v:.3f}" for k, v in balances.items())
            return (
                f"[ECU update — Round {cycle + 1}] "
                f"Approximate weights: {w_str}. "
                f"Your balance: {balances.get(list(balances.keys())[0], 0):.3f} ecus."
            )
        return ""

    def _build_visible_history(self, agent_name: str) -> list[AgentOutput]:
        """
        Filter the message log according to visibility_mode.

        "Current state only"  → empty list
        "Previous agent only" → last submission in the log
        "Full history"        → entire log
        "Summary only"        → last submission per agent
        """
        if not self._log:
            return []

        mode = self.visibility_mode

        if mode == "Current state only":
            return []
        elif mode == "Previous agent only":
            return [self._log[-1]]
        elif mode == "Full history":
            return list(self._log)
        elif mode == "Summary only":
            return list(self._latest_per_agent().values())

        return list(self._log)

    def _latest_per_agent(self) -> dict[str, AgentOutput]:
        """Return {agent_name: most_recent_output} for every agent."""
        result: dict[str, AgentOutput] = {}
        for output in self._log:
            result[output.agent_name] = output  # later entries overwrite earlier
        return result

#TODO: recompute the changed label for all agents against its own annotations from the previous round (not against the hub aggregation)