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
    ecu_weights: dict[str, float] = field(default_factory=dict)


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
        φ₁ — what each agent sees about others' contributions before writing
        their own (Phase 1 visibility).
        "Blind"          → agent sees nothing from previous round
        "Previous round" → agent sees every agent's most recent contribution
        "Full history"   → agent sees all contributions from all agents and rounds
    review_depth : str
        φ₂ — what a reviewer sees about the agent they are scoring during
        peer review (Phase 2 review depth).
        "Current Only"   → reviewer sees only the current round's contribution
        "Previous Round" → reviewer sees current + previous round side-by-side
        "Full History"   → reviewer sees the full contribution trajectory
    agent_names : list[str]
        Ordered list of agent names (used for convergence checks).
    ledger : EcuLedger | None
        If provided, ecu scoring runs after every submission.
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
        review_depth: str = "Previous Round",
    ):
        self.item_id = item_id
        self.item_data = dict(item_data)
        self.visibility_mode = visibility_mode
        self.agent_names = agent_names
        self.ledger = ledger
        self.ecu_info_condition = ecu_info_condition
        self.review_depth = review_depth

        self._log: list[AgentOutput] = []
        self._peer_review_log: list[PeerReviewOutput] = []
        self._prompt_log: list[dict] = []  # {cycle, agent, phase, prompt, response}
        self._current_contribution: Any = None
        self.originator_name: str | None = None
        self.originator_contribution: Any = None
        self.converged: bool = False
        self._roster_events: list[dict] = []  # {cycle, action: "add"/"remove", agent_name}

    # ------------------------------------------------------------------
    # Dynamic roster (Agent 0 mode)
    # ------------------------------------------------------------------

    def add_agent(self, name: str, cycle: int | None = None) -> None:
        """Add an agent to the active roster starting the given cycle."""
        if name not in self.agent_names:
            self.agent_names.append(name)
            self._roster_events.append({"cycle": cycle, "action": "add", "agent_name": name})

    def remove_agent(self, name: str, cycle: int | None = None) -> None:
        """
        Remove an agent from the active roster. Past contributions and peer
        reviews stay in the log — only future participation stops, since
        _log/_peer_review_log are never touched here.
        """
        if name in self.agent_names:
            self.agent_names.remove(name)
            self._roster_events.append({"cycle": cycle, "action": "remove", "agent_name": name})

    @property
    def roster_events(self) -> list[dict]:
        return list(self._roster_events)

    # ------------------------------------------------------------------
    # Core interface used by the protocol
    # ------------------------------------------------------------------

    def build_context(self, agent_name: str, cycle: int) -> ContextPacket:
        """
        Build and return a ContextPacket for the given agent.
        Called by the protocol just before dispatching to an agent.
        """
        ecu_balances: dict[str, float] = {}
        ecu_weights: dict[str, float] = {}
        if self.ledger:
            if self.ecu_info_condition == "transparent":
                ecu_balances = self.ledger.balances
                ecu_weights = dict(self.ledger.ecu_weights)
            elif self.ecu_info_condition == "semi-transparent":
                own = self.ledger.balance_for(agent_name)
                ecu_balances = {agent_name: own}
                # Cached per cycle on the ledger so this matches whatever the
                # Phase 2 peer-review prompt shows for the same cycle - see
                # EcuLedger.noisy_weights_for_cycle().
                ecu_weights = self.ledger.noisy_weights_for_cycle(cycle)

        return ContextPacket(
            item_id=self.item_id,
            item_data=self.item_data,
            current_contribution=self._current_contribution,
            visible_history=self._build_visible_history(agent_name),
            cycle=cycle,
            agent_name=agent_name,
            ecu_balances=ecu_balances,
            ecu_weights=ecu_weights,
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
            self._current_contribution = output.contribution

        if not self._log:
            self.originator_name = output.agent_name
            self.originator_contribution = output.contribution

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

    def check_convergence(self) -> bool:
        """
        Return True if all agents produced identical contributions last cycle.
        Sets self.converged = True if so. Requires every agent to have submitted.
        """
        latest = self._latest_per_agent()
        if set(latest.keys()) != set(self.agent_names):
            return False
        outputs = list(latest.values())
        ref = outputs[0]
        for other in outputs[1:]:
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
        return self._current_contribution

    # Keep current_labels as a convenience alias for classification tasks
    @property
    def current_labels(self) -> dict[str, Any]:
        if isinstance(self._current_contribution, dict):
            return self._current_contribution
        return {}

    def set_current_labels(self, labels: dict[str, Any]) -> None:
        """
        Override the current contribution state directly.
        Used by aggregating protocols (e.g. Crowd) to push the round's
        aggregated result after all individual submissions are collected.
        """
        self._current_contribution = labels

    @property
    def num_submissions(self) -> int:
        return len(self._log)

    def last_submission(self) -> AgentOutput | None:
        return self._log[-1] if self._log else None

    def submissions_by_agent(self, agent_name: str) -> list[AgentOutput]:
        return [o for o in self._log if o.agent_name == agent_name]

    def log_prompt(self, cycle: int, agent_name: str, phase: str,
                   prompt: str, response: str = "") -> None:
        """Record a prompt sent to an agent for debugging."""
        self._prompt_log.append({
            "cycle": cycle,
            "agent": agent_name,
            "phase": phase,  # "contribution" or "peer_review"
            "prompt": prompt,
            "response": response,
        })

    @property
    def prompt_log(self) -> list[dict]:
        return list(self._prompt_log)

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
            "prompt_log": self._prompt_log,
            "roster_events": self._roster_events,
        }
        if self.ledger:
            d["ecu"] = self.ledger.to_dict()
        return d

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


    def _build_visible_history(self, agent_name: str) -> list[AgentOutput]:
        """
        φ₁ — filter the Phase 1 message log per visibility_mode.

        "Blind"          → empty list (agent writes without seeing anyone)
        "Previous round" → last submission from each agent (one per agent)
        "Full history"   → entire log across all rounds

        Legacy values "Current state only", "Summary only", "Previous agent only"
        are mapped to their equivalents for backwards compatibility.
        """
        if not self._log:
            return []

        mode = self.visibility_mode

        if mode in ("Blind", "Current state only"):
            return []
        elif mode in ("Previous round", "Summary only"):
            return list(self._latest_per_agent().values())
        elif mode == "Full history":
            return list(self._log)
        elif mode == "Previous agent only":
            # Legacy gossip mode — keep for compatibility
            return [self._log[-1]]

        return list(self._latest_per_agent().values())

    def build_review_history(self, cycle: int) -> dict[str, list[str]]:
        """
        φ₂ — build per-agent contribution history for Phase 2 peer review.

        Returns {agent_name: [contribution_round_0, contribution_round_1, ...]}
        The caller (protocol) slices this based on review_depth.

        "Current Only"   → only the current cycle's contribution
        "Previous Round" → current + previous cycle (if any)
        "Full History"   → all cycles up to and including current
        """
        # Group contributions by agent and cycle
        per_agent: dict[str, list[tuple[int, Any]]] = {}
        for out in self._log:
            if out.agent_name not in per_agent:
                per_agent[out.agent_name] = []
            per_agent[out.agent_name].append((out.cycle, out.contribution))

        result: dict[str, list[str]] = {}
        for agent, entries in per_agent.items():
            # Sort by cycle
            entries_sorted = sorted(entries, key=lambda x: x[0])

            if self.review_depth == "Current Only":
                # Only the current cycle
                current = [c for c in entries_sorted if c[0] == cycle]
                result[agent] = [str(c[1]) for c in current] if current else []

            elif self.review_depth == "Previous Round":
                # Current + one round back
                relevant = [c for c in entries_sorted if c[0] >= cycle - 1]
                result[agent] = [str(c[1]) for c in relevant]

            else:  # "Full History"
                result[agent] = [str(c[1]) for c in entries_sorted]

        return result

    def _latest_per_agent(self) -> dict[str, AgentOutput]:
        """Return {agent_name: most_recent_output} for every agent."""
        result: dict[str, AgentOutput] = {}
        for output in self._log:
            result[output.agent_name] = output  # later entries overwrite earlier
        return result