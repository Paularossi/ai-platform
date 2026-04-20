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
from typing import Any

from core.state import AgentOutput


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
        The raw input (caption, image path / URL, etc.)
    current_labels : dict
        The most recent label for each field, regardless of who set it.
    visible_history : list[AgentOutput]
        The slice of the message log this agent is allowed to see.
        Content depends on visibility_mode - see Hub._build_visible_history.
    cycle : int
        Which cycle we are currently in (0-indexed).
    agent_name : str
        The name of the agent about to be called (so the agent knows who it is).
    """
    item_id: str
    item_data: dict[str, Any]
    current_labels: dict[str, Any]
    visible_history: list[AgentOutput]
    cycle: int
    agent_name: str


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
    """

    def __init__(
        self,
        item_id: str,
        item_data: dict[str, Any],
        visibility_mode: str,
        agent_names: list[str],
    ):
        self.item_id = item_id
        self.item_data = copy.deepcopy(item_data)
        self.visibility_mode = visibility_mode
        self.agent_names = agent_names

        # The message log - append-only, ordered chronologically
        self._log: list[AgentOutput] = []

        # Current label state - updated after every submission
        self._current_labels: dict[str, Any] = {}

        # Originator info - set on first submission
        self.originator_name: str | None = None
        self.originator_labels: dict[str, Any] = {}

        # Convergence flag - set by check_convergence()
        self.converged: bool = False

    # ------------------------------------------------------------------
    # Core interface used by the protocol
    # ------------------------------------------------------------------

    def build_context(self, agent_name: str, cycle: int) -> ContextPacket:
        """
        Build and return a ContextPacket for the given agent.
        Called by the protocol just before dispatching to an agent.
        """
        return ContextPacket(
            item_id=self.item_id,
            item_data=copy.deepcopy(self.item_data),
            current_labels=copy.deepcopy(self._current_labels),
            visible_history=self._build_visible_history(agent_name),
            cycle=cycle,
            agent_name=agent_name,
        )

    def submit(self, output: AgentOutput) -> None:
        """
        Accept an AgentOutput from an agent and store it in the log.
        Updates current labels and records originator on first call.
        Also computes the per-field `changed` flags.
        """
        prev = copy.deepcopy(self._current_labels)

        # Compute which fields changed vs the previous state
        output.changed = {
            fname: (output.labels.get(fname) != prev.get(fname))
            for fname in output.labels
        }

        # Update the running label state
        self._current_labels.update(output.labels)

        # Record originator on the very first submission
        if not self._log:
            self.originator_name = output.agent_name
            self.originator_labels = copy.deepcopy(output.labels)

        self._log.append(output)

    def check_convergence(self, field_names: list[str] | None = None) -> bool:
        """
        Return True if all agents' most recent outputs agree on all
        (or the specified subset of) label fields.
        Sets self.converged = True if so.

        Requires at least one full cycle to have completed (i.e. every
        agent has submitted at least once).
        """
        latest = self._latest_per_agent()

        # Need every agent to have submitted at least once
        if set(latest.keys()) != set(self.agent_names):
            return False

        agents = list(latest.keys())
        ref = latest[agents[0]]

        for agent in agents[1:]:
            for fname, val in latest[agent].labels.items():
                if field_names and fname not in field_names:
                    continue
                if val != ref.labels.get(fname):
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
    def current_labels(self) -> dict[str, Any]:
        return copy.deepcopy(self._current_labels)

    def set_current_labels(self, labels: dict[str, Any]) -> None:
        """
        Override the current label state directly.
        Used by aggregating protocols (e.g. Crowd) to push the round's
        aggregated result after all individual submissions are collected.
        """
        self._current_labels = copy.deepcopy(labels)

    @property
    def num_submissions(self) -> int:
        return len(self._log)

    def last_submission(self) -> AgentOutput | None:
        return self._log[-1] if self._log else None

    def submissions_by_agent(self, agent_name: str) -> list[AgentOutput]:
        return [o for o in self._log if o.agent_name == agent_name]

    def to_dict(self) -> dict:
        """Serialise the hub state for saving / analysis."""
        return {
            "item_id": self.item_id,
            "current_labels": self._current_labels,
            "originator_name": self.originator_name,
            "originator_labels": self.originator_labels,
            "converged": self.converged,
            "num_submissions": self.num_submissions,
            "log": [o.to_dict() for o in self._log],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_visible_history(self, agent_name: str) -> list[AgentOutput]:
        """
        Filter the message log according to visibility_mode.

        "Current state only"  → empty list (agent sees only current_labels
                                 which is always passed via ContextPacket)
        "Previous agent only" → last submission in the log, regardless of who
        "Full history"        → entire log
        "Summary only"        → last submission per agent (one entry each)
        """
        if not self._log:
            return []

        mode = self.visibility_mode

        if mode == "Current state only":
            # No history - the agent only sees the current label state
            return []

        elif mode == "Previous agent only":
            # The single most recent submission
            return [self._log[-1]]

        elif mode == "Full history":
            return list(self._log)

        elif mode == "Summary only":
            # Most recent submission from each agent
            return list(self._latest_per_agent().values())

        # Fallback
        return list(self._log)

    def _latest_per_agent(self) -> dict[str, AgentOutput]:
        """Return {agent_name: most_recent_output} for every agent."""
        result: dict[str, AgentOutput] = {}
        for output in self._log:
            result[output.agent_name] = output  # later entries overwrite earlier
        return result

#TODO: recompute the changed label for all agents against its own annotations from the previous round (not against the hub aggregation)