"""
core/state.py

Defines the shared state object that is passed between agents during a
multi-agent annotation run. Each agent receives a copy of the current state,
produces an AgentOutput, and the orchestrator appends it to the history.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Single agent turn output
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    """
    The structured output produced by one agent on one turn.

    Fields
    ------
    agent_name : str
        Identifier of the agent that produced this output.
    cycle : int
        Which cycle (round) this turn belongs to (0-indexed).
    labels : dict[str, Any]
        Verdict per field: field_name → chosen option code (or list for multi-label).
    probabilities : dict[str, dict[str, float]]
        Full probability distribution per field.
        field_name → {option_code: probability, ...}  (values sum to ~1)
    confidence : dict[str, float]
        Per-field confidence = probability of the chosen verdict.
        Derived from probabilities; field_name → float in [0, 1].
    pros : list[str]
        Free-text arguments in favour of the classification.
    cons : list[str]
        Free-text arguments against the classification / alternative readings.
    changed : dict[str, bool]
        For each field, did this agent change the previous label?
        Populated by the hub after comparing with the previous state.
    raw_response : str | None
        The raw text returned by the LLM (for debugging / logging).
    timestamp : str
        ISO-8601 timestamp of when the output was produced.
    """

    agent_name: str
    cycle: int
    labels: dict[str, Any] = field(default_factory=dict)
    probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    changed: dict[str, bool] = field(default_factory=dict)
    raw_response: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "cycle": self.cycle,
            "labels": self.labels,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "pros": self.pros,
            "cons": self.cons,
            "changed": self.changed,
            "raw_response": self.raw_response,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Shared experiment state
# ---------------------------------------------------------------------------

@dataclass
class ExperimentState:
    """
    The evolving shared state for a single item (e.g. one ad) being annotated
    by a multi-agent protocol.

    The orchestrator creates one ExperimentState per item, passes it to each
    agent in sequence, and appends each AgentOutput to `history`.

    Fields
    ------
    item_id : str
        Identifier for the item being annotated.
    item_data : dict[str, Any]
        The raw input data for this item (image URL, caption, etc.).
    current_labels : dict[str, Any]
        The most recent label values - updated after each agent turn.
    history : list[AgentOutput]
        Full ordered trajectory of all agent outputs so far.
    cycle : int
        Current cycle number (increments when all agents have gone once).
    converged : bool
        Set to True by the orchestrator when a stopping criterion is met.
    originator_name : str | None
        Name of the agent that produced the very first label (cycle 0, turn 0).
    originator_labels : dict[str, Any]
        Labels as set by the originating agent.
    metadata : dict[str, Any]
        Arbitrary extra info (experiment name, protocol, etc.).
    """

    item_id: str
    item_data: dict[str, Any] = field(default_factory=dict)
    current_labels: dict[str, Any] = field(default_factory=dict)
    history: list[AgentOutput] = field(default_factory=list)
    cycle: int = 0
    converged: bool = False
    originator_name: str | None = None
    originator_labels: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def apply_output(self, output: AgentOutput) -> None:
        """
        Apply an agent's output to the shared state:
        - compute per-field `changed` flags
        - update current_labels
        - record originator on first turn
        - append to history
        """
        prev = copy.deepcopy(self.current_labels)

        # Compute changed flags
        output.changed = {
            field_name: (output.labels.get(field_name) != prev.get(field_name))
            for field_name in output.labels
        }

        # Update current labels
        self.current_labels.update(output.labels)

        # Record originator
        if not self.history:
            self.originator_name = output.agent_name
            self.originator_labels = copy.deepcopy(output.labels)

        self.history.append(output)

    def increment_cycle(self) -> None:
        """Call this after all agents have completed a full round."""
        self.cycle += 1

    def last_output_for_agent(self, agent_name: str) -> AgentOutput | None:
        """Return the most recent output from a specific agent, or None."""
        for output in reversed(self.history):
            if output.agent_name == agent_name:
                return output
        return None

    def last_output(self) -> AgentOutput | None:
        """Return the most recent output from any agent."""
        return self.history[-1] if self.history else None

    # ------------------------------------------------------------------
    # Convergence helpers
    # ------------------------------------------------------------------

    def latest_labels_by_agent(self) -> dict[str, dict[str, Any]]:
        """
        Return a dict of {agent_name: labels} using each agent's most
        recent output. Useful for computing inter-agent agreement.
        """
        result: dict[str, dict[str, Any]] = {}
        for output in self.history:
            result[output.agent_name] = output.labels
        return result

    def all_agents_agree(self, field_names: list[str] | None = None) -> bool:
        """
        Return True if all agents in the latest round agree on all
        (or the specified subset of) fields.
        """
        latest = self.latest_labels_by_agent()
        if len(latest) < 2:
            return False

        agents = list(latest.keys())
        ref = latest[agents[0]]

        for agent in agents[1:]:
            for fname, val in latest[agent].items():
                if field_names and fname not in field_names:
                    continue
                if val != ref.get(fname):
                    return False
        return True

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "item_data": self.item_data,
            "current_labels": self.current_labels,
            "history": [o.to_dict() for o in self.history],
            "cycle": self.cycle,
            "converged": self.converged,
            "originator_name": self.originator_name,
            "originator_labels": self.originator_labels,
            "metadata": self.metadata,
        }