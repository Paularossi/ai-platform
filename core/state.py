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
