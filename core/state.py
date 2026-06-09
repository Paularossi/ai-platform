"""
core/state.py

Defines two output dataclasses:

  AgentOutput      — one agent's contribution on one turn (Phase 1)
  PeerReviewOutput — one agent's peer review table (Phase 2)

And ExperimentState — the serialised result for a completed item run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Phase 1: contribution output
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    """
    Structured output from one agent on one contribution turn (Phase 1).

    Fields
    ------
    agent_name : str
    cycle : int
        0-indexed round number.
    contribution : Any
        str for deliberation tasks, dict[field→verdict] for classification.
    changed : dict[str, bool]
        Per-field change flags vs previous state (set by hub).
    ecu_scores : dict[str, float]
        Quality scores received from peer review (set after Phase 2).
    ecu_earned : float | None
        ECUs earned this turn (set after Phase 2).
    raw_response : str | None
    timestamp : str
    """
    agent_name: str
    cycle: int
    contribution: Any = None
    changed: dict[str, bool] = field(default_factory=dict)
    ecu_scores: dict[str, float] = field(default_factory=dict)
    ecu_earned: float | None = None
    raw_response: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def is_classification(self) -> bool:
        return isinstance(self.contribution, dict)

    @property
    def labels(self) -> dict[str, Any]:
        return self.contribution if isinstance(self.contribution, dict) else {}

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "cycle": self.cycle,
            "contribution": self.contribution,
            "changed": self.changed,
            "ecu_scores": self.ecu_scores,
            "ecu_earned": self.ecu_earned,
            "raw_response": self.raw_response,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Phase 2: peer review output
# ---------------------------------------------------------------------------

@dataclass
class PeerReviewOutput:
    """
    Structured output from one agent's peer review of all other agents (Phase 2).

    Produced once per agent per round, after all Phase 1 contributions
    are revealed.

    Fields
    ------
    reviewer_name : str
        The agent doing the reviewing.
    cycle : int
    scores : dict[str, dict[str, float]]
        {reviewed_agent_name: {dimension: score}}
        Scores given by this reviewer to each other agent.
        Excludes self unless include_self_assessment=True.
    self_scores : dict[str, float] | None
        Self-assessment scores {dimension: score}.
        None when self-assessment is disabled.
    coalition_scores : dict[str, float]
        {agent_name: consensus_score in [0, 1]}
        Backward-compatible field name. Coalitions are derived from mutual consensus scores.
    review_justifications : dict[str, str]
        {agent_name: brief explanation of the peer review}
    raw_response : str | None
    timestamp : str
    """
    reviewer_name: str
    cycle: int
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    self_scores: dict[str, float] | None = None
    coalition_scores: dict[str, float] = field(default_factory=dict)
    review_justifications: dict[str, str] = field(default_factory=dict)
    coalition_justifications: dict[str, str] = field(default_factory=dict)  # deprecated alias
    raw_response: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "reviewer_name": self.reviewer_name,
            "cycle": self.cycle,
            "scores": self.scores,
            "self_scores": self.self_scores,
            "coalition_scores": self.coalition_scores,
            "review_justifications": self.review_justifications,
            "coalition_justifications": self.coalition_justifications,
            "raw_response": self.raw_response,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Serialised experiment state (one completed item run)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentState:
    """Serialised result for a single item after a protocol run completes."""
    item_id: str
    item_data: dict[str, Any] = field(default_factory=dict)
    current_contribution: Any = None
    history: list[AgentOutput] = field(default_factory=list)
    peer_review_history: list[PeerReviewOutput] = field(default_factory=list)
    cycle: int = 0
    converged: bool = False
    originator_name: str | None = None
    originator_contribution: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "item_data": self.item_data,
            "current_contribution": self.current_contribution,
            "history": [o.to_dict() for o in self.history],
            "peer_review_history": [p.to_dict() for p in self.peer_review_history],
            "cycle": self.cycle,
            "converged": self.converged,
            "originator_name": self.originator_name,
            "originator_contribution": self.originator_contribution,
            "metadata": self.metadata,
        }