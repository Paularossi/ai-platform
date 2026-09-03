"""
core/state.py

Defines two output dataclasses:

  AgentOutput      — one agent's contribution on one turn (Phase 1)
  PeerReviewOutput — one agent's peer review table (Phase 2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


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
    contribution : str
        The agent's free-text contribution for this round.
    changed : dict[str, bool]
        Change flags vs previous state (set by hub).
    ecu_scores : dict[str, float]
        Quality scores received from peer review (set after Phase 2).
    ecu_earned : float | None
        ECUs earned this turn (set after Phase 2).
    raw_response : str | None
    timestamp : str
    """
    agent_name: str
    cycle: int
    contribution: str | None = None
    changed: dict[str, bool] = field(default_factory=dict)
    ecu_scores: dict[str, float] = field(default_factory=dict)
    ecu_earned: float | None = None
    raw_response: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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
        Derived from the consensus dimension; used by CoalitionTracker.
    justifications : dict[str, str]
        {agent_name: one-sentence justification from the peer review}
    importance_votes : dict[str, float]
        {dimension: points} — this agent's importance vote.
        Points sum to 100. Collected during Phase 2, used by the Orchestrator.
        Empty dict when orchestrator is disabled or vote parsing failed.
    raw_response : str | None
    timestamp : str
    """
    reviewer_name: str
    cycle: int
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    self_scores: dict[str, float] | None = None
    coalition_scores: dict[str, float] = field(default_factory=dict)
    justifications: dict[str, str] = field(default_factory=dict)
    importance_votes: dict[str, float] = field(default_factory=dict)
    raw_response: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "reviewer_name": self.reviewer_name,
            "cycle": self.cycle,
            "scores": self.scores,
            "self_scores": self.self_scores,
            "coalition_scores": self.coalition_scores,
            "justifications": self.justifications,
            "importance_votes": self.importance_votes,
            "raw_response": self.raw_response,
            "timestamp": self.timestamp,
        }
