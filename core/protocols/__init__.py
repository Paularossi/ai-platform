# core/protocols/__init__.py
"""
Shared protocol primitives.

RunEvent is defined here so every protocol (Gossip, Crowd, ...) can emit the
same event type and the UI render loop in 5_Run.py needs no changes when a
new protocol is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.hub import ContextPacket
from core.state import AgentOutput


def build_ecu_info_str(hub, agent_name: str) -> str:
    """
    Build the ECU context string for the peer review prompt.
    Returns an empty string under the opaque condition.
    """
    if not hub.ledger or hub.ecu_info_condition == "opaque":
        return ""

    balances = hub.ledger.balances
    weights = hub.ledger.ecu_weights

    if hub.ecu_info_condition == "transparent":
        b_str = ", ".join(f"{k}: {v:.3f}" for k, v in balances.items())
        w_str = ", ".join(f"{k}={v:.2f}" for k, v in weights.items())
        return f"ECU balances: {b_str}. Dimension weights: {w_str}."

    if hub.ecu_info_condition == "semi-transparent":
        import random
        noisy = {k: round(v * random.uniform(0.8, 1.2), 2) for k, v in weights.items()}
        w_str = ", ".join(f"{k}≈{v}" for k, v in noisy.items())
        own = balances.get(agent_name, 0.0)
        return f"Your ECU balance: {own:.3f}. Approximate dimension weights: {w_str}."

    return ""


@dataclass
class RunEvent:
    """
    A single step event emitted by any protocol's run_iter() generator.

    kind == "dispatch"      : prompt built, about to call agent — the user
                              may resume this generator with .send(injected)
                              instead of next() to edit the turn before it
                              fires; see `prompt` and the run_iter() docstrings
                              in gossip.py / crowd.py for the injected shape.
    kind == "stream_chunk"  : one piece of an agent's response, as it's
                              generated — zero or more of these are emitted
                              between an agent's "dispatch" and "submission"
                              for a real LLM call, so it appears as a live typing effect. d
    kind == "submission"   : agent returned output, hub stored it
    kind == "aggregate"    : crowd round aggregate computed
    kind == "peer_review"  : one agent's Phase 2 peer review completed
    kind == "ecu_update"   : ECU ledger updated after a full peer review round

    prompt : dict | None
        Only set on "dispatch" events: {"system": str, "user": str} — the
        exact text about to be sent to the LLM. Not derivable from `packet`
        alone (packet is structured context; this is the rendered prompt).

    chunk : str | None
        Only set on "stream_chunk" events: the next piece of text.
    """
    kind: Literal["dispatch", "stream_chunk", "submission", "aggregate", "peer_review", "ecu_update"]
    cycle: int
    agent_name: str
    packet: ContextPacket
    output: AgentOutput | None = None
    prompt: dict[str, str] | None = None
    chunk: str | None = None