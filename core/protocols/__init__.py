# core/protocols/__init__.py
"""
Shared protocol primitives.

RunEvent is defined here so every protocol (Gossip, Crowd, Duel, Court, ...)
can emit the same event type and the UI render loop in 6_Run.py needs no changes
when a new protocol is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.hub import ContextPacket
from core.state import AgentOutput


def build_ecu_info_str(hub, cycle: int, agent_name: str) -> str:
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

    kind == "dispatch"     : hub built a context packet, about to call agent
    kind == "submission"   : agent returned output, hub stored it
    kind == "aggregate"    : crowd round aggregate computed
    kind == "peer_review"  : one agent's Phase 2 peer review completed
    kind == "ecu_update"   : ECU ledger updated after a full peer review round
    """
    kind: Literal["dispatch", "submission", "aggregate", "peer_review", "ecu_update"]
    cycle: int
    agent_name: str
    packet: ContextPacket
    output: AgentOutput | None = None