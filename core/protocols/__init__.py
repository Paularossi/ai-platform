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