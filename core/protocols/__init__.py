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

    kind == "dispatch"   : hub has built a context packet and is about to
                           call the agent. packet is set, output is None.
    kind == "submission" : agent has returned its output and the hub has stored it.
                           output is set, packet is the one that was sent.
    """
    kind: Literal["dispatch", "submission", "aggregate"]
    cycle: int
    agent_name: str
    packet: ContextPacket
    output: AgentOutput | None = None
