"""
core/protocols/gossip.py

Gossip (sequential) protocol - hub-mediated.

The protocol is now a thin outer loop. All message passing, visibility
filtering, and convergence tracking live in the CommunicationHub.

Flow for one item
-----------------
For each cycle (up to max_cycles):
    For each agent in order:
        1. Hub builds a ContextPacket for this agent  → yields RunEvent("dispatch")
        2. Protocol dispatches the packet to the agent
        3. Agent calls the LLM, returns AgentOutput
        4. Protocol submits the output back to the hub → yields RunEvent("submission")
        5. Hub stores it, updates current labels, computes changed flags
    Check stopping rule → break if met

Usage
-----
    from core.hub import CommunicationHub, ContextPacket
    from core.agent import Agent
    from core.protocols.gossip import GossipProtocol, RunEvent

    hub = CommunicationHub(...)
    protocol = GossipProtocol(agents, experiment_config)

    for event in protocol.run_iter(hub):
        if event.kind == "dispatch":
            print(f"Hub → {event.agent_name}: {event.packet.visible_history}")
        elif event.kind == "submission":
            print(f"{event.agent_name} → Hub: {event.output.labels}")
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from core.agent import Agent
from core.hub import CommunicationHub, ContextPacket
from core.state import AgentOutput


# ---------------------------------------------------------------------------
# RunEvent - emitted by run_iter for each step
# ---------------------------------------------------------------------------

@dataclass
class RunEvent:
    """
    A single step event emitted by the protocol generator.

    kind == "dispatch"   : hub has built a context packet and is about to
                           call the agent. packet is populated, output is None.
    kind == "submission" : agent has returned its output and hub has stored it.
                           output is populated, packet is the one that was sent.
    """
    kind: Literal["dispatch", "submission"]
    cycle: int
    agent_name: str
    packet: ContextPacket
    output: AgentOutput | None = None


class GossipProtocol:
    """
    Sequential gossip protocol mediated by a CommunicationHub.

    Parameters
    ----------
    agents : list[Agent]
        Ordered list of agents.
    experiment_config : dict
        Full experiment config - used to read protocol settings.
    dry_run : bool
        If True, all agent LLM calls are skipped (returns placeholders).
        Useful for UI testing without an API key.
    """

    def __init__(
        self,
        agents: list[Agent],
        experiment_config: dict[str, Any],
        dry_run: bool = False,
    ):
        self.agents = agents
        self.dry_run = dry_run

        protocol = experiment_config.get("protocol", {})
        self.max_cycles: int = int(protocol.get("max_cycles", 5))
        self.stopping_rule: str = protocol.get("stopping_rule", "Either")
        self.order_type: str = protocol.get("order_type", "Fixed")
        self.initializer_name: str | None = protocol.get("initializer_agent")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, hub: CommunicationHub) -> CommunicationHub:
        """Run the full protocol to completion and return the hub."""
        for _ in self.run_iter(hub):
            pass
        return hub

    def run_iter(self, hub: CommunicationHub) -> Iterator[RunEvent]:
        """
        Generator - yields a RunEvent for every step:
          - "dispatch"   before the LLM call (shows what hub sent to the agent)
          - "submission" after the LLM call  (shows what the agent returned)
        """
        ordered_agents = self._build_agent_order()

        for cycle_idx in range(self.max_cycles):

            if self.order_type == "Randomized each cycle" and cycle_idx > 0:
                ordered_agents = self._build_agent_order(randomize=True)

            for agent in ordered_agents:
                # Step 1 - hub builds context packet
                packet = hub.build_context(
                    agent_name=agent.name,
                    cycle=cycle_idx,
                )

                # Yield BEFORE the LLM call so UI can show "hub dispatched to agent"
                yield RunEvent(
                    kind="dispatch",
                    cycle=cycle_idx,
                    agent_name=agent.name,
                    packet=packet,
                )

                # Step 2 - agent calls LLM
                output = agent.call(packet, dry_run=self.dry_run)

                # Step 3 - submit back to hub
                hub.submit(output)

                # Yield AFTER submission so UI can show agent's output
                yield RunEvent(
                    kind="submission",
                    cycle=cycle_idx,
                    agent_name=agent.name,
                    packet=packet,
                    output=output,
                )

            # Check stopping rule after each full cycle
            if self.stopping_rule in ("Convergence", "Either"):
                if hub.check_convergence():
                    return

        hub.check_convergence()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_agent_order(self, randomize: bool = False) -> list[Agent]:
        agents = list(self.agents)

        if self.initializer_name:
            names = [a.name for a in agents]
            if self.initializer_name in names:
                idx = names.index(self.initializer_name)
                agents = agents[idx:] + agents[:idx]

        if randomize:
            random.shuffle(agents)

        return agents