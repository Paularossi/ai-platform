"""
core/protocols/gossip.py

Gossip (sequential) protocol with Phase 2 peer review.

Round structure
---------------
Phase 1 (Contribution):
  Agents take turns sequentially, each seeing the previous agent's output.
  Yields dispatch + submission RunEvents as before.

Phase 2 (Peer review):
  After all agents have contributed, each agent receives all contributions
  and fills in the peer review table.
  Yields one peer_review RunEvent per agent reviewer.
  After all reviews are collected, ECUs are computed and an ecu_update
  event is emitted.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from core.agent import Agent
from core.hub import CommunicationHub
from core.protocols import RunEvent
from core.state import AgentOutput


class GossipProtocol:

    def __init__(
        self,
        agents: list[Agent],
        experiment_config: dict[str, Any],
        peer_reviewer: Any | None = None,
        coalition_tracker: Any | None = None,
        dry_run: bool = False,
    ):
        self.agents = agents
        self.dry_run = dry_run
        self.peer_reviewer = peer_reviewer
        self.coalition_tracker = coalition_tracker

        protocol = experiment_config.get("protocol", {})
        self.max_cycles: int = int(protocol.get("max_cycles", 5))
        self.stopping_rule: str = protocol.get("stopping_rule", "Either")
        self.order_type: str = protocol.get("order_type", "Fixed")
        self.initializer_name: str | None = protocol.get("initializer_agent")

    def run(self, hub: CommunicationHub) -> CommunicationHub:
        for _ in self.run_iter(hub):
            pass
        return hub

    def run_iter(self, hub: CommunicationHub) -> Iterator[RunEvent]:
        ordered_agents = self._build_agent_order()

        for cycle_idx in range(self.max_cycles):

            if self.order_type == "Randomized each cycle" and cycle_idx > 0:
                ordered_agents = self._build_agent_order(randomize=True)

            # ── Phase 1: contributions ───────────────────────────────────
            for agent in ordered_agents:
                packet = hub.build_context(agent_name=agent.name, cycle=cycle_idx)
                yield RunEvent(kind="dispatch", cycle=cycle_idx,
                               agent_name=agent.name, packet=packet)
                output = agent.call(packet, dry_run=self.dry_run)
                hub.submit(output)
                yield RunEvent(kind="submission", cycle=cycle_idx,
                               agent_name=agent.name, packet=packet, output=output)

            # ── Phase 2: peer review ─────────────────────────────────────
            yield from self._run_peer_review(hub, cycle_idx, ordered_agents)

            # ── Stopping rule ────────────────────────────────────────────
            if self.stopping_rule in ("Convergence", "Either"):
                if hub.check_convergence():
                    return

        hub.check_convergence()

    def _run_peer_review(
        self,
        hub: CommunicationHub,
        cycle_idx: int,
        ordered_agents: list[Agent],
    ) -> Iterator[RunEvent]:
        if not self.peer_reviewer:
            return

        all_contributions = {
            o.agent_name: o.contribution
            for o in hub.log if o.cycle == cycle_idx
        }
        if not all_contributions:
            return

        item_context = ", ".join(
            f"{k}: {v}" for k, v in hub.item_data.items()
            if k.lower() not in ("image", "image_url", "image_path")
        )
        ecu_info = hub.ecu_info_str(cycle_idx)

        for agent in ordered_agents:
            if self.peer_reviewer.dry_run:
                # Dry run: parse without an LLM call
                review = self.peer_reviewer.parse(
                    reviewer_name=agent.name,
                    cycle=cycle_idx,
                    raw="[dry-run]",
                    all_contributions=all_contributions,
                )
            else:
                # Real run: build prompt and call the agent's own LLM
                prompt = self.peer_reviewer.build_prompt(
                    reviewer_name=agent.name,
                    reviewer_contribution=all_contributions.get(agent.name, ""),
                    all_contributions=all_contributions,
                    cycle=cycle_idx,
                    item_context=item_context,
                    ecu_info=ecu_info,
                )
                # Call agent via a lightweight direct OpenAI call
                # (peer review is a structured task, not a full Agent.call())
                try:
                    from openai import OpenAI
                    client = OpenAI()
                    response = client.chat.completions.create(
                        model=agent.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=800,
                    )
                    raw = response.choices[0].message.content or "{}"
                except Exception as exc:
                    raw = f"[ERROR: {exc}]"

                review = self.peer_reviewer.parse(
                    reviewer_name=agent.name,
                    cycle=cycle_idx,
                    raw=raw,
                    all_contributions=all_contributions,
                )

            hub.submit_peer_review(review)
            last_packet = hub.build_context(agent_name=agent.name, cycle=cycle_idx)
            yield RunEvent(
                kind="peer_review",
                cycle=cycle_idx,
                agent_name=agent.name,
                packet=last_packet,
            )

        # Coalition tracking must run before the ECU update is rendered, so the UI
        # shows the coalition for the current cycle rather than the previous one.
        full_coalition_reached = False
        if self.coalition_tracker:
            round_reviews = [r for r in hub.peer_review_log if r.cycle == cycle_idx]
            coalition = self.coalition_tracker.find_coalition(round_reviews)
            full_coalition_reached = len(coalition) == len(hub.agent_names)
            if full_coalition_reached:
                hub.converged = True

        # Compute ECUs after all reviews are in
        if hub.ledger:
            hub.compute_ecus_for_round(cycle_idx)
            last_packet = hub.build_context(
                agent_name=ordered_agents[0].name, cycle=cycle_idx
            )
            yield RunEvent(
                kind="ecu_update",
                cycle=cycle_idx,
                agent_name="__all__",
                packet=last_packet,
            )

        if full_coalition_reached:
            return

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