"""
core/protocols/gossip.py  (Sequential protocol)

Agents contribute one by one within each round. Each agent's context packet
is built just before their turn, so they see all prior contributions from
the same round (agents earlier in the order). This creates within-round
information asymmetry — the first agent has no context; the last agent
has seen everyone else.

This is the protocol to use when studying origination bias and sequential
anchoring effects.

Round structure
---------------
Phase 1:
    For each agent in order:
        build context (sees prior same-round submissions)
        dispatch  → RunEvent("dispatch")
        LLM call  → RunEvent("submission")

Phase 2 (if peer_reviewer configured):
    For each agent:
        peer review LLM call  → RunEvent("peer_review")
    ECU update               → RunEvent("ecu_update")
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from core.agent import Agent
from core.hub import CommunicationHub
from core.protocols import RunEvent, run_peer_review_for_agent
from core.state import AgentOutput


class GossipProtocol:
    """
    Sequential deliberation protocol.

    Agents contribute one by one; later agents see earlier agents' same-round
    contributions before writing their own. Studies origination bias and
    sequential anchoring.
    """

    def __init__(
        self,
        agents: list[Agent],
        experiment_config: dict[str, Any],
        peer_reviewer: Any | None = None,
        coalition_tracker: Any | None = None,
        orchestrator: Any | None = None,
        dry_run: bool = False,
    ):
        self.agents = agents
        self.peer_reviewer = peer_reviewer
        self.coalition_tracker = coalition_tracker
        self.orchestrator = orchestrator
        self.dry_run = dry_run

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
            # Snapshot the official pre-round state. Counterfactual orchestrator
            # evaluations clone this snapshot so sandbox outputs never enter the
            # official dialogue history.

            if self.order_type == "Randomized each cycle" and cycle_idx > 0:
                ordered_agents = self._build_agent_order(randomize=True)

            # ── Phase 1: sequential contributions ────────────────────────
            # Each packet is built just before the agent's turn, so later
            # agents see earlier agents' same-round contributions.
            round_outputs: list[AgentOutput] = []

            for agent in ordered_agents:
                packet = hub.build_context(agent.name, cycle_idx)
                yield RunEvent(
                    kind="dispatch",
                    cycle=cycle_idx,
                    agent_name=agent.name,
                    packet=packet,
                )
                output = agent.call(packet, dry_run=self.dry_run)
                hub.submit(output)
                round_outputs.append(output)
                hub.log_prompt(cycle_idx, agent.name, "contribution",
                               prompt=f"[SYSTEM]\n{agent._last_system_prompt}\n\n[USER]\n{agent._last_user_message}",
                               response=output.raw_response or "")
                yield RunEvent(
                    kind="submission",
                    cycle=cycle_idx,
                    agent_name=agent.name,
                    packet=packet,
                    output=output,
                )

            # ── Phase 2: peer review ──────────────────────────────────────
            if self.peer_reviewer:
                yield from self._run_peer_review(
                    hub, cycle_idx, round_outputs, ordered_agents
                )

            # ── Stopping rule ─────────────────────────────────────────────
            if self.stopping_rule in ("Convergence", "Either"):
                if hub.check_convergence():
                    return

        hub.check_convergence()

    def _run_peer_review(
        self,
        hub: CommunicationHub,
        cycle_idx: int,
        round_outputs: list[AgentOutput],
        ordered_agents: list[Agent],
    ) -> Iterator[RunEvent]:

        all_contributions = {o.agent_name: o.contribution for o in round_outputs}
        if not all_contributions:
            return

        item_context = ", ".join(
            f"{k}: {v}" for k, v in hub.item_data.items()

        )
        ref_packet = hub.build_context(ordered_agents[0].name, cycle_idx)
        collect_votes = self.orchestrator is not None and self.orchestrator.enabled

        # Build review history once — same for all reviewers in a round
        agent_histories = hub.build_review_history(cycle_idx)

        for agent in ordered_agents:
            review = run_peer_review_for_agent(
                agent, hub, cycle_idx, self.peer_reviewer, all_contributions,
                item_context, agent_histories, collect_votes, self.peer_reviewer.dry_run,
            )
            hub.submit_peer_review(review)
            yield RunEvent(
                kind="peer_review",
                cycle=cycle_idx,
                agent_name=agent.name,
                packet=ref_packet,
            )

        round_reviews = [r for r in hub.peer_review_log if r.cycle == cycle_idx]
        if self.coalition_tracker:
            self.coalition_tracker.find_coalition(round_reviews)

        if hub.ledger:
            hub.compute_ecus_for_round(cycle_idx)
            hub.ledger.record_social_welfare(cycle_idx, round_reviews)

            if self.orchestrator and self.orchestrator.should_update(
                cycle_idx, is_final_cycle=cycle_idx >= self.max_cycles - 1
            ):
                importance_votes = {
                    r.reviewer_name: r.importance_votes
                    for r in round_reviews
                    if r.importance_votes
                }
                self.orchestrator.update(
                    cycle=cycle_idx,
                    ledger=hub.ledger,
                    importance_votes=importance_votes,
                )

            yield RunEvent(
                kind="ecu_update",
                cycle=cycle_idx,
                agent_name="__all__",
                packet=ref_packet,
            )


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