"""
core/protocols/crowd.py  (Simultaneous protocol)

All agents contribute independently in Phase 1 — none see each other's
answers within the same round. Packets are built upfront from a snapshot
of the hub state, guaranteeing blindness within a round.

After all agents contribute, Phase 2 runs peer review and ECU scoring.
In the next round, each agent sees the previous round's contributions
(according to φ₁ visibility setting).

Round structure
---------------
Phase 1:
    For each agent (packets built upfront — all blind):
        dispatch  → RunEvent("dispatch")
        LLM call  → RunEvent("submission")

Phase 2 (if peer_reviewer configured):
    For each agent:
        peer review LLM call  → RunEvent("peer_review")
    ECU update               → RunEvent("ecu_update")
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from core.agent import Agent
from core.hub import CommunicationHub, ContextPacket
from core.protocols import RunEvent, finalize_peer_review_round, run_peer_review_for_agent
from core.state import AgentOutput


class CrowdProtocol:
    """
    Simultaneous deliberation protocol.

    All agents contribute blindly in Phase 1 (no within-round cross-visibility),
    then score each other in Phase 2 (peer review).
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


    def run(self, hub: CommunicationHub) -> CommunicationHub:
        for _ in self.run_iter(hub):
            pass
        return hub

    def run_iter(self, hub: CommunicationHub) -> Iterator[RunEvent]:

        for cycle_idx in range(self.max_cycles):

            # ── Phase 1: simultaneous blind contributions ─────────────────
            # Build all packets from a pre-round snapshot so no agent sees
            # another's current-round contribution during Phase 1.
            packets: list[tuple[Agent, ContextPacket]] = [
                (agent, hub.build_context(agent.name, cycle_idx))
                for agent in self.agents
            ]

            round_outputs: list[AgentOutput] = []

            for agent, packet in packets:
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
                    hub, cycle_idx, round_outputs, packets[0][1]
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
        ref_packet: ContextPacket,
    ) -> Iterator[RunEvent]:

        all_contributions = {o.agent_name: o.contribution for o in round_outputs}
        item_context = ", ".join(f"{k}: {v}" for k, v in hub.item_data.items())
        # Collect importance votes when orchestrator is enabled
        collect_votes = self.orchestrator is not None and self.orchestrator.enabled

        # Build review history once — same for all reviewers in a round
        agent_histories = hub.build_review_history(cycle_idx)

        for agent in self.agents:
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

        finalize_peer_review_round(
            hub, cycle_idx, self.coalition_tracker, self.orchestrator,
            is_final_cycle=cycle_idx >= self.max_cycles - 1,
        )
        if hub.ledger:
            yield RunEvent(
                kind="ecu_update",
                cycle=cycle_idx,
                agent_name="__all__",
                packet=ref_packet,
            )