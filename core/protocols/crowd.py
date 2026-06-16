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

import copy
from collections.abc import Iterator
from typing import Any

from core.agent import Agent, get_openai_client
from core.hub import CommunicationHub, ContextPacket
from core.protocols import RunEvent
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
        self._openai_client_cache = None

        protocol = experiment_config.get("protocol", {})
        self.max_cycles: int = int(protocol.get("max_cycles", 5))
        self.stopping_rule: str = protocol.get("stopping_rule", "Either")


    def run(self, hub: CommunicationHub) -> CommunicationHub:
        for _ in self.run_iter(hub):
            pass
        return hub

    def run_iter(self, hub: CommunicationHub) -> Iterator[RunEvent]:

        for cycle_idx in range(self.max_cycles):

            # Snapshot the official pre-round state. Counterfactual orchestrator
            # evaluations clone this snapshot so sandbox outputs never enter the
            # official dialogue history.

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
                # Log the prompt for debugging
                from core.agent import _build_system_prompt, _build_user_message
                sys_p = _build_system_prompt(agent.name, agent.role,
                    agent._instructions.get("base_instructions",""),
                    agent._instructions.get("guideline_notes",""),
                    agent._overrides, agent._questions)
                usr_p = _build_user_message(packet, agent._questions)
                hub.log_prompt(cycle_idx, agent.name, "contribution",
                               prompt=f"[SYSTEM]\n{sys_p}\n\n[USER]\n{usr_p}",
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
        item_context = ", ".join(
            f"{k}: {v}" for k, v in hub.item_data.items()

        )
        # Collect importance votes when orchestrator is enabled
        collect_votes = self.orchestrator is not None and self.orchestrator.enabled

        for agent in self.agents:
            ecu_info = hub.ecu_info_str(cycle_idx, calling_agent=agent.name)
            if self.peer_reviewer.dry_run:
                review = self.peer_reviewer.parse(
                    reviewer_name=agent.name,
                    cycle=cycle_idx,
                    raw="[dry-run]",
                    all_contributions=all_contributions,
                )
            else:
                prompt = self.peer_reviewer.build_prompt(
                    reviewer_name=agent.name,
                    reviewer_contribution=all_contributions.get(agent.name, ""),
                    all_contributions=all_contributions,
                    cycle=cycle_idx,
                    item_context=item_context,
                    ecu_info=ecu_info,
                    reviewer_role=agent.role,
                    agent_histories=hub.build_review_history(cycle_idx),
                    collect_importance_votes=collect_votes,
                )
                try:
                    response = get_openai_client().chat.completions.create(
                        model=agent.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=800,
                    )
                    raw = response.choices[0].message.content or "{}"
                except Exception as exc:
                    raw = f"[ERROR: {exc}]"
                hub.log_prompt(cycle_idx, agent.name, "peer_review", prompt=prompt, response=raw)
                review = self.peer_reviewer.parse(
                    reviewer_name=agent.name,
                    cycle=cycle_idx,
                    raw=raw,
                    all_contributions=all_contributions,
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