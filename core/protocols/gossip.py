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

Phase 2 (if Peer Review enabled):
    For each agent:
        peer review LLM call  → RunEvent("peer_review")
    ECU update               → RunEvent("ecu_update")
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from core.agent import Agent
from core.providers import get_provider
from core.hub import CommunicationHub
from core.protocols import RunEvent, build_ecu_info_str
from core.state import AgentOutput


class GossipProtocol:
    """
    Sequential deliberation protocol.

    Agents contribute one by one; later agents see earlier agents' same-round
    contributions before writing their own. Studies origination bias and sequential anchoring.
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
        self.custom_order: list[str] = protocol.get("custom_order", [])

    def run_iter(self, hub: CommunicationHub) -> Iterator[RunEvent]:
        ordered_agents = self._build_agent_order()

        for cycle_idx in range(self.max_cycles):
            # Snapshot the official pre-round state.

            if self.order_type == "Randomized each cycle" and cycle_idx > 0:
                ordered_agents = self._build_agent_order(randomize=True)

            # ── Phase 1: sequential contributions ────────────────────────
            # Each packet is built just before the agent's turn, so later
            # agents see earlier agents' same-round contributions.
            round_outputs: list[AgentOutput] = []

            for agent in ordered_agents:
                packet = hub.build_context(agent.name, cycle_idx)
                system_prompt, user_message = agent.build_prompt(packet)
                orig_system_prompt, orig_user_message = system_prompt, user_message
                injected = yield RunEvent(
                    kind="dispatch",
                    cycle=cycle_idx,
                    agent_name=agent.name,
                    packet=packet,
                    prompt={"system": system_prompt, "user": user_message},
                )
                # A user that resumes with .send({...}) instead of next()
                # can override the prompt text and/or supply a human-typed
                # contribution before this turn is dispatched. 
                human_input = None
                if isinstance(injected, dict):
                    system_prompt = injected.get("system", system_prompt)
                    user_message = injected.get("user", user_message)
                    human_input = injected.get("human_input")

                if self.dry_run or human_input is not None:
                    output = agent.send(
                        packet, system_prompt, user_message,
                        dry_run=self.dry_run, human_input=human_input,
                    )
                else:
                    # Stream the LLM call chunk by chunk so it appears as a live typing effect.
                    for chunk in agent.stream(packet, system_prompt, user_message):
                        yield RunEvent(
                            kind="stream_chunk",
                            cycle=cycle_idx,
                            agent_name=agent.name,
                            packet=packet,
                            chunk=chunk,
                        )
                    output = agent.finalize_stream(packet)
                hub.submit(output)
                round_outputs.append(output)
                edited = (system_prompt, user_message) != (orig_system_prompt, orig_user_message)
                hub.log_prompt(
                    cycle_idx, agent.name, "contribution",
                    prompt=f"[SYSTEM]\n{agent._last_system_prompt}\n\n[USER]\n{agent._last_user_message}",
                    response=output.raw_response or "",
                    original_prompt=(
                        f"[SYSTEM]\n{orig_system_prompt}\n\n[USER]\n{orig_user_message}" if edited else None
                    ),
                )
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
            # Reviewing is deferred for human agents (see Agent.is_human) —
            # they're still fully reviewable by everyone else since
            # all_contributions treats their text like any other agent's.
            if agent.is_human:
                continue
            ecu_info = build_ecu_info_str(hub, cycle_idx, agent.name)
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
                    agent_histories=agent_histories,
                    collect_importance_votes=collect_votes,
                )
                try:
                    pr_kwargs = {"json_mode": True} if agent.provider == "Google" else {}
                    raw = get_provider(agent.provider).complete(
                        model=agent.model,
                        system_prompt="You are a helpful assistant evaluating contributions in a deliberation experiment. Read the evaluation instructions carefully and respond with the requested JSON.",
                        user_message=prompt,
                        max_tokens=800,
                        temperature=agent.temperature,
                        **pr_kwargs,
                    )
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


    def _build_agent_order(self, randomize: bool = False) -> list[Agent]:
        agents = list(self.agents)

        if self.order_type == "Custom order" and self.custom_order:
            by_name = {a.name: a for a in agents}
            ordered = [by_name[name] for name in self.custom_order if name in by_name]
            # Any agent not named in custom_order (e.g. added after the order
            # was set) is appended at the end rather than silently dropped.
            remaining = [a for a in agents if a.name not in set(self.custom_order)]
            return ordered + remaining

        if self.initializer_name:
            names = [a.name for a in agents]
            if self.initializer_name in names:
                idx = names.index(self.initializer_name)
                agents = agents[idx:] + agents[:idx]
        if randomize:
            random.shuffle(agents)
        return agents