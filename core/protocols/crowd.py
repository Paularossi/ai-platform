"""
core/protocols/crowd.py

Crowd (simultaneous) protocol - hub-mediated.

All agents classify independently each round - none see each other's
answers within the same round. After all agents submit, the hub aggregates
by majority vote and the result becomes the shared state for the next round.

Flow for one item
-----------------
For each round (up to max_cycles):
    1. Snapshot hub state → build context packets for ALL agents upfront
    2. For each agent (order doesn't matter):
       a. Dispatch pre-built packet  → yields RunEvent("dispatch")
       b. Agent calls LLM            → yields RunEvent("submission")
       c. Submit output to hub log
    3. Aggregate all outputs by majority vote per field
    4. Push aggregated labels back to hub via set_current_labels()
    5. Yield RunEvent("aggregate") showing the round result + vote counts
    6. Check stopping rule → break if met
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from typing import Any

from core.agent import Agent
from core.hub import CommunicationHub, ContextPacket
from core.protocols import RunEvent
from core.state import AgentOutput


class CrowdProtocol:
    """
    Simultaneous crowd protocol mediated by a CommunicationHub.

    Parameters
    ----------
    agents : list[Agent]
        The agent roster - all treated as peers, no ordering required.
    experiment_config : dict
        Full experiment config - used to read protocol settings.
    dry_run : bool
        If True, all agent LLM calls are skipped (returns placeholders).
    """

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
        Generator - yields RunEvents for every step.

        Per round:
          - "dispatch"   x N  (one per agent, before LLM call)
          - "submission" x N  (one per agent, after LLM call)
          - "aggregate"  x 1  (after all agents, showing the round result)
        """
        for cycle_idx in range(self.max_cycles):

            # Step 1: snapshot hub state for all agents before any submission.
            # Building all packets upfront ensures no agent sees another's
            # current-round answer, even though we call them sequentially.
            packets: list[tuple[Agent, ContextPacket]] = [
                (agent, hub.build_context(agent.name, cycle_idx))
                for agent in self.agents
            ]

            round_outputs: list[AgentOutput] = []

            # Step 2: dispatch and collect
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

                yield RunEvent(
                    kind="submission",
                    cycle=cycle_idx,
                    agent_name=agent.name,
                    packet=packet,
                    output=output,
                )

            # Step 3: aggregate (classification only) or just record round end
            is_classification = any(
                isinstance(o.contribution, dict) for o in round_outputs if o.contribution
            )
            if is_classification:
                aggregate_labels, vote_distribution = _majority_vote(
                    round_outputs, hub.current_labels
                )
                hub.set_current_labels(aggregate_labels)
                aggregate_output = AgentOutput(
                    agent_name="__aggregate__",
                    cycle=cycle_idx,
                    contribution=aggregate_labels,
                    prob_distribution=vote_distribution,
                )
                yield RunEvent(
                    kind="aggregate",
                    cycle=cycle_idx,
                    agent_name="__aggregate__",
                    packet=packets[0][1],
                    output=aggregate_output,
                )

            # ── Phase 2: peer review (if configured) ────────────────────
            if self.peer_reviewer:
                all_contributions = {
                    o.agent_name: o.contribution
                    for o in round_outputs
                }
                item_context = ", ".join(
                    f"{k}: {v}" for k, v in hub.item_data.items()
                    if k.lower() not in ("image", "image_url", "image_path")
                )
                ecu_info = hub.ecu_info_str(cycle_idx)

                for agent in self.agents:
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
                        )
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
                    yield RunEvent(
                        kind="peer_review",
                        cycle=cycle_idx,
                        agent_name=agent.name,
                        packet=packets[0][1],
                    )

                if hub.ledger:
                    hub.compute_ecus_for_round(cycle_idx)
                    yield RunEvent(
                        kind="ecu_update",
                        cycle=cycle_idx,
                        agent_name="__all__",
                        packet=packets[0][1],
                    )

                if self.coalition_tracker:
                    round_reviews = [r for r in hub.peer_review_log if r.cycle == cycle_idx]
                    self.coalition_tracker.find_coalition(round_reviews)

            # ── Stopping rule ────────────────────────────────────────────
            if self.stopping_rule in ("Convergence", "Either"):
                if hub.check_convergence():
                    return

        hub.check_convergence()


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------
#TODO: reducible space - on a tie, reduce the option set for the next round and re-run.
#TODO: entropy-based stopping criterion.


def _majority_vote(
    outputs: list[AgentOutput],
    current_labels: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """
    Aggregate AgentOutputs into a single label set by majority vote.
    Works on classification outputs (contribution = dict).
    """
    if not outputs:
        return dict(current_labels), {}

    all_fields: set[str] = set()
    for output in outputs:
        all_fields.update(output.labels.keys())  # .labels is safe — returns {} for non-classification

    labels: dict[str, Any] = {}
    vote_distribution: dict[str, dict[str, float]] = {}
    n = len(outputs)

    for field in all_fields:
        votes = []
        for output in outputs:
            val = output.labels.get(field)
            if val is None:
                continue
            # Normalise multi-label lists to a sorted comma-joined string for counting
            votes.append(",".join(sorted(val)) if isinstance(val, list) else str(val))

        if not votes:
            labels[field] = current_labels.get(field)
            continue

        counts = Counter(votes)
        vote_distribution[field] = {k: v / n for k, v in counts.items()}

        top = counts.most_common(2)
        # Tie: keep current label (unchanged from previous round)
        if len(top) > 1 and top[0][1] == top[1][1]:
            labels[field] = current_labels.get(field)
        else:
            winner = top[0][0]
            # Restore list type for multi-label fields
            if "," in winner and isinstance(outputs[0].labels.get(field), list):
                labels[field] = [v.strip() for v in winner.split(",")]
            else:
                labels[field] = winner

    return labels, vote_distribution