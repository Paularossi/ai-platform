# core/protocols/__init__.py
"""
Shared protocol primitives.

RunEvent is defined here so every protocol (Gossip, Crowd, Duel, Court, ...)
can emit the same event type and the UI render loop in 6_Run.py needs no changes
when a new protocol is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from core.hub import ContextPacket
from core.state import AgentOutput, PeerReviewOutput


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
        # Cached per cycle on the ledger so this matches whatever the Phase 1
        # contribution prompt showed for the same cycle - see
        # EcuLedger.noisy_weights_for_cycle().
        noisy = hub.ledger.noisy_weights_for_cycle(cycle)
        w_str = ", ".join(f"{k}≈{v}" for k, v in noisy.items())
        own = balances.get(agent_name, 0.0)
        return f"Your ECU balance: {own:.3f}. Approximate dimension weights: {w_str}."

    return ""


def run_peer_review_for_agent(
    agent: Any,
    hub: Any,
    cycle: int,
    peer_reviewer: Any,
    all_contributions: dict[str, Any],
    item_context: str,
    agent_histories: dict[str, list[str]],
    collect_importance_votes: bool,
    dry_run: bool,
) -> PeerReviewOutput:
    """
    Run one agent's peer review call: build the prompt, call its provider,
    log the prompt, and parse the response into a PeerReviewOutput.

    Shared by every protocol (Crowd, Gossip, Agent 0's loop) so the actual
    peer-review mechanics - prompt construction, the provider call, response
    parsing - live in exactly one place instead of being copied per protocol.
    Each protocol still owns its own control flow around this call (event
    streaming, orchestrator hookup, etc.), since that genuinely differs.
    """
    if dry_run:
        return peer_reviewer.parse(
            reviewer_name=agent.name, cycle=cycle, raw="[dry-run]",
            all_contributions=all_contributions,
        )

    ecu_info = build_ecu_info_str(hub, cycle, agent.name)
    prompt = peer_reviewer.build_prompt(
        reviewer_name=agent.name,
        reviewer_contribution=all_contributions.get(agent.name, ""),
        all_contributions=all_contributions,
        cycle=cycle,
        item_context=item_context,
        ecu_info=ecu_info,
        reviewer_role=agent.role,
        agent_histories=agent_histories,
        collect_importance_votes=collect_importance_votes,
    )
    try:
        from core.providers import get_provider
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

    hub.log_prompt(cycle, agent.name, "peer_review", prompt=prompt, response=raw)
    return peer_reviewer.parse(
        reviewer_name=agent.name, cycle=cycle, raw=raw, all_contributions=all_contributions,
    )


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