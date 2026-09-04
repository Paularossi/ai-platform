"""
core/runner.py

Construction helpers shared by the Streamlit UI (5_Run.py) and any batch / scripting entry-point.

The UI owns the streaming loop and visual rendering; this module owns
everything needed to build a runnable experiment item:

    build_agents          → list[Agent]
    build_peer_reviewer   → PeerReviewRound | None
    build_ecu_components  → (EcuLedger | None, CoalitionTracker | None, Orchestrator | None)
    build_protocol        → CrowdProtocol | GossipProtocol
    build_hub             → CommunicationHub
    build_item_data       → dict  (from a DataFrame row + column mapping)
    collect_result        → dict  (serialisable result from a completed hub)
    results_to_df         → pd.DataFrame  (flatten a list of result dicts)
    build_transcript_md   → str  (a readable, human-facing markdown transcript)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.agent import Agent
from core.ecu import CoalitionTracker, DEFAULT_DIMENSIONS, EcuLedger, PeerReviewRound
from core.hub import CommunicationHub
from core.orchestrator import Orchestrator
from core.protocols.crowd import CrowdProtocol
from core.protocols.gossip import GossipProtocol


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def build_agents(cfg: dict) -> list[Agent]:
    """Instantiate Agent objects from the experiment config."""
    return [Agent(a, cfg) for a in cfg.get("agents", [])]


# ---------------------------------------------------------------------------
# ECU / peer review construction
# ---------------------------------------------------------------------------

def build_peer_reviewer(cfg: dict, review_depth: str) -> PeerReviewRound | None:
    """
    Build the shared PeerReviewRound instance (one per experiment, reused across items).
    Returns None when ECU scoring is disabled.
    """
    ecu_cfg = cfg.get("ecu", {})
    if not ecu_cfg.get("enabled", False):
        return None

    dim_configs = ecu_cfg.get("dimensions") or DEFAULT_DIMENSIONS
    active_dims = [d for d in DEFAULT_DIMENSIONS if d["name"] in {dd["name"] for dd in dim_configs}]
    return PeerReviewRound(
        dimensions=active_dims,
        include_self_assessment=ecu_cfg.get("include_self_assessment", False),
        review_depth=review_depth,
    )


def build_ecu_components(
    cfg: dict,
    agent_names: list[str],
    review_depth: str,
) -> tuple[EcuLedger | None, CoalitionTracker | None, Orchestrator | None]:
    """
    Build a fresh EcuLedger, CoalitionTracker, and Orchestrator for one item.
    Must be called once per item (not reused across items).
    Returns (None, None, None) when ECU scoring is disabled.
    """
    ecu_cfg = cfg.get("ecu", {})
    if not ecu_cfg.get("enabled", False):
        return None, None, None

    dim_configs = ecu_cfg.get("dimensions") or DEFAULT_DIMENSIONS
    ecu_weights = {d["name"]: float(d.get("weight", 1.0)) for d in dim_configs}
    sw_weights = {d["name"]: float(d.get("sw_weight", 1.0)) for d in dim_configs}
    active_dims = [d for d in DEFAULT_DIMENSIONS if d["name"] in ecu_weights]

    ledger = EcuLedger(
        agent_names=agent_names,
        dimensions=active_dims,
        ecu_weights=ecu_weights,
        sw_weights=sw_weights,
        include_self_assessment=ecu_cfg.get("include_self_assessment", False),
    )
    coalition = CoalitionTracker(threshold=float(ecu_cfg.get("coalition_threshold", 0.6)))
    orchestrator: Orchestrator | None = None
    if ecu_cfg.get("orchestrator_enabled", False):
        orchestrator = Orchestrator(
            update_every=int(ecu_cfg.get("orchestrator_every", 1)),
            enabled=True,
        )

    return ledger, coalition, orchestrator


# ---------------------------------------------------------------------------
# Protocol construction
# ---------------------------------------------------------------------------

def build_protocol(
    agents: list[Agent],
    cfg: dict,
    peer_reviewer: PeerReviewRound | None = None,
    coalition_tracker: CoalitionTracker | None = None,
    orchestrator: Orchestrator | None = None,
    dry_run: bool = False,
) -> CrowdProtocol | GossipProtocol:
    """Return the correct protocol instance based on the configured setting."""
    setting = cfg.get("protocol", {}).get("setting", "Simultaneous")
    kwargs = dict(
        peer_reviewer=peer_reviewer,
        coalition_tracker=coalition_tracker,
        orchestrator=orchestrator,
        dry_run=dry_run,
    )
    if setting in ("Simultaneous", "Crowd (parallel)"):
        return CrowdProtocol(agents, cfg, **kwargs)
    elif setting in ("Sequential", "Gossip (sequential)"):
        return GossipProtocol(agents, cfg, **kwargs)
    raise ValueError(f"Unknown protocol setting: '{setting}'")


# ---------------------------------------------------------------------------
# Hub construction
# ---------------------------------------------------------------------------

def build_hub(
    item_id: str,
    item_data: dict[str, Any],
    cfg: dict,
    agent_names: list[str],
    ledger: EcuLedger | None = None,
) -> CommunicationHub:
    """Create a CommunicationHub for one experiment item."""
    proto = cfg.get("protocol", {})
    ecu_cfg = cfg.get("ecu", {})
    return CommunicationHub(
        item_id=item_id,
        item_data=item_data,
        visibility_mode=proto.get("visibility_mode", "Previous round"),
        agent_names=agent_names,
        ledger=ledger,
        ecu_info_condition=ecu_cfg.get("info_condition", "opaque"),
        review_depth=proto.get("review_depth", "Previous Round"),
    )


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def collect_result(
    hub: CommunicationHub,
    ledger: EcuLedger | None,
    coalition: CoalitionTracker | None,
    orchestrator: Orchestrator | None,
) -> dict:
    """
    Serialise the completed hub + ECU objects into a result dict.
    Safe to call once protocol.run_iter(hub) has been fully drained.
    """
    return {
        "item_id": hub.item_id,
        "num_turns": hub.num_submissions,
        "originator_name": hub.originator_name,
        "ecu_balances": hub.ecu_balances,
        "coalition_history": coalition.to_dict() if coalition else {},
        "orchestrator": orchestrator.to_dict() if orchestrator else {},
        "ecu_ledger": ledger.to_dict() if ledger else {},
        "log": [o.to_dict() for o in hub.log],
        "peer_review_log": [p.to_dict() for p in hub.peer_review_log],
        "prompt_log": hub.prompt_log,
        "total_tokens": hub.total_tokens,
    }


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def build_item_data(row: pd.Series, column_mapping: dict) -> dict:
    """Build the item_data dict from a DataFrame row using the column mapping."""
    return {
        field_name: str(row[col_name])
        for field_name, col_name in column_mapping.items()
        if col_name and col_name != "- not mapped -" and col_name in row.index
    }


def results_to_df(results: list[dict]) -> pd.DataFrame:
    """Flatten a list of result dicts into a wide DataFrame (one row per item)."""
    rows = []
    for r in results:
        row: dict[str, Any] = {
            "item_id": r["item_id"],
            "num_turns": r["num_turns"],
            "originator": r["originator_name"],
        }

        coalition_hist = r.get("coalition_history", {}).get("history", [])
        if coalition_hist:
            last = coalition_hist[-1]
            row["coalition_final"] = ", ".join(last.get("coalition", [])) or "none"
            row["coalition_size"] = last.get("size", 0)
            row["coalition_reached"] = last.get("size", 0) >= 2
        else:
            row["coalition_final"] = "—"
            row["coalition_size"] = 0
            row["coalition_reached"] = False

        for agent, bal in r.get("ecu_balances", {}).items():
            row[f"ecu_{agent}"] = round(bal, 4)

        sw_history = r.get("ecu_ledger", {}).get("social_welfare_history", [])
        if sw_history:
            row["social_welfare_final"] = sw_history[-1].get("social_welfare")

        pr_log = r.get("peer_review_log", [])
        if pr_log:
            last_cycle = max(p["cycle"] for p in pr_log)
            last_round = [p for p in pr_log if p["cycle"] == last_cycle]
            score_sums: dict[str, dict[str, float]] = {}
            score_counts: dict[str, int] = {}
            for review in last_round:
                for reviewed, dim_scores in review.get("scores", {}).items():
                    if reviewed not in score_sums:
                        score_sums[reviewed] = {}
                        score_counts[reviewed] = 0
                    score_counts[reviewed] += 1
                    for dim, score in dim_scores.items():
                        score_sums[reviewed][dim] = score_sums[reviewed].get(dim, 0) + score
            for agent, sums in score_sums.items():
                n = score_counts[agent]
                for dim, total in sums.items():
                    row[f"pr_{agent}_{dim}"] = round(total / n, 4)

        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Readable transcript export
# ---------------------------------------------------------------------------

def build_transcript_md(result: dict, cfg: dict) -> str:
    """
    Render a single run's result dict as a readable markdown transcript —
    topic, each round's contributions, peer review (if it ran), and the
    recorded outcome — meant for students to read or hand in, as opposed
    to the full JSON log which is meant for inspection/debugging.
    """
    lines: list[str] = []

    exp_name = cfg.get("overview", {}).get("name") or "Deliberation"
    topic = cfg.get("task", {}).get("description", "").strip()
    proto = cfg.get("protocol", {})
    agents_cfg = cfg.get("agents", [])

    lines.append(f"# {exp_name} — Transcript")
    lines.append("")
    if topic:
        lines.append(f"**Topic:** {topic}")
    lines.append(
        f"**Setting:** {proto.get('setting', '-')}  ·  "
        f"**Agents:** {len(agents_cfg)}  ·  "
        f"**Turns:** {result.get('num_turns', '-')}"
    )
    lines.append("")

    log = result.get("log", [])
    pr_log = result.get("peer_review_log", [])
    cycles = sorted({o["cycle"] for o in log})

    for cycle in cycles:
        lines.append(f"## Round {cycle + 1}")
        lines.append("")
        for out in [o for o in log if o["cycle"] == cycle]:
            contrib = out.get("contribution") or "*(empty)*"
            lines.append(f"**{out['agent_name']}:**")
            lines.append("")
            lines.append(str(contrib))
            lines.append("")

        round_reviews = [p for p in pr_log if p["cycle"] == cycle]
        if round_reviews:
            lines.append("**Peer review:**")
            lines.append("")
            for review in round_reviews:
                for reviewed, dim_scores in review.get("scores", {}).items():
                    scores_str = ", ".join(f"{d}: {round(s, 2)}" for d, s in dim_scores.items())
                    justification = review.get("justifications", {}).get(reviewed, "")
                    lines.append(f"- *{review['reviewer_name']} → {reviewed}* — {scores_str}")
                    if justification:
                        lines.append(f"  \"{justification}\"")
            lines.append("")

    balances = result.get("ecu_balances")
    if balances:
        lines.append("## ECU balances (final)")
        lines.append("")
        for name, bal in balances.items():
            lines.append(f"- **{name}:** {bal:.3f}")
        lines.append("")

    coalition_hist = result.get("coalition_history", {}).get("history", [])
    if coalition_hist:
        last = coalition_hist[-1]
        coalition = last.get("coalition", [])
        if len(coalition) >= 2:
            lines.append(f"**Coalition reached:** {', '.join(coalition)}")
            lines.append("")

    # Outcome goes last, as it's the verdict on everything above.
    outcome = result.get("outcome")
    if outcome:
        lines.append("---")
        lines.append("")
        lines.append("## Outcome of the debate")
        lines.append("")
        lines.append(f"**{outcome.get('label', '-')}**")
        if outcome.get("notes"):
            lines.append("")
            lines.append(outcome["notes"])
        lines.append("")

    reflection = result.get("reflection")
    if reflection:
        lines.append("## Reflection")
        lines.append("")
        lines.append(reflection)
        lines.append("")

    return "\n".join(lines)
