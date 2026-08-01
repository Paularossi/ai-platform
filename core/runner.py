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
    collect_result        → dict  (serialisable result from a completed hub)
    results_to_df         → pd.DataFrame  (flatten a list of result dicts)
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

def _resolve_dimensions(dim_configs: list[dict]) -> list[dict]:
    """
    Each dimension config supplies its own rubric when present (e.g. Agent 0
    mode, where dimensions are authored fresh per experiment — including
    dimension names that don't exist in DEFAULT_DIMENSIONS). If a config
    omits "rubric" (the manual-mode UI only ever stores name/label/weight/
    sw_weight, since its dimension picker is drawn from DEFAULT_DIMENSIONS),
    fall back to the matching DEFAULT_DIMENSIONS entry by name.
    """
    defaults_by_name = {d["name"]: d for d in DEFAULT_DIMENSIONS}
    resolved: list[dict] = []
    for dd in dim_configs:
        if dd.get("rubric"):
            resolved.append(dd)
        elif dd["name"] in defaults_by_name:
            resolved.append({**defaults_by_name[dd["name"]], **dd})
        # else: a named dimension with no rubric and no known default — skip,
        # nothing to show the reviewer for it.
    return resolved


def build_peer_reviewer(cfg: dict, review_depth: str) -> PeerReviewRound | None:
    """
    Build the shared PeerReviewRound instance (one per experiment, reused across items).
    Returns None when ECU scoring is disabled.
    """
    ecu_cfg = cfg.get("ecu", {})
    if not ecu_cfg.get("enabled", False):
        return None

    dim_configs = ecu_cfg.get("dimensions") or DEFAULT_DIMENSIONS
    active_dims = _resolve_dimensions(dim_configs)
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
    active_dims = _resolve_dimensions(dim_configs)

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
    if setting == "Simultaneous":
        return CrowdProtocol(agents, cfg, **kwargs)
    elif setting == "Sequential":
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
    Safe to call after protocol.run() or after the streaming loop finishes.
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
    }


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def _agent_zero_result_row(r: dict) -> dict:
    """Summarise an Agent 0-mode result (different shape from manual mode) into one row."""
    rounds = r.get("rounds", [])
    last = rounds[-1] if rounds else {}
    row: dict[str, Any] = {
        "item_id": r["item_id"],
        "num_turns": r["num_turns"],
        "ended_reason": r.get("ended_reason"),
        "num_rounds": len(rounds),
        "initial_roster_size": len(r.get("initial_roster", [])),
        "final_roster_size": len(last.get("roster", [])),
    }
    coalition = last.get("coalition", [])
    row["coalition_final"] = ", ".join(coalition) if len(coalition) >= 2 else "none"
    row["coalition_reached"] = len(coalition) >= 2
    row["social_welfare_final"] = last.get("social_welfare")
    for agent, bal in r.get("final_ecu_balances", {}).items():
        row[f"ecu_{agent}"] = round(bal, 4)
    return row


def results_to_df(results: list[dict]) -> pd.DataFrame:
    """Flatten a list of result dicts into a wide DataFrame (one row per item)."""
    rows = []
    for r in results:
        if r.get("mode") == "agent_zero":
            rows.append(_agent_zero_result_row(r))
            continue

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
