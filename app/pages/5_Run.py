"""
6_Run.py  -  Experiment runner

Streams the gossip protocol live, item by item, agent by agent.
Shows a live feed of each agent turn, then a per-item results table,
and finally lets you download the full results as CSV + JSON.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── make sure core/ is importable when running from app/ ──────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.agent import Agent
from core.hub import CommunicationHub
from core.protocols.gossip import GossipProtocol
from core.protocols.crowd import CrowdProtocol
from core.orchestrator import Orchestrator
from core.ecu import EcuLedger, PeerReviewRound, CoalitionTracker, DEFAULT_DIMENSIONS

st.set_page_config(page_title="Run Experiment", page_icon="🚀", layout="wide")

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("▶ Running")
st.sidebar.progress(1.0)
st.sidebar.markdown("""
**Steps**
1. Overview
2. Agents
3. Instructions & topic
4. Review
5. **Run ← you are here**
"""
)


# ---------- helpers ----------

def get_config() -> dict:
    return st.session_state.get("experiment_config", {})


def build_agents(cfg: dict) -> list[Agent]:
    return [Agent(a, cfg) for a in cfg.get("agents", [])]


def build_protocol(agents: list[Agent], cfg: dict, dry_run: bool,
                   peer_reviewer=None, coalition_tracker=None, orchestrator=None):
    """Return the correct protocol instance based on the configured setting."""
    setting = cfg.get("protocol", {}).get("setting", "Simultaneous")
    if setting in ("Simultaneous", "Crowd (parallel)"):
        return CrowdProtocol(agents, cfg, peer_reviewer=peer_reviewer,
                             coalition_tracker=coalition_tracker,
                             orchestrator=orchestrator, dry_run=dry_run)
    elif setting in ("Sequential", "Gossip (sequential)"):
        return GossipProtocol(agents, cfg, peer_reviewer=peer_reviewer,
                              coalition_tracker=coalition_tracker,
                              orchestrator=orchestrator, dry_run=dry_run)
    raise ValueError(f"Unknown protocol setting: '{setting}'")


def _normalize_image_key(raw: Any) -> str:
    """Normalize an item/image identifier for tolerant filename matching."""
    s = str(raw).strip()
    s = re.sub(r"\.0+$", "", s)
    return s.lower()


def resolve_image_path(
    item_id: str,
    dataset_cfg: dict,
    zip_bytes: bytes | None,
) -> str | None:
    """Try to find the image file on disk for a given item_id stem."""
    image_dir = st.session_state.get("dataset_image_dir")
    if not image_dir:
        return None

    lookup: dict[str, str] = st.session_state.get("dataset_image_lookup", {}) or {}
    norm = _normalize_image_key(item_id)

    # Direct lookup from ZIP extraction
    if norm in lookup and os.path.exists(lookup[norm]):
        return lookup[norm]

    # Fallback scan of the image directory
    try:
        for name in os.listdir(image_dir):
            full = os.path.join(image_dir, name)
            if os.path.isfile(full) and _normalize_image_key(os.path.splitext(name)[0]) == norm:
                return full
    except Exception:
        pass

    return None


def build_item_data(row: pd.Series, column_mapping: dict, image_path: str | None) -> dict:
    """
    Build the item_data dict from a dataframe row using the column mapping.
    Image is stored as a path; the agent will load it when needed.
    """
    data: dict[str, Any] = {}
    for field_name, col_name in column_mapping.items():
        if col_name == "- not mapped -" or not col_name:
            continue
        if col_name in row.index:
            data[field_name] = str(row[col_name])

    if image_path:
        data["image_path"] = image_path

    return data


def results_to_df(results: list[dict]) -> pd.DataFrame:
    """Flatten results into a wide DataFrame. One row per item."""
    rows = []
    for r in results:
        row: dict[str, Any] = {
            "item_id": r["item_id"],
            "num_turns": r["num_turns"],
            "originator": r["originator_name"],
        }

        # Coalition outcome
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

        # ECU final balances
        for agent, bal in r.get("ecu_balances", {}).items():
            row[f"ecu_{agent}"] = round(bal, 4)

        # Per-agent mean peer scores from last round
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


# ---------- page ----------

st.title("🚀 Run Experiment")
cfg = get_config()

# ── Guard: check experiment is configured ─────────────────────────────────────
agents_cfg = cfg.get("agents", [])
questions = cfg.get("questions", [])
task_mode = cfg.get("task", {}).get("mode", "deliberation")
dataset_cfg = cfg.get("dataset", {})
df: pd.DataFrame | None = st.session_state.get("dataset_df")
column_mapping: dict = st.session_state.get("column_mapping", {})

missing = []
if not agents_cfg:
    missing.append("No agents configured — go to Step 2")
if df is None:
    missing.append("No deliberation topic set — go to Step 3")

if missing:
    st.error("Cannot run - please complete setup first:")
    for m in missing:
        st.markdown(f"- {m}")
    if st.button("← Back to Review"):
        st.switch_page("pages/5_Review.py")
    st.stop()

# ── Run configuration panel ───────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Run configuration")

    col1, col2 = st.columns([2, 3])

    with col1:
        n_items = st.number_input(
            "Items to process",
            min_value=1,
            max_value=len(df),
            value=min(5, len(df)),
            step=1,
            help=f"Dataset has {len(df)} rows total.",
        )

    with col2:
        api_key_input = st.text_input(
            "OpenAI API key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            help="Leave blank if OPENAI_API_KEY is already set in your environment.",
        )

dry_run = False  # Always use real API calls

proto = cfg.get("protocol", {})
c1, c2, c3, c4 = st.columns([5, 2, 2, 3])
c1.metric("Setting", proto.get("setting", "-"))
c2.metric("Agents", len(agents_cfg))
c3.metric("Max cycles", proto.get("max_cycles", "-"))
c4.metric("Stopping rule", proto.get("stopping_rule", "-"))

# ── Results renderer (called both after run and on download rerun) ────────────
def _show_results(results: list[dict], experiment_cfg: dict) -> None:
    st.divider()
    st.subheader("Results")

    col1, col2 = st.columns(2)
    col1.metric("Items processed", len(results))
    col2.metric("Total turns", sum(r["num_turns"] for r in results))

    results_df = results_to_df(results)
    st.dataframe(results_df, width="stretch", hide_index=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    exp_name = experiment_cfg.get("overview", {}).get("name", "experiment").replace(" ", "_").lower()

    dl1, dl2 = st.columns(2)
    with dl1:
        csv_bytes = results_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Results CSV",
            data=csv_bytes,
            file_name=f"{exp_name}_{timestamp}_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        full_json = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "⬇ Full log JSON",
            data=full_json.encode(),
            file_name=f"{exp_name}_{timestamp}_log.json",
            mime="application/json",
            use_container_width=True,
            help="Includes contributions, peer reviews, ECU history, Orchestrator updates, and all prompts.",
        )


def _show_nav_buttons():
    st.divider()
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("🔄 Restart debate", use_container_width=True,
                     help="Clear results and run the same experiment again."):
            st.session_state.pop("run_results", None)
            st.rerun()
    with nav2:
        if st.button("✨ Start new experiment", use_container_width=True, type="primary",
                     help="Clear all settings and start from scratch."):
            keys_to_clear = [
                "experiment_config", "question_sets",
                "base_instructions", "guideline_notes", "agent_prompt_overrides",
                "agents", "num_agents", "interaction_setting", "supervision_mode",
                "visibility_mode", "review_depth", "order_type", "max_cycles",
                "stopping_rule", "initializer_agent", "judge_agent",
                "exp_name", "author", "protocol_id", "task_description",
                "dataset_df", "dataset_filename", "column_mapping", "dataset_ready",
                "dataset_source", "inline_rows", "run_results",
                "ecu_enabled", "ecu_info_condition", "ecu_self_assessment",
                "ecu_coalition_threshold", "ecu_dimensions",
            ]
            for key in keys_to_clear:
                st.session_state.pop(key, None)
            st.switch_page("pages/1_Welcome.py")


# ── Launch button ─────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    launch = st.button("▶ Launch", type="primary", width="stretch")

if not launch:
    # If a previous run's results are stored, show them even without relaunching
    if "run_results" in st.session_state:
        _show_results(st.session_state["run_results"], cfg)
        _show_nav_buttons()
    st.stop()

# Clear any previous results so a fresh run always starts clean
st.session_state.pop("run_results", None)

# ── Set API key if provided ───────────────────────────────────────────────────
if not dry_run and api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

# ── Build agents ──────────────────────────────────────────────────────────────
agents = build_agents(cfg)
agent_names = [a.name for a in agents]
visibility_mode = proto.get("visibility_mode", "Previous round")
review_depth = proto.get("review_depth", "previous_round")

# ── Diagnostics ───────────────────────────────────────────────────────────────
with st.expander("🔍 Pre-run diagnostics", expanded=False):
    st.markdown(f"**Task mode:** {task_mode}")
    st.markdown(f"**Agents:** {[a.name for a in agents]}")
    st.markdown(f"**Visibility mode:** {visibility_mode}")
    st.markdown(f"**Column mapping:** {st.session_state.get('column_mapping', {})}")

    if task_mode == "classification":
        if cfg.get("questions"):
            st.markdown("**Question fields:** " + ", ".join(
                f"`{q['field_name']}`" for q in cfg["questions"]
            ))
    else:
        st.info("Deliberation mode — no questions required. Agents produce free-text contributions.")

    # Image diagnostics (only relevant when images are in the dataset)
    img_dir = st.session_state.get("dataset_image_dir")
    if img_dir:
        st.divider()
        st.markdown("**Image resolution diagnostics**")
        st.success(f"Image temp dir: `{img_dir}`")
        try:
            files_on_disk = os.listdir(img_dir)
            st.markdown(f"Files in temp dir ({len(files_on_disk)} total), first 10:")
            st.code("\n".join(files_on_disk[:10]))
        except Exception as e:
            st.error(f"Cannot list temp dir: {e}")

    st.divider()
    if agents:
        st.markdown("**System prompt preview (Agent 1):**")
        from core.agent import _build_system_prompt
        preview = _build_system_prompt(
            agent_name=agents[0].name,
            agent_role=agents[0].role,
            base_instructions=cfg.get("instructions", {}).get("base_instructions", ""),
            guideline_notes=cfg.get("instructions", {}).get("guideline_notes", ""),
            agent_overrides=cfg.get("agent_prompt_overrides", {}),
            questions=cfg.get("questions", []) if task_mode == "classification" else [],
        )
        st.code(preview, language=None)

# ── Build ECU peer reviewer and ledger (if enabled) ───────────────────────────
ecu_cfg = cfg.get("ecu", {})
ecu_enabled = ecu_cfg.get("enabled", False)
ecu_info_condition = ecu_cfg.get("info_condition", "opaque")
include_self_assessment = ecu_cfg.get("include_self_assessment", False)
coalition_threshold = float(ecu_cfg.get("coalition_threshold", 0.6))

peer_reviewer: PeerReviewRound | None = None
if ecu_enabled:
    dim_configs = ecu_cfg.get("dimensions") or DEFAULT_DIMENSIONS
    active_dims = [d for d in DEFAULT_DIMENSIONS if d["name"] in {dd["name"] for dd in dim_configs}]
    peer_reviewer = PeerReviewRound(
        dimensions=active_dims,
        include_self_assessment=include_self_assessment,
        review_depth=review_depth,
        dry_run=dry_run,
    )
    if dry_run:
        st.info("ECU peer review is in dry-run mode — all scores will be 0.5.", icon="💡")

# ── Slice dataset ─────────────────────────────────────────────────────────────
subset = df.head(int(n_items)).reset_index(drop=True)

# ── Results accumulator ───────────────────────────────────────────────────────
all_results: list[dict] = []

st.divider()
st.subheader(f"Running on {len(subset)} item(s)…")

progress_bar = st.progress(0.0, text="Starting…")
results_placeholder = st.empty()

# ── Per-item status table (updates live) ──────────────────────────────────────
status_rows: list[dict] = []

# ── Main loop ─────────────────────────────────────────────────────────────────
for item_idx, row in subset.iterrows():
    id_col = column_mapping.get("id", column_mapping.get("image", None))
    if id_col and id_col in df.columns:
        item_id = str(row[id_col])
    elif len(df.columns) > 0:
        item_id = str(row[df.columns[0]])
    else:
        item_id = f"item_{item_idx}"

    image_path = resolve_image_path(item_id, dataset_cfg, None)
    item_data = build_item_data(row, column_mapping, image_path)

    # Build a fresh ledger, coalition tracker, and orchestrator per item
    item_ledger: EcuLedger | None = None
    item_coalition: CoalitionTracker | None = None
    item_orchestrator: Orchestrator | None = None
    if ecu_enabled and peer_reviewer:
        dim_configs = ecu_cfg.get("dimensions") or DEFAULT_DIMENSIONS
        ecu_weights = {d["name"]: float(d.get("weight", 1.0)) for d in dim_configs}
        sw_weights = {d["name"]: float(d.get("sw_weight", 1.0)) for d in dim_configs}
        active_dims = [d for d in DEFAULT_DIMENSIONS if d["name"] in ecu_weights]
        item_ledger = EcuLedger(
            agent_names=agent_names,
            dimensions=active_dims,
            ecu_weights=ecu_weights,
            sw_weights=sw_weights,
            include_self_assessment=include_self_assessment,
        )
        item_coalition = CoalitionTracker(threshold=coalition_threshold)
        if ecu_cfg.get("orchestrator_enabled", False):
            item_orchestrator = Orchestrator(
                step_size=float(ecu_cfg.get("orchestrator_step_size", 0.1)),
                update_every=int(ecu_cfg.get("orchestrator_every", 2)),
                enabled=True,
            )

    # Build protocol fresh per item
    protocol = build_protocol(
        agents, cfg, dry_run=dry_run,
        peer_reviewer=peer_reviewer,
        coalition_tracker=item_coalition,
        orchestrator=item_orchestrator,
    )

    # Create hub for this item
    hub = CommunicationHub(
        item_id=item_id,
        item_data=item_data,
        visibility_mode=visibility_mode,
        agent_names=agent_names,
        ledger=item_ledger,
        ecu_info_condition=ecu_info_condition,
        review_depth=review_depth,
    )

    # ── Live turn feed for this item ──────────────────────────────────────
    with st.expander(f"📄 Item {item_idx + 1} / {len(subset)}  -  `{item_id}`", expanded=True):

        if item_data.get("image_path"):
            st.caption(f"Image: {item_data['image_path']}")

        feed = st.empty()
        entries: list[str] = []   # accumulated markdown lines rendered all at once

        t_start = time.time()

        is_crowd = cfg.get("protocol", {}).get("setting", "") in ("Simultaneous", "Crowd (parallel)")
        # In Crowd mode we render one compact "context" block at the START of each
        # new cycle (before any submissions) instead of a per-agent dispatch block.
        _last_crowd_cycle_rendered = -1

        for event in protocol.run_iter(hub):

            if event.kind == "dispatch":
                if is_crowd:
                    if event.cycle > 0 and event.cycle != _last_crowd_cycle_rendered:
                        _last_crowd_cycle_rendered = event.cycle
                        packet = event.packet
                        agent_list = [a.name for a in agents]
                        dlines = [
                            f"#### 📬 Round {event.cycle + 1} · Context sent to all agents",
                            f"**Agents:** {', '.join(agent_list)}",
                        ]
                        if packet.visible_history:
                            round_labels = ", ".join(
                                f"{h.agent_name} (round {h.cycle + 1})"
                                for h in packet.visible_history
                            )
                            dlines.append(f"**Previous contributions visible:** {round_labels}")
                        else:
                            dlines.append("**Round 1 — agents contribute blind (no prior context).**")
                        entries.append("\n\n".join(dlines))
                    continue

                # ── What the hub sent to this agent ──────────────────────
                packet = event.packet
                lines = [f"#### 🔀 Hub → **{event.agent_name}**  ·  Cycle {event.cycle + 1}"]

                current = packet.current_contribution
                if isinstance(current, dict) and current:
                    label_str = "  ,  ".join(f"`{k}`: {v}" for k, v in current.items())
                    lines.append(f"**Current answers:** {label_str}")
                elif isinstance(current, str) and current.strip():
                    lines.append(f"**Current position:** {current}")
                else:
                    lines.append("**Current state:** *(none yet)*")

                if packet.visible_history:
                    lines.append(f"**Visible history** ({len(packet.visible_history)} entry/entries):")
                    for h in packet.visible_history:
                        h_str = str(h.contribution)[:120] if h.contribution else "(none)"
                        lines.append(f"- Round {h.cycle + 1} · **{h.agent_name}**: {h_str}")
                else:
                    lines.append(f"**Visible history:** *(none — {visibility_mode})*")

                entries.append("\n\n".join(lines))

            elif event.kind == "submission":
                # ── What the agent returned to the hub ───────────────────
                output = event.output
                changed_fields = [f for f, c in output.changed.items() if c]
                change_note = f"✏️ revised: {', '.join(changed_fields)}" if changed_fields else "✔ no changes"

                lines = [f"#### 📨 **{event.agent_name}** → Hub  ·  Round {event.cycle + 1}  ·  {change_note}"]
                contrib = str(output.contribution) if output.contribution else "*(empty)*"
                lines.append(f"**Contribution:** {contrib}")
                if not output.contribution and output.raw_response and not output.raw_response.startswith("[dry-run"):
                    lines.append(f"⚠️ **Raw response (parse failed):**\n```\n{output.raw_response}\n```")
                entries.append("\n\n".join(lines))

            elif event.kind == "peer_review":
                round_reviews = [
                    r for r in hub.peer_review_log
                    if r.cycle == event.cycle and r.reviewer_name == event.agent_name
                ]
                if round_reviews:
                    review = round_reviews[-1]
                    lines = [f"#### 📋 **{event.agent_name}** peer review  ·  Round {event.cycle + 1}"]
                    for reviewed, dim_scores in review.scores.items():
                        score_str = "  ,  ".join(f"{d}: **{s:.2f}**" for d, s in dim_scores.items())
                        justification = review.coalition_justifications.get(reviewed, "")
                        lines.append(f"→ **{reviewed}**: {score_str}")
                        if justification and justification not in ("[dry-run]", ""):
                            lines.append(f"  _{justification}_")
                    if review.self_scores:
                        self_str = "  ,  ".join(f"{d}: {s:.2f}" for d, s in review.self_scores.items())
                        lines.append(f"→ **Self**: {self_str}")
                    entries.append("\n\n".join(lines))

            elif event.kind == "ecu_update":
                balances = hub.ecu_balances
                lines = [f"#### 💰 ECU update  ·  after Round {event.cycle + 1}"]

                if item_ledger:
                    round_records = [r for r in item_ledger.history if r.cycle == event.cycle]
                    if round_records:
                        lines.append("**This round's scores (peer-averaged):**")
                        for rec in round_records:
                            score_str = "  ,  ".join(f"{d}: {s:.2f}" for d, s in rec.aggregated_scores.items())
                            lines.append(f"→ **{rec.agent_name}**: {score_str}  →  **+{rec.ecu_earned:.3f} ecus**")

                if balances:
                    bal_str = "  ·  ".join(f"**{n}**: {b:.3f}" for n, b in balances.items())
                    lines.append(f"**Cumulative balances:** {bal_str}")

                if item_coalition and item_coalition.history:
                    last_c = item_coalition.history[-1]
                    c = last_c.get("coalition", [])
                    lines.append(
                        f"**Coalition** (τ={coalition_threshold}): "
                        + (f"**{', '.join(c)}**" if c else "*(none above threshold)*")
                    )

                if item_orchestrator:
                    cycle_updates = [u for u in item_orchestrator.history if u.cycle == event.cycle]
                    ran_this_cycle = (event.cycle + 1) % item_orchestrator.update_every == 0
                    if cycle_updates:
                        lines.append("**Orchestrator weight updates:**")
                        for u in cycle_updates:
                            lines.append(
                                f"→ {u.dimension}: {u.old_weight:.2f} → **{u.new_weight:.2f}** "
                                f"(SW: {u.sw_before:.3f} → {u.sw_after:.3f})"
                            )
                        # Show current full weight vector after updates
                        current_w = {k: round(v, 3) for k, v in item_ledger.ecu_weights.items()}
                        w_str = "  ,  ".join(f"{k}={v}" for k, v in current_w.items())
                        lines.append(f"Updated ECU weights: {w_str}")
                    elif ran_this_cycle:
                        lines.append("**Orchestrator:** ran — no weight improvement found (already near-optimal for this round).")

                entries.append("\n\n".join(lines))

            # Re-render the full feed after every event
            feed.markdown("\n\n---\n\n".join(entries))

        elapsed = time.time() - t_start

        st.success(f"✅ Done  ·  {hub.num_submissions} turn(s)  ·  {elapsed:.1f}s")

        # ECU balances summary
        if hub.ecu_balances:
            bal_cols = st.columns(len(hub.ecu_balances))
            for col, (name, bal) in zip(bal_cols, hub.ecu_balances.items()):
                col.metric(name, f"{bal:.3f} ecus")

        # Orchestrator weight trajectory (debug)
        if item_orchestrator and item_orchestrator.history:
            with st.expander("📊 Orchestrator weight trajectory", expanded=False):
                if item_ledger and item_ledger.social_welfare_history:
                    st.markdown("**Social welfare trajectory**")
                    st.dataframe(pd.DataFrame(item_ledger.social_welfare_history), hide_index=True, use_container_width=True)
                rows = []
                for u in item_orchestrator.history:
                    rows.append({
                        "Round": u.cycle + 1,
                        "Dimension": u.dimension,
                        "Old weight": round(u.old_weight, 3),
                        "New weight": round(u.new_weight, 3),
                        "Direction": u.direction,
                        "SW before": round(u.sw_before, 4),
                        "SW after": round(u.sw_after, 4),
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.caption(
                    "Only ECU incentive weights w^ECU are updated. SW weights w^SW stay fixed."
                )

    # Store result
    result = {
        "item_id": item_id,
        "num_turns": hub.num_submissions,
        "originator_name": hub.originator_name,
        "ecu_balances": hub.ecu_balances,
        "coalition_history": item_coalition.to_dict() if item_coalition else {},
        "orchestrator": item_orchestrator.to_dict() if item_orchestrator else {},
        "ecu_ledger": item_ledger.to_dict() if item_ledger else {},
        "social_welfare_history": item_ledger.social_welfare_history if item_ledger else [],
        "log": [o.to_dict() for o in hub.log],
        "peer_review_log": [p.to_dict() for p in hub.peer_review_log],
        "prompt_log": hub.prompt_log,
    }
    all_results.append(result)

    status_row: dict = {
        "Item": item_id,
        "Converged": "✅" if hub.converged else "⏹",
        "Turns": hub.num_submissions,
        "Time (s)": f"{elapsed:.1f}",
    }
    if hub.ecu_balances:
        for name, bal in hub.ecu_balances.items():
            status_row[f"ECU {name}"] = f"{bal:.2f}"
    status_rows.append(status_row)

    # Update progress
    progress = (item_idx + 1) / len(subset)
    progress_bar.progress(progress, text=f"{item_idx + 1} / {len(subset)} items done")

    # Update live summary table
    results_placeholder.dataframe(
        pd.DataFrame(status_rows),
        width="stretch",
        hide_index=True,
    )

progress_bar.progress(1.0, text="Done!")

# Save results so they survive the download-button rerun
st.session_state["run_results"] = all_results

# ── Summary + download ────────────────────────────────────────────────────────
_show_results(all_results, cfg)
_show_nav_buttons()