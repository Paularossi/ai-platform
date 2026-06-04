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
from core.protocols import RunEvent
from core.protocols.gossip import GossipProtocol
from core.state import AgentOutput
from core.ecu import EcuLedger, PeerReviewRound, CoalitionTracker, DEFAULT_DIMENSIONS

st.set_page_config(page_title="Run Experiment", page_icon="🚀", layout="wide")

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("▶ Running")
st.sidebar.progress(1.0)
st.sidebar.markdown(
    """
**Steps**
1. Task
2. Agents
3. Instructions
4. Dataset
5. Review
6. **Run ← you are here**
"""
)


# ---------- helpers ----------

def get_config() -> dict:
    return st.session_state.get("experiment_config", {})


def build_agents(cfg: dict) -> list[Agent]:
    return [Agent(a, cfg) for a in cfg.get("agents", [])]


def build_protocol(agents: list[Agent], cfg: dict, dry_run: bool,
                   peer_reviewer=None, coalition_tracker=None):
    """Return the correct protocol instance based on the configured setting."""
    setting = cfg.get("protocol", {}).get("setting", "Gossip (sequential)")
    if setting == "Gossip (sequential)":
        return GossipProtocol(agents, cfg, peer_reviewer=peer_reviewer,
                              coalition_tracker=coalition_tracker, dry_run=dry_run)
    elif setting == "Crowd (parallel)":
        from core.protocols.crowd import CrowdProtocol
        return CrowdProtocol(agents, cfg, peer_reviewer=peer_reviewer,
                             coalition_tracker=coalition_tracker, dry_run=dry_run)
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
    """Flatten results list into a wide DataFrame, one row per item."""
    rows = []
    for r in results:
        row: dict[str, Any] = {
            "item_id": r["item_id"],
            "converged": r["converged"],
            "num_turns": r["num_turns"],
            "originator": r["originator_name"],
        }
        final = r.get("final_contribution")
        orig = r.get("originator_contribution")

        if isinstance(final, dict):
            # Classification task — expand fields as columns
            for fname, val in final.items():
                row[f"final_{fname}"] = val if not isinstance(val, list) else ", ".join(val)
            if isinstance(orig, dict):
                for fname, val in orig.items():
                    row[f"orig_{fname}"] = val if not isinstance(val, list) else ", ".join(val)
                for fname in final:
                    row[f"changed_{fname}"] = (
                        final.get(fname) != orig.get(fname)
                    )
        else:
            # Deliberation task — store as single text column
            row["final_contribution"] = str(final) if final else ""
            row["orig_contribution"] = str(orig) if orig else ""
            row["changed"] = final != orig

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
if task_mode == "classification" and not questions:
    missing.append("Classification task with no questions loaded — go to Step 3")
if df is None:
    missing.append("No dataset loaded — go to Step 4")

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

    col1, col2, col3 = st.columns([2, 2, 3])

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
        dry_run = st.toggle(
            "Dry run (no API calls)",
            value=True,
            help="Run without calling the OpenAI API. Agents return placeholder labels. "
                 "Useful for testing the pipeline end-to-end.",
        )

    with col3:
        api_key_input = st.text_input(
            "OpenAI API key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            help="Leave blank if OPENAI_API_KEY is already set in your environment.",
            disabled=dry_run,
        )

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

    n_converged = sum(1 for r in results if r["converged"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Items processed", len(results))
    col2.metric("Converged", f"{n_converged} / {len(results)}")
    col3.metric("Total turns", sum(r["num_turns"] for r in results))

    results_df = results_to_df(results)
    st.dataframe(results_df, width="stretch", hide_index=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    exp_name = experiment_cfg.get("overview", {}).get("name", "experiment").replace(" ", "_").lower()

    dl1, dl2 = st.columns(2)
    with dl1:
        csv_bytes = results_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download results CSV",
            data=csv_bytes,
            file_name=f"{exp_name}_{timestamp}_results.csv",
            mime="text/csv",
            width="stretch",
        )
    with dl2:
        full_json = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "⬇ Download full log JSON",
            data=full_json.encode(),
            file_name=f"{exp_name}_{timestamp}_log.json",
            mime="application/json",
            width="stretch",
        )


# ── Launch button ─────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    launch = st.button("▶ Launch", type="primary", width="stretch")

if not launch:
    # If a previous run's results are stored, show them even without relaunching
    if "run_results" in st.session_state:
        _show_results(st.session_state["run_results"], cfg)
    st.stop()

# Clear any previous results so a fresh run always starts clean
st.session_state.pop("run_results", None)

# ── Set API key if provided ───────────────────────────────────────────────────
if not dry_run and api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

# ── Build agents ──────────────────────────────────────────────────────────────
agents = build_agents(cfg)
agent_names = [a.name for a in agents]
visibility_mode = proto.get("visibility_mode", "Current state only")

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
            st.error("⚠️ No questions found — agents will return empty labels. Go to Step 3 and load your question set JSON.")
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

    # Build a fresh ledger and coalition tracker per item
    item_ledger: EcuLedger | None = None
    item_coalition: CoalitionTracker | None = None
    if ecu_enabled and peer_reviewer:
        dim_configs = ecu_cfg.get("dimensions") or DEFAULT_DIMENSIONS
        weights = {d["name"]: float(d.get("weight", 1.0)) for d in dim_configs}
        active_dims = [d for d in DEFAULT_DIMENSIONS if d["name"] in weights]
        item_ledger = EcuLedger(
            agent_names=agent_names,
            dimensions=active_dims,
            weights=weights,
            include_self_assessment=include_self_assessment,
        )
        item_coalition = CoalitionTracker(threshold=coalition_threshold)

    # Build protocol fresh per item (coalition tracker is per-item)
    protocol = build_protocol(
        agents, cfg, dry_run=dry_run,
        peer_reviewer=peer_reviewer,
        coalition_tracker=item_coalition,
    )

    # Create hub for this item
    hub = CommunicationHub(
        item_id=item_id,
        item_data=item_data,
        visibility_mode=visibility_mode,
        agent_names=agent_names,
        ledger=item_ledger,
        ecu_info_condition=ecu_info_condition,
    )

    # ── Live turn feed for this item ──────────────────────────────────────
    with st.expander(f"📄 Item {item_idx + 1} / {len(subset)}  -  `{item_id}`", expanded=True):

        if item_data.get("image_path"):
            st.caption(f"Image: {item_data['image_path']}")

        feed = st.empty()
        entries: list[str] = []   # accumulated markdown lines rendered all at once

        t_start = time.time()

        is_crowd = cfg.get("protocol", {}).get("setting", "") == "Crowd (parallel)"
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
                        current = packet.current_contribution
                        if isinstance(current, dict) and current:
                            lstr = "  ,  ".join(f"`{k}`: {v}" for k, v in current.items())
                            dlines.append(f"**Current answers:** {lstr}")
                        elif isinstance(current, str) and current.strip():
                            dlines.append(f"**Current position:** {current}")
                        else:
                            dlines.append("**Current state:** *(none yet — first round)*")
                        if packet.visible_history:
                            dlines.append(f"**Visible history:** {len(packet.visible_history)} entry/entries")
                        else:
                            dlines.append(f"**Visible history:** *(none — {visibility_mode})*")
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
                        if h.is_classification:
                            h_str = "  ,  ".join(f"`{k}`: {v}" for k, v in h.labels.items())
                        else:
                            h_str = str(h.contribution)[:120]
                        real_pros = [p for p in h.pros if p not in ("[dry-run]", "[dry-run placeholder]")]
                        pros_note = f"  \n  *{real_pros[0]}*" if real_pros else ""
                        lines.append(f"- Cycle {h.cycle + 1} · **{h.agent_name}**: {h_str}{pros_note}")
                else:
                    lines.append(f"**Visible history:** *(none — visibility mode: {visibility_mode})*")

                entries.append("\n\n".join(lines))

            elif event.kind == "submission":
                # ── What the agent returned to the hub ───────────────────
                output = event.output
                changed_fields = [f for f, c in output.changed.items() if c]
                change_note = f"✏️ revised: {', '.join(changed_fields)}" if changed_fields else "✔ no changes"

                lines = [f"#### 📨 **{event.agent_name}** → Hub  ·  {change_note}"]

                if output.is_classification:
                    label_str = "  ,  ".join(
                        f"`{k}`: {v}" for k, v in output.labels.items()
                    ) if output.labels else "*(no labels parsed)*"
                    lines.append(f"**Answers:** {label_str}")
                    if output.prob_distribution:
                        for fname, probs in output.prob_distribution.items():
                            conf_str = f"  *(confidence: {output.confidence:.2f})*" if output.confidence is not None else ""
                            prob_parts = ", ".join(f"{code}={p:.2f}" for code, p in probs.items())
                            lines.append(f"**{fname} distribution:** {prob_parts}{conf_str}")
                else:
                    contrib = str(output.contribution) if output.contribution else "*(empty)*"
                    lines.append(f"**Contribution:** {contrib}")
                    if output.confidence is not None:
                        lines.append(f"**Confidence:** {output.confidence:.2f}")

                if output.pros:
                    real_pros = [p for p in output.pros if p not in ("[dry-run]", "[dry-run placeholder]")]
                    if real_pros:
                        lines.append("**Pros:** " + " · ".join(f"_{p}_" for p in real_pros))
                if output.cons:
                    real_cons = [c for c in output.cons if c not in ("[dry-run]", "[dry-run placeholder]")]
                    if real_cons:
                        lines.append("**Cons:** " + " · ".join(f"_{c}_" for c in real_cons))

                if not output.contribution and output.raw_response and not output.raw_response.startswith("[dry-run"):
                    lines.append(f"⚠️ **Raw response (parse failed):**\n```\n{output.raw_response}\n```")

                entries.append("\n\n".join(lines))

            elif event.kind == "aggregate":
                # ── Round aggregate (Crowd protocol) ─────────────────────
                output = event.output
                if output.is_classification and output.labels:
                    label_str = "  ,  ".join(f"`{k}`: {v}" for k, v in output.labels.items())
                else:
                    label_str = "*(no consensus)*"

                lines = [f"#### 🗳️ Round {event.cycle + 1} aggregate"]
                lines.append(f"**Result:** {label_str}")

                if output.prob_distribution:
                    for fname, dist in output.prob_distribution.items():
                        vote_parts = ", ".join(
                            f"{code}: {p*100:.0f}%" for code, p in dist.items()
                        )
                        lines.append(f"**{fname} votes:** {vote_parts}")

                entries.append("\n\n".join(lines))

            elif event.kind == "peer_review":
                # ── Phase 2: one agent's peer review completed ────────────
                # Find the review in the hub log
                round_reviews = [
                    r for r in hub.peer_review_log
                    if r.cycle == event.cycle and r.reviewer_name == event.agent_name
                ]
                if round_reviews:
                    review = round_reviews[-1]
                    lines = [f"#### 📋 **{event.agent_name}** peer review  ·  Cycle {event.cycle + 1}"]
                    for reviewed, dim_scores in review.scores.items():
                        score_str = "  ,  ".join(
                            f"{d}: {s:.2f}" for d, s in dim_scores.items()
                        )
                        coalition_score = review.coalition_scores.get(reviewed, None)
                        coalition_str = f"  ·  coalition: **{coalition_score:.2f}**" if coalition_score is not None else ""
                        justification = review.coalition_justifications.get(reviewed, "")
                        lines.append(
                            f"→ **{reviewed}**: {score_str}{coalition_str}"
                            + (f"  \n  _{justification}_" if justification and justification != "[dry-run]" else "")
                        )
                    if review.self_scores:
                        self_str = "  ,  ".join(f"{d}: {s:.2f}" for d, s in review.self_scores.items())
                        lines.append(f"→ **Self**: {self_str}")
                    entries.append("\n\n".join(lines))

            elif event.kind == "ecu_update":
                # ── ECU balances updated after full peer review round ─────
                balances = hub.ecu_balances
                if balances:
                    lines = [f"#### 💰 ECU update  ·  after Cycle {event.cycle + 1}"]
                    bal_str = "  ·  ".join(
                        f"**{name}**: {bal:.3f}" for name, bal in balances.items()
                    )
                    lines.append(f"Cumulative balances: {bal_str}")

                    # Show coalition if available
                    if item_coalition and item_coalition.history:
                        last_coalition = item_coalition.history[-1]
                        c = last_coalition.get("coalition", [])
                        lines.append(
                            f"Coalition (τ={coalition_threshold}): "
                            + (f"**{', '.join(c)}**" if c else "*(none)*")
                        )
                    entries.append("\n\n".join(lines))

            # Re-render the full feed after every event
            feed.markdown("\n\n---\n\n".join(entries))

        elapsed = time.time() - t_start

        # Final state for this item
        final = hub.current_contribution
        st.success(
            f"{'✅ Converged' if hub.converged else '⏹ Stopped'}  ·  "
            f"{hub.num_submissions} turn(s)  ·  {elapsed:.1f}s"
        )

        # Show final contribution
        if isinstance(final, dict) and final:
            st.markdown("**Final answers:**")
            label_cols = st.columns(min(len(final), 4))
            for col, (fname, val) in zip(label_cols, final.items()):
                display_val = ", ".join(val) if isinstance(val, list) else str(val)
                col.metric(fname, display_val)
        elif isinstance(final, str) and final.strip():
            st.markdown("**Final contribution:**")
            st.info(final)
        else:
            st.warning("No contribution was returned by any agent for this item.")

        # ECU balances
        if hub.ecu_balances:
            st.markdown("**ECU balances:**")
            bal_cols = st.columns(len(hub.ecu_balances))
            for col, (name, bal) in zip(bal_cols, hub.ecu_balances.items()):
                col.metric(name, f"{bal:.3f}")

    # Store result
    result = {
        "item_id": item_id,
        "converged": hub.converged,
        "num_turns": hub.num_submissions,
        "originator_name": hub.originator_name,
        "originator_contribution": hub.originator_contribution,
        "final_contribution": hub.current_contribution,
        "ecu_balances": hub.ecu_balances,
        "coalition_history": item_coalition.to_dict() if item_coalition else {},
        "log": [o.to_dict() for o in hub.log],
        "peer_review_log": [p.to_dict() for p in hub.peer_review_log],
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




# ==================
# TODO:
# - add more LLM providers (Gemini, Anthropic, etc.) - not urgent, to add in the future