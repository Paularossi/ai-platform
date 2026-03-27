"""
6_Run.py  -  Experiment runner

Streams the gossip protocol live, item by item, agent by agent.
Shows a live feed of each agent turn, then a per-item results table,
and finally lets you download the full results as CSV + JSON.
"""

from __future__ import annotations

import base64
import io
import json
import os
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
from core.protocols.gossip import GossipProtocol, RunEvent
from core.state import ExperimentState, AgentOutput

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


def load_image_b64(image_path: str) -> str | None:
    """Load an image from disk and return as base64 string for the OpenAI API."""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp",
                "gif": "image/gif"}.get(ext, "image/png")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return None


def resolve_image_path(
    item_id: str,
    dataset_cfg: dict,
    zip_bytes: bytes | None,
) -> str | None:
    """
    Try to find the image file on disk.
    Checks the dataset folder path stored in session state.
    Handles the _img suffix mismatch (strips it if needed).
    """
    image_dir = st.session_state.get("dataset_image_dir")
    if not image_dir:
        return None

    # Try exact match first, then strip _img suffix
    stems_to_try = [item_id, item_id.replace("_img", "")]
    extensions = [".png", ".jpg", ".jpeg", ".webp"]

    for stem in stems_to_try:
        for ext in extensions:
            path = os.path.join(image_dir, stem + ext)
            if os.path.exists(path):
                return path
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


def hub_to_experiment_state(hub: CommunicationHub) -> ExperimentState:
    """Convert a finished hub into an ExperimentState for serialisation."""
    state = ExperimentState(
        item_id=hub.item_id,
        item_data=hub.item_data,
        current_labels=hub.current_labels,
        history=hub.log,
        converged=hub.converged,
        originator_name=hub.originator_name,
        originator_labels=hub.originator_labels,
    )
    return state


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
        # Final labels
        for fname, val in r["final_labels"].items():
            row[f"final_{fname}"] = val if not isinstance(val, list) else ", ".join(val)
        # Originator labels
        for fname, val in r["originator_labels"].items():
            row[f"orig_{fname}"] = val if not isinstance(val, list) else ", ".join(val)
        # Did final == originator for each field?
        for fname in r["final_labels"]:
            row[f"changed_{fname}"] = (
                r["final_labels"].get(fname) != r["originator_labels"].get(fname)
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- page ----------

st.title("🚀 Run Experiment")
cfg = get_config()

# ── Guard: check experiment is configured ─────────────────────────────────────
agents_cfg = cfg.get("agents", [])
questions = cfg.get("questions", [])
dataset_cfg = cfg.get("dataset", {})
df: pd.DataFrame | None = st.session_state.get("dataset_df")
column_mapping: dict = st.session_state.get("column_mapping", {})

missing = []
if not agents_cfg:
    missing.append("No agents configured - go to Step 2")
if not questions:
    missing.append("No questions loaded - go to Step 3")
if df is None:
    missing.append("No dataset loaded - go to Step 4")

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

    col1, col2, col3 = st.columns(3)

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
    c1, c2, c3, c4 = st.columns(4)
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

# ── Set API key if provided ───────────────────────────────────────────────────
if not dry_run and api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

# ── Build agents and protocol ─────────────────────────────────────────────────
agents = build_agents(cfg)
protocol = GossipProtocol(agents, cfg, dry_run=dry_run)
agent_names = [a.name for a in agents]
visibility_mode = proto.get("visibility_mode", "Current state only")

# ── Diagnostics ───────────────────────────────────────────────────────────────
with st.expander("🔍 Pre-run diagnostics (expand if labels are empty)", expanded=False):
    st.markdown(f"**Questions loaded:** {len(cfg.get('questions', []))}")
    st.markdown(f"**Agents:** {[a.name for a in agents]}")
    st.markdown(f"**Visibility mode:** {visibility_mode}")
    st.markdown(f"**Column mapping:** {st.session_state.get('column_mapping', {})}")
    if cfg.get("questions"):
        st.markdown("**Question field names:** " + ", ".join(
            f"`{q['field_name']}`" for q in cfg["questions"]
        ))
    else:
        st.error("⚠️ No questions found in experiment config - agents will have nothing structured to answer and labels will be empty. Go back to Step 3 and load your question set JSON.")
    if agents:
        st.markdown("**System prompt preview (Agent 1):**")
        from core.agent import _build_system_prompt
        preview = _build_system_prompt(
            agent_name=agents[0].name,
            agent_role=agents[0].role,
            base_instructions=cfg.get("instructions", {}).get("base_instructions", ""),
            guideline_notes=cfg.get("instructions", {}).get("guideline_notes", ""),
            agent_overrides=cfg.get("agent_prompt_overrides", {}),
            questions=cfg.get("questions", []),
        )
        st.code(preview, language=None)

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
    item_id = str(row.get(
        column_mapping.get("image", column_mapping.get("id", df.columns[0])),
        f"item_{item_idx}"
    ))

    image_path = resolve_image_path(item_id, dataset_cfg, None)
    item_data = build_item_data(row, column_mapping, image_path)

    # Create hub for this item
    hub = CommunicationHub(
        item_id=item_id,
        item_data=item_data,
        visibility_mode=visibility_mode,
        agent_names=agent_names,
    )

    # ── Live turn feed for this item ──────────────────────────────────────
    with st.expander(f"📄 Item {item_idx + 1} / {len(subset)}  -  `{item_id}`", expanded=True):

        feed = st.empty()
        entries: list[str] = []   # accumulated markdown lines rendered all at once

        t_start = time.time()

        for event in protocol.run_iter(hub):

            if event.kind == "dispatch":
                # ── What the hub sent to this agent ──────────────────────
                packet = event.packet
                lines = [f"#### 🔀 Hub → **{event.agent_name}**  ·  Cycle {event.cycle + 1}"]

                if packet.current_labels:
                    label_str = "  ,  ".join(
                        f"`{k}`: {v}" for k, v in packet.current_labels.items()
                    )
                    lines.append(f"**Current labels:** {label_str}")
                else:
                    lines.append("**Current labels:** *(none yet)*")

                if packet.visible_history:
                    lines.append(f"**Visible history** ({len(packet.visible_history)} entry/entries):")
                    for h in packet.visible_history:
                        h_labels = "  ,  ".join(f"`{k}`: {v}" for k, v in h.labels.items())
                        lines.append(
                            f"- Cycle {h.cycle + 1} · **{h.agent_name}**: {h_labels}"
                            + (f"  \n  *{h.reasoning}*" if h.reasoning and h.reasoning != "[dry-run placeholder]" else "")
                        )
                else:
                    lines.append(f"**Visible history:** *(none - visibility mode: {visibility_mode})*")

                entries.append("\n\n".join(lines))

            elif event.kind == "submission":
                # ── What the agent returned to the hub ───────────────────
                output = event.output
                changed_fields = [f for f, c in output.changed.items() if c]
                change_note = f"✏️ revised: {', '.join(changed_fields)}" if changed_fields else "✔ no changes"

                label_str = "  ,  ".join(
                    f"`{k}`: {v}" for k, v in output.labels.items()
                ) if output.labels else "*(no labels parsed)*"

                lines = [f"#### 📨 **{event.agent_name}** → Hub  ·  {change_note}"]
                lines.append(f"**Labels:** {label_str}")

                # Probabilities per field
                if output.probabilities:
                    for fname, probs in output.probabilities.items():
                        conf_val = output.confidence.get(fname)
                        conf_str = f"  *(confidence: {conf_val:.2f})*" if conf_val is not None else ""
                        prob_parts = ", ".join(f"{code}={p:.2f}" for code, p in probs.items())
                        lines.append(f"**{fname} distribution:** {prob_parts}{conf_str}")

                # Pros / cons
                if output.pros:
                    lines.append("**Pros:** " + " · ".join(f"_{p}_" for p in output.pros
                                                            if p != "[dry-run placeholder]"))
                if output.cons:
                    lines.append("**Cons:** " + " · ".join(f"_{c}_" for c in output.cons
                                                            if c != "[dry-run placeholder]"))

                # Warn if nothing was parsed
                if not output.labels and output.raw_response and not output.raw_response.startswith("[dry-run"):
                    lines.append(f"⚠️ **Raw response (parse failed):**\n```\n{output.raw_response}\n```")

                entries.append("\n\n".join(lines))

            # Re-render the full feed after every event
            feed.markdown("\n\n---\n\n".join(entries))

        elapsed = time.time() - t_start

        # Final state for this item
        final_labels = hub.current_labels
        st.success(
            f"{'✅ Converged' if hub.converged else '⏹ Stopped'}  ·  "
            f"{hub.num_submissions} turn(s)  ·  {elapsed:.1f}s"
        )

        # Show final labels
        if final_labels:
            st.markdown("**Final labels:**")
            label_cols = st.columns(min(len(final_labels), 4))
            for col, (fname, val) in zip(label_cols, final_labels.items()):
                display_val = ", ".join(val) if isinstance(val, list) else str(val)
                col.metric(fname, display_val)
        else:
            st.warning("No labels were returned by any agent for this item.")

    # Store result
    result = {
        "item_id": item_id,
        "converged": hub.converged,
        "num_turns": hub.num_submissions,
        "originator_name": hub.originator_name,
        "originator_labels": hub.originator_labels,
        "final_labels": hub.current_labels,
        "log": [o.to_dict() for o in hub.log],
    }
    all_results.append(result)

    status_rows.append({
        "Item": item_id,
        "Converged": "✅" if hub.converged else "⏹",
        "Turns": hub.num_submissions,
        "Time (s)": f"{elapsed:.1f}",
    })

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
# bugs to fix:
# - if i rerun this page it still keeps the results
# - i still get the `use_container_width` will be removed after 2025-12-31.` warning in 4.dataset when mapping the columns
# - change the metadata showing to only 5 rows by default
# - adjust the column widths for some text fields in the Review and Run pages
# - apparently the image is not provided in the prompt ?????????? wtf i just realised