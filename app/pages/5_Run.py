"""
5_Run.py — Agent 0 mode: runs the debate, streams round-by-round progress,
and shows the final policy brief.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.protocols.agent_zero_loop import run_agent_zero_experiment

_AMS = ZoneInfo("Europe/Amsterdam")

cfg = st.session_state.get("experiment_config")

if not cfg or not cfg.get("task", {}).get("description", "").strip():
    st.error("No topic set yet.")
    if st.button("← Back to Setup"):
        st.switch_page("pages/1_Welcome.py")
    st.stop()

topic = cfg["task"]["description"]
az_cfg = cfg.get("agent_zero", {})

st.title("🚀 Agent 0 Debate")
with st.container(border=True):
    st.markdown(f"**Topic:** {topic}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agent 0", f"{az_cfg.get('provider', '?')}")
    c2.metric("Model", az_cfg.get("model", "?"))
    c3.metric("Max rounds", az_cfg.get("max_rounds", "?"))
    c4.metric("Max agents", az_cfg.get("max_agents", "?"))


# ---------- results renderer ----------

def _render_design(design: dict) -> None:
    """Show what Agent 0 set up, before the debate itself starts."""
    with st.expander("🧭 Agent 0's design", expanded=True):
        agents = design.get("agents", [])
        st.markdown(f"**Roster ({len(agents)}):**")
        for a in agents:
            st.markdown(f"- **{a['name']}** ({a['provider']}/{a['model']}): {a['role']}")

        st.markdown("**Base instructions:**")
        st.caption(design.get("base_instructions", ""))
        if design.get("guideline_notes"):
            st.markdown("**Guideline notes:**")
            st.caption(design["guideline_notes"])

        st.markdown("**Quality dimensions:**")
        dims = design.get("ecu_dimensions", [])
        if dims:
            st.dataframe(
                pd.DataFrame([
                    {"Dimension": d["label"], "Rubric": d["rubric"],
                     "ECU weight": d["weight"], "SW weight": d["sw_weight"]}
                    for d in dims
                ]),
                hide_index=True, use_container_width=True,
                column_config={"Rubric": st.column_config.TextColumn(width="large")},
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("φ1 visibility", design.get("visibility_mode", "—"))
        c2.metric("φ2 review depth", design.get("review_depth", "—"))
        c3.metric("ECU info condition", design.get("info_condition", "—"))
        st.caption(f"Coalition threshold τ = {design.get('coalition_threshold', '—')}")

        if design.get("reasoning"):
            st.markdown("**Agent 0's reasoning for this setup:**")
            st.caption(design["reasoning"])


def _render_round(round_summary: dict, container) -> None:
    cycle = round_summary["cycle"]
    roster = [a["name"] for a in round_summary["roster"]]
    sw = round_summary.get("social_welfare")
    label = f"Round {cycle + 1}  ·  {len(roster)} agent(s)"
    if sw is not None:
        label += f"  ·  SW {sw:.3f}"

    with container:
        with st.status(label, state="complete", expanded=(cycle == 0)):
            st.markdown(f"**Roster:** {', '.join(roster)}")
            st.markdown("**Contributions:**")
            for name, text in round_summary["contributions_preview"].items():
                st.markdown(f"- **{name}:** {text}")

            dim_scores = round_summary.get("dimension_scores", {})
            if dim_scores:
                st.markdown("**Peer review scores** (mean per dimension, 0-1):")
                dim_names = sorted({d for scores in dim_scores.values() for d in scores})
                rows = []
                earned = round_summary.get("ecu_earned_this_round", {})
                for agent, scores in dim_scores.items():
                    row = {"Agent": agent, **{d: scores.get(d, "—") for d in dim_names}}
                    row["ECU earned"] = earned.get(agent, "—")
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            coalition = round_summary.get("coalition", [])
            if len(coalition) >= 2:
                st.markdown(f"**Coalition:** {', '.join(coalition)}")
            else:
                st.caption("Coalition: none")

            if round_summary["agents_added"] or round_summary["agents_removed"]:
                st.markdown(
                    f"**Agent 0 roster edit:** "
                    f"+{round_summary['agents_added'] or '—'}  "
                    f"−{round_summary['agents_removed'] or '—'}"
                )
            for name, instr in round_summary["agent_instructions_issued"].items():
                st.markdown(f"**Agent 0 → {name}:** {instr}")

            st.markdown("**Agent 0's reasoning:**")
            st.caption(round_summary["agent_zero_reasoning"])


def _show_result(result: dict) -> None:
    st.divider()
    st.subheader("📋 Final policy brief")
    with st.container(border=True):
        st.markdown(result["final_brief"])

    n_rounds = len(result.get("rounds", []))
    st.caption(
        f"Ended: **{result['ended_reason']}**  ·  {n_rounds} round(s)  ·  "
        f"{result['num_turns']} turn(s)"
    )

    with st.expander("💰 Final ECU balances"):
        balances = result.get("final_ecu_balances", {})
        if balances:
            cols = st.columns(len(balances))
            for col, (name, bal) in zip(cols, balances.items()):
                col.metric(name, f"{bal:.3f} ecus")

    timestamp = datetime.now(_AMS).strftime("%Y%m%d_%H%M")
    dl1, dl2 = st.columns(2)
    with dl1:
        brief_bytes = result["final_brief"].encode()
        st.download_button(
            "⬇ Final brief (.md)",
            data=brief_bytes,
            file_name=f"agent0_brief_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with dl2:
        full_json = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "⬇ Full log JSON",
            data=full_json.encode(),
            file_name=f"agent0_log_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
            help="'rounds' is the readable per-round record; 'debug' has raw prompts/scores.",
        )


def _show_nav_buttons() -> None:
    st.divider()
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("🔄 Restart debate", use_container_width=True,
                     help="Run the same topic and settings again."):
            st.session_state.pop("az_result", None)
            st.rerun()
    with nav2:
        if st.button("✨ New topic", use_container_width=True, type="primary"):
            for key in ("experiment_config", "az_result", "az_topic"):
                st.session_state.pop(key, None)
            st.switch_page("pages/1_Welcome.py")


# ---------- launch / display ----------

if "az_result" not in st.session_state:
    launch = st.button("▶ Run debate", type="primary", use_container_width=True)
    if not launch:
        st.stop()

    design_container = st.container()
    rounds_container = st.container()

    def _on_init(design: dict) -> None:
        with design_container:
            _render_design(design)

    with st.spinner("Agent 0 is designing the debate…"):
        result = run_agent_zero_experiment(
            cfg, dry_run=False,
            on_init=_on_init,
            on_round=lambda rs: _render_round(rs, rounds_container),
        )
    st.session_state.az_result = result
else:
    # Cached result (e.g. after a download-button rerun) — design wasn't just rendered live, so show it here instead.
    result = st.session_state.az_result
    if result.get("agent_zero_design"):
        _render_design(result["agent_zero_design"])

_show_result(st.session_state.az_result)
_show_nav_buttons()
