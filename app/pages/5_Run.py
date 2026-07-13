"""Run page for Agent 0 mode.

Runs the debate, shows what Agent 0 set up before the first round, streams
each round as it completes, then shows the final policy brief.
"""

from __future__ import annotations

import html
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pdf_export import brief_to_pdf_bytes
from core.protocols.agent_zero_loop import run_agent_zero_experiment

_AMS = ZoneInfo("Europe/Amsterdam")
_ACCENT = "#2E5945"

cfg = st.session_state.get("experiment_config")

if not cfg or not cfg.get("task", {}).get("description", "").strip():
    st.error("No topic set yet.")
    if st.button("Back to setup"):
        st.switch_page("pages/1_Welcome.py")
    st.stop()

topic = cfg["task"]["description"]
az_cfg = cfg.get("agent_zero", {})

st.title("Debate")
st.markdown(
    f"""
    <div style="border-left: 4px solid {_ACCENT}; padding: 0.85rem 1.25rem;
                margin-bottom: 1rem; background: rgba(46,89,69,0.06); border-radius: 4px;">
        <div style="font-size:0.75rem; letter-spacing:0.08em; text-transform:uppercase;
                    color:{_ACCENT}; font-weight:600; margin-bottom:0.3rem;">Topic</div>
        <div style="font-family:'Lora',serif; font-size:1.35rem; font-style:italic;
                    line-height:1.4;">{html.escape(topic)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([2, 3, 1, 1])
    c1.metric("Moderator", az_cfg.get("provider", "-"))
    c2.metric("Model", az_cfg.get("model", "-"))
    c3.metric("Max rounds", az_cfg.get("max_rounds", "-"))
    c4.metric("Max panel size", az_cfg.get("max_agents", "-"))


# ---------- renderers ----------

def _reasoning_box(label: str, text: str) -> None:
    """Styled callout for Agent 0's own reasoning, distinct from round content."""
    st.markdown(
        f"""
        <div style="border-left: 3px solid {_ACCENT}; padding: 0.6rem 1rem;
                    margin-top: 0.6rem; background: rgba(46,89,69,0.05); border-radius: 4px;">
            <div style="font-size:0.72rem; letter-spacing:0.06em; text-transform:uppercase;
                        color:{_ACCENT}; font-weight:600; margin-bottom:0.25rem;">{html.escape(label)}</div>
            <div style="font-style: italic; line-height:1.5;">{html.escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_design(design: dict) -> None:
    """Show what Agent 0 set up, before the debate itself starts."""
    with st.expander("Panel and rules", expanded=True):
        agents = design.get("agents", [])
        st.markdown(f"**Panel ({len(agents)}):**")
        for a in agents:
            st.markdown(f"- **{a['name']}** ({a['provider']}/{a['model']}): {a['role']}")

        st.markdown("**Instructions given to every agent:**")
        st.caption(design.get("base_instructions", ""))
        if design.get("guideline_notes"):
            st.markdown("**Shared context:**")
            st.caption(design["guideline_notes"])

        st.markdown("**Scoring criteria:**")
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
        c1.metric("Visibility (φ1)", design.get("visibility_mode", "-"))
        c2.metric("Review depth (φ2)", design.get("review_depth", "-"))
        c3.metric("ECU visibility", design.get("info_condition", "-"))
        st.caption(f"Coalition threshold: {design.get('coalition_threshold', '-')}")

        if design.get("reasoning"):
            _reasoning_box("Why Agent 0 set it up this way", design["reasoning"])


def _render_round(round_summary: dict, container) -> None:
    cycle = round_summary["cycle"]
    roster = [a["name"] for a in round_summary["roster"]]
    sw = round_summary.get("social_welfare")
    label = f"Round {cycle + 1} · {len(roster)} agents"
    if sw is not None:
        label += f" · social welfare {sw:.3f}"

    with container:
        with st.status(label, state="complete", expanded=(cycle == 0)):
            st.markdown(f"**Panel:** {', '.join(roster)}")
            st.markdown("**Contributions:**")
            for name, text in round_summary["contributions"].items():
                with st.popover(name, use_container_width=True):
                    st.markdown(text)

            dim_scores = round_summary.get("dimension_scores", {})
            if dim_scores:
                st.markdown("**Peer review scores** (mean per dimension, 0 to 1):")
                dim_names = sorted({d for scores in dim_scores.values() for d in scores})
                rows = []
                earned = round_summary.get("ecu_earned_this_round", {})
                for agent, scores in dim_scores.items():
                    row = {"Agent": agent, **{d: scores.get(d, "-") for d in dim_names}}
                    row["ECU earned"] = earned.get(agent, "-")
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            weights = round_summary.get("ecu_weights", {})
            if weights:
                weights_str = "  ·  ".join(f"{k}: {v:.3f}" for k, v in weights.items())
                st.caption(f"ECU weights after this round: {weights_str}")

            coalition = round_summary.get("coalition", [])
            if len(coalition) >= 2:
                st.markdown(f"**Coalition:** {', '.join(coalition)}")
            else:
                st.caption("Coalition: none")

            if round_summary["agents_added"] or round_summary["agents_removed"]:
                st.markdown(
                    f"**Panel change:** added {round_summary['agents_added'] or 'none'}, "
                    f"removed {round_summary['agents_removed'] or 'none'}"
                )
            for name, instr in round_summary["agent_instructions_issued"].items():
                st.markdown(f"**Note to {name}:** {instr}")

            _reasoning_box("Agent 0's reasoning", round_summary["agent_zero_reasoning"])


def _stream_words(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.012)


def _show_result(result: dict) -> None:
    st.divider()
    st.subheader("Policy brief")
    with st.container(border=True):
        if st.session_state.pop("az_stream_brief", False):
            st.write_stream(_stream_words(result["final_brief"]))
        else:
            st.markdown(result["final_brief"])

    n_rounds = len(result.get("rounds", []))
    st.caption(
        f"Ended: **{result['ended_reason']}** · {n_rounds} rounds · "
        f"{result['num_turns']} turns"
    )

    with st.expander("ECU balances"):
        balances = result.get("final_ecu_balances", {})
        if balances:
            cols = st.columns(len(balances))
            for col, (name, bal) in zip(cols, balances.items()):
                col.metric(name, f"{bal:.3f}")

    timestamp = datetime.now(_AMS).strftime("%Y%m%d_%H%M")
    dl1, dl2, dl3, dl4 = st.columns(4)
    with dl1:
        brief_bytes = result["final_brief"].encode()
        st.download_button(
            "Download brief (.md)",
            data=brief_bytes,
            file_name=f"agent0_brief_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with dl2:
        pdf_bytes = brief_to_pdf_bytes(result["final_brief"], topic)
        st.download_button(
            "Download brief (.pdf)",
            data=pdf_bytes,
            file_name=f"agent0_brief_{timestamp}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with dl3:
        transcript = {
            "topic": topic,
            "rounds": [
                {"round": r["cycle"] + 1, "contributions": r["contributions"]}
                for r in result.get("rounds", [])
            ],
        }
        transcript_json = json.dumps(transcript, indent=2, ensure_ascii=False)
        st.download_button(
            "Download transcript (.json)",
            data=transcript_json.encode(),
            file_name=f"agent0_transcript_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
            help="Just what each agent said, round by round. No scores, no reasoning, no metadata.",
        )
    with dl4:
        full_json = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "Download full log (.json)",
            data=full_json.encode(),
            file_name=f"agent0_log_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
            help="'rounds' holds the readable per-round record; 'debug' holds raw prompts and scores.",
        )


def _show_nav_buttons() -> None:
    st.divider()
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("Run again", use_container_width=True,
                     help="Run the same topic and settings again."):
            st.session_state.pop("az_result", None)
            st.rerun()
    with nav2:
        if st.button("Start new debate", use_container_width=True, type="primary"):
            for key in ("experiment_config", "az_result", "az_topic"):
                st.session_state.pop(key, None)
            st.switch_page("pages/1_Welcome.py")


# ---------- launch / display ----------

if "az_result" not in st.session_state:
    launch = st.button("Run debate", type="primary", use_container_width=True)
    if not launch:
        st.stop()

    design_container = st.container()
    rounds_container = st.container()

    def _on_init(design: dict) -> None:
        with design_container:
            _render_design(design)

    with st.spinner("Agent 0 is setting up the debate..."):
        result = run_agent_zero_experiment(
            cfg, dry_run=False,
            on_init=_on_init,
            on_round=lambda rs: _render_round(rs, rounds_container),
        )
    st.session_state.az_result = result
    st.session_state.az_stream_brief = True  # animate the brief once, right after this run

    try:
        from core.db import ensure_schema, save_experiment
        ensure_schema()
        save_experiment(result, cfg)
    except Exception as exc:
        # Never let a database hiccup take down a finished debate - the
        # person still gets their brief and downloads either way.
        st.caption(f"Note: this run wasn't saved to the shared log ({exc}).")
else:
    # Cached result (e.g. after a download-button rerun) - the live callbacks
    # that rendered the design panel and each round only fire during the run
    # itself, so on any later rerun they have to be rebuilt from the stored
    # result instead, or they'd vanish the moment a download button is clicked.
    result = st.session_state.az_result
    if result.get("agent_zero_design"):
        _render_design(result["agent_zero_design"])
    rounds_container = st.container()
    for round_summary in result.get("rounds", []):
        _render_round(round_summary, rounds_container)

_show_result(st.session_state.az_result)
_show_nav_buttons()
