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

from app.call_diagnostics import render_call_diagnostics
from core.llm.search import search_suggestions_frame, search_summary
from core.exports.pdf_export import brief_to_pdf_bytes
from core.exports.html_export import run_to_html
from core.protocols.agent_zero_loop import run_agent_zero_experiment

_AMS = ZoneInfo("Europe/Amsterdam")
_ACCENT = "#2E5945"


@st.cache_data(show_spinner=False)
def _cached_brief_pdf(brief_text: str, topic_text: str) -> bytes:
    """Streamlit reruns this page on every widget interaction — cache the PDF
    so it's rendered once per brief instead of on every rerun."""
    return brief_to_pdf_bytes(brief_text, topic_text)

cfg = st.session_state.get("experiment_config")

if not cfg or not cfg.get("task", {}).get("description", "").strip():
    st.error("No topic set yet.")
    if st.button("Back to setup"):
        st.switch_page("pages/1_Welcome.py")
    st.stop()

topic = cfg["task"]["description"]
if cfg.get("shared_documents"):
    with st.expander("Shared source documents"):
        for document in cfg["shared_documents"]:
            st.text(document["name"])
            st.code(document["text"], language=None)

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


def _render_search_summary(agents, searches, enabled=True):
    rows = search_summary(agents, searches, enabled)
    st.dataframe(pd.DataFrame([{k: v for k, v in row.items() if k != "Queries"} for row in rows]),
                 hide_index=True, use_container_width=True)
    with st.expander("Search queries"):
        for row in rows:
            st.markdown(f"**{row['Agent']}**")
            st.code("\n".join(row["Queries"]) or "No queries reported", language=None)
    st.caption("Counts use provider-reported units. Missing counts are not treated as zero.")


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
            st.markdown(design["guideline_notes"])

        if az_cfg.get("web_search_enabled"):
            st.markdown("**Superagent setup search**")
            _render_search_summary(["Agent0"], {"Agent0": design.get("setup_search", {})})
            if design.get("setup_search_error"):
                st.warning("Setup search failed: " + design["setup_search_error"])
            suggestions = search_suggestions_frame(design.get("setup_search", {}))
            if suggestions:
                st.markdown(suggestions, unsafe_allow_html=True)
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
            for issue in round_summary.get("call_errors", []):
                st.warning(f"{issue['agent']} · {issue['phase']}: {issue['error']}")
            st.markdown("**Contributions:**")
            for name, text in round_summary["contributions"].items():
                with st.popover(name, use_container_width=True):
                    st.markdown(text)
                    suggestions = search_suggestions_frame(round_summary.get("web_search", {}).get(name, {}))
                    if suggestions:
                        st.markdown(suggestions, unsafe_allow_html=True)

            if round_summary.get("web_search_enabled") or round_summary.get("web_search"):
                st.markdown("**Web searches this round**")
                _render_search_summary(round_summary["contributions"], round_summary.get("web_search", {}))
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

            coverage = round_summary.get("review_coverage", {})
            if coverage:
                st.markdown("**Valid reviews / expected reviews**, by dimension")
                st.dataframe(pd.DataFrame([
                    {"Agent": name, **{dim: f"{count}/{entry['expected']}"
                     for dim, count in entry["valid_by_dimension"].items()}}
                    for name, entry in coverage.items()]), hide_index=True, use_container_width=True)
                st.caption("Missing scores are unavailable. ECU rewards require a valid mean on every dimension; otherwise balances remain unchanged. Social welfare uses available scores and may have unequal coverage.")

            checks = round_summary.get("peer_review_checks", [])
            if checks:
                with st.expander("Hallucination and risk peer assessments"):
                    st.caption("Peer judgments, not verified facts. Risk scores assess the quality of risk reasoning; higher is better. These checks do not affect ECU rewards.")
                    for check in checks:
                        st.markdown(f"**{check['reviewer']} reviewing {check['agent']}**")
                        if check.get("error"):
                            st.warning(f"Assessment unavailable: {check['error']}")
                            continue
                        hall, risk = check['hallucination'], check['risk']
                        if hall.get("error"):
                            st.warning("Hallucination assessment unavailable: " + hall["error"])
                        else:
                            st.write(f"Hallucination: {hall['status'].replace('_', ' ')}. {hall['explanation']}")
                            for claim in hall['flagged_claims']:
                                st.write(f"Flagged claim: {claim['claim']}")
                                st.caption(claim['reason'])
                        if risk.get("error"):
                            st.warning("Risk assessment unavailable: " + risk["error"])
                        else:
                            score = risk['assessment_quality']
                            st.write(f"Risk assessment quality: {score if score is not None else 'not assessed'}. {risk['explanation']}")
                            for concern in risk['concerns']:
                                st.write(concern)

            weights = round_summary.get("ecu_weights", {})
            if weights:
                weights_str = "  ·  ".join(f"{k}: {v:.3f}" for k, v in weights.items())
                st.caption(f"ECU weights used this round: {weights_str}")

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

    render_call_diagnostics(result.get("debug", {}).get("prompt_log", []), "current_run")
    if st.session_state.get("az_save_error"):
        st.warning(st.session_state.az_save_error)

    timestamp = datetime.now(_AMS).strftime("%Y%m%d_%H%M")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "Download full transcript (.html)",
            data=run_to_html(result, cfg, st.session_state.get("az_save_error")).encode("utf-8"),
            file_name=f"agent0_transcript_{timestamp}.html",
            mime="text/html",
            use_container_width=True,
            help="The complete debate page, including panel, rounds, assessments, brief and expandable API diagnostics.",
        )
    with dl2:
        pdf_bytes = _cached_brief_pdf(result["final_brief"], topic)
        st.download_button(
            "Download brief (.pdf)",
            data=pdf_bytes,
            file_name=f"agent0_brief_{timestamp}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with dl3:
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
            for key in ("experiment_config", "az_result", "az_topic", "az_brief_instructions", "az_pending_calls", "az_save_error", "shared_document_uploads"):
                st.session_state.pop(key, None)
            st.switch_page("pages/1_Welcome.py")


# ---------- launch / display ----------

if "az_result" not in st.session_state:
    launch = st.button("Run debate", type="primary", use_container_width=True)
    if not launch:
        if st.session_state.get("az_pending_calls"):
            render_call_diagnostics(st.session_state.az_pending_calls, "previous_attempt")
        st.stop()

    design_container = st.container()
    rounds_container = st.container()

    def _on_init(design: dict) -> None:
        with design_container:
            _render_design(design)

    from core.db import ensure_schema, begin_experiment, save_call, save_experiment, mark_experiment_failed
    debate_id = None
    st.session_state.pop("az_save_error", None)
    st.session_state.az_pending_calls = []
    try:
        ensure_schema()
        debate_id = begin_experiment(cfg)
    except Exception as exc:
        st.session_state.az_save_error = f"Database logging unavailable; download the full log after the run. ({exc})"
        st.warning(st.session_state.az_save_error)

    def _on_call(call):
        if not any(c["call_id"] == call["call_id"] for c in st.session_state.az_pending_calls):
            st.session_state.az_pending_calls.append(call)
        if call.get("status") == "error":
            st.warning(f"{call.get('agent')} · {call.get('phase')}: {call.get('error')}")
        if debate_id is not None:
            save_call(debate_id, call)

    try:
        with st.spinner("Agent 0 is running the debate..."):
            result = run_agent_zero_experiment(cfg, dry_run=False, on_init=_on_init, on_call=_on_call,
                on_round=lambda rs: _render_round(rs, rounds_container))
    except Exception as exc:
        if debate_id is not None:
            try:
                mark_experiment_failed(debate_id)
            except Exception:
                pass
        st.error(f"The debate stopped unexpectedly: {exc}")
        render_call_diagnostics(st.session_state.az_pending_calls, "interrupted_run")
        st.stop()
    st.session_state.az_result = result
    st.session_state.az_stream_brief = True
    try:
        if debate_id is None:
            ensure_schema()
        save_experiment(result, cfg, debate_id=debate_id)
        st.session_state.pop("az_save_error", None)
    except Exception as exc:
        st.session_state.az_save_error = f"This run was not fully saved. Download the full log to keep all details. ({exc})"

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
