"""Step 4 — Review experiment configuration and launch."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

_AMS = ZoneInfo("Europe/Amsterdam")

import pandas as pd
import streamlit as st
from components.utils import restore_draft

st.set_page_config(page_title="Review Experiment", page_icon="🧠", layout="wide")

st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 4 of 4")
st.sidebar.progress(1.0)
st.sidebar.markdown("""
**Steps**
1. Overview
2. Agents
3. Instructions & topic
4. Review
""")

st.title("Review & Launch")
st.caption("Check your configuration before launching.")


def val(v, fallback="—"):
    if v is None:
        return fallback
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else fallback
    if isinstance(v, str):
        return v.strip() or fallback
    return v


def check(label: str, ok: bool):
    st.markdown(f"{'✅' if ok else '❌'} {label}")


def ensure_single_topic_dataset(cfg: dict) -> None:
    """Keep a one-row internal dataset for the runner."""
    topic = cfg.get("task", {}).get("description", "").strip() or st.session_state.get("task_description", "").strip()
    if topic:
        st.session_state.dataset_df = pd.DataFrame([{"topic": topic}])
        st.session_state.column_mapping = {"topic": "topic"}
        cfg["dataset"] = {"source": "single_topic", "num_rows": 1, "columns": ["topic"]}


def build_draft() -> dict:
    cfg = st.session_state.get("experiment_config", {})
    ensure_single_topic_dataset(cfg)
    return {
        "meta": {"saved_at": datetime.now(_AMS).isoformat(), "app_version": "0.1"},
        "overview": cfg.get("overview", {
            "name": st.session_state.get("exp_name", ""),
            "author": st.session_state.get("author", ""),
            "protocol_id": st.session_state.get("protocol_id", ""),
        }),
        "task": cfg.get("task", {"description": st.session_state.get("task_description", "")}),
        "instructions": cfg.get("instructions", {
            "base_instructions": st.session_state.get("base_instructions", ""),
            "guideline_notes": st.session_state.get("guideline_notes", ""),
        }),
        "agent_prompt_overrides": cfg.get("agent_prompt_overrides", st.session_state.get("agent_prompt_overrides", {})),
        "agents": cfg.get("agents", st.session_state.get("agents", [])),
        "protocol": cfg.get("protocol", {}),
        "ecu": cfg.get("ecu", {}),
        "dataset": cfg.get("dataset", {}),
    }


def readiness_checks(draft: dict) -> tuple[list[str], list[str]]:
    warnings, errors = [], []
    if not draft["overview"].get("name"):
        warnings.append("Experiment name is not set.")
    if not draft["task"].get("description", "").strip():
        errors.append("Deliberation topic/question is not set (Step 3).")
    if not draft["agents"]:
        errors.append("No agents configured (Step 2).")
    if not draft["instructions"].get("base_instructions", "").strip():
        warnings.append("Base instructions are empty (Step 3).")
    return warnings, errors


cfg = st.session_state.get("experiment_config", {})
draft = build_draft()
warnings, errors = readiness_checks(draft)

main_col, action_col = st.columns([2.2, 1], gap="large")

with main_col:
    if errors:
        st.error("**Issues to fix before launching:**\n" + "\n".join(f"- {e}" for e in errors))
    elif warnings:
        st.warning("**Warnings:**\n" + "\n".join(f"- {w}" for w in warnings))
    else:
        st.success("✅ Ready to launch.")

    with st.container(border=True):
        st.subheader("1. Overview")
        ov = draft["overview"]
        topic = draft["task"].get("description", "").strip()
        c1, c2, c3 = st.columns([2.5, 2.5, 1.2])
        c1.markdown(f"**Name**\n\n{val(ov.get('name'))}")
        c2.markdown(f"**Author**\n\n{val(ov.get('author'))}")
        c3.markdown(f"**Version**\n\n{val(ov.get('protocol_id'))}")
        st.divider()
        st.markdown("**Deliberation topic / question**")
        st.info(topic or "No topic set.")

    with st.container(border=True):
        st.subheader("2. Protocol & agents")
        proto = draft["protocol"]

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Setting**\n\n{val(proto.get('setting'))}")
        c2.markdown(f"**Max cycles**\n\n{val(proto.get('max_cycles'))}")
        c3.markdown(f"**Stopping rule**\n\n{val(proto.get('stopping_rule'))}")

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**φ₁ Visibility**\n\n{val(proto.get('visibility_mode'))}")
        c2.markdown(f"**φ₂ Review depth**\n\n{val(proto.get('review_depth'))}")
        c3.markdown(f"**Order**\n\n{val(proto.get('order_type'))}")

        ecu = draft["ecu"]
        if ecu.get("enabled"):
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**ECU**\n\n✅ enabled")
            c2.markdown(f"**T/S/O**\n\n{val(ecu.get('info_condition'))}")
            c3.markdown(f"**Self-assessment**\n\n{'Yes' if ecu.get('include_self_assessment') else 'No'}")
            c4.markdown(f"**Coalition τ**\n\n{val(ecu.get('coalition_threshold'))}")
            if ecu.get("orchestrator_enabled"):
                st.markdown(
                    f"**Orchestrator:** ✅ enabled  ·  "
                    f"ε={ecu.get('orchestrator_step_size', 0.1)}  ·  "
                    f"every {ecu.get('orchestrator_every', 2)} rounds"
                )
            dims = ecu.get("dimensions", [])
            if dims:
                dim_str = "  ·  ".join(
                    f"{d['label']}: SW={d.get('sw_weight', 1.0)}, ECU={d.get('weight', 1.0)}"
                    for d in dims
                )
                st.caption(f"Weights: {dim_str}")
        else:
            st.divider()
            st.markdown("**ECU / peer review:** ❌ disabled")

        agents = draft["agents"]
        if agents:
            st.divider()
            st.markdown(f"**{len(agents)} agent(s)**")
            cols = st.columns(min(len(agents), 3))
            for i, agent in enumerate(agents):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{agent.get('name', f'Agent {i+1}')}**")
                        st.caption(f"`{agent.get('provider', '?')}` / `{agent.get('model', '?')}`")
                        custom_role = agent.get("custom_role", "").strip()
                        if custom_role:
                            st.caption(custom_role)
                        override = draft["agent_prompt_overrides"].get(agent.get("name", ""), "")
                        if override.strip():
                            st.caption("✏️ Has prompt override")

    with st.container(border=True):
        st.subheader("3. Instructions")
        instructions = draft["instructions"]
        base = instructions.get("base_instructions", "").strip()
        notes = instructions.get("guideline_notes", "").strip()
        if base:
            st.markdown("**Base instructions:**")
            st.caption(base[:300] + ("..." if len(base) > 300 else ""))
        else:
            st.warning("No base instructions set.")
        if notes:
            st.caption(f"Guideline notes: {notes[:150]}")

    st.divider()
    nav1, _, nav3 = st.columns([2, 2, 2])
    with nav1:
        if st.button("← Back"):
            st.switch_page("pages/3_Instructions.py")
    with nav3:
        launch_disabled = bool(errors)
        if st.button("🚀 Launch", type="primary", disabled=launch_disabled):
            st.switch_page("pages/5_Run.py")

with action_col:
    with st.container(border=True):
        st.subheader("Readiness")
        check("Experiment name set", bool(draft["overview"].get("name")))
        check("Topic/question set", bool(draft["task"].get("description", "").strip()))
        check("Agents configured", bool(draft["agents"]))
        check("Protocol configured", bool(draft["protocol"]))
        check("Instructions set", bool(draft["instructions"].get("base_instructions", "").strip()))

    with st.container(border=True):
        st.subheader("Save draft")
        st.caption("Save full configuration as JSON.")
        exp_name = draft["overview"].get("name", "experiment").strip() or "experiment"
        timestamp = datetime.now(_AMS).strftime("%Y%m%d_%H%M")
        filename_input = st.text_input(
            "Filename",
            value=f"{exp_name.lower().replace(' ', '_')}_{timestamp}.json",
            key="draft_filename",
        )
        if not filename_input.endswith(".json"):
            filename_input += ".json"
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(draft, indent=2, ensure_ascii=False),
            file_name=filename_input,
            mime="application/json",
            type="primary",
            width="stretch",
        )

    with st.container(border=True):
        st.subheader("Load draft")
        uploaded_draft = st.file_uploader("Upload JSON", type=["json"], key="draft_uploader")
        if uploaded_draft is not None:
            if st.button("Restore", type="secondary"):
                try:
                    loaded = json.load(uploaded_draft)
                    restore_draft(loaded)
                    st.success("Draft restored.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")