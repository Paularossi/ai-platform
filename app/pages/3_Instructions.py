"""Step 3 — Deliberation topic and agent instructions."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.agent import _build_system_prompt

st.set_page_config(page_title="Instructions", page_icon="🧠", layout="wide")

st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 3 of 4")
st.sidebar.progress(3 / 4)
st.sidebar.markdown("""
**Steps**
1. Overview
2. Agents
3. Instructions & topic
4. Review
""")

st.title("Instructions & topic")
st.caption(
    "Define the deliberation question and the instructions that shape how agents argue."
)


def init_state():
    if "experiment_config" not in st.session_state:
        st.session_state.experiment_config = {}
    cfg = st.session_state.experiment_config
    instr = cfg.get("instructions", {})
    task = cfg.get("task", {})

    if "task_description" not in st.session_state:
        st.session_state.task_description = task.get("description", "")
    if "base_instructions" not in st.session_state:
        st.session_state.base_instructions = instr.get("base_instructions", "")
    if "guideline_notes" not in st.session_state:
        st.session_state.guideline_notes = instr.get("guideline_notes", "")
    if "agent_prompt_overrides" not in st.session_state:
        st.session_state.agent_prompt_overrides = cfg.get("agent_prompt_overrides", {})


def sync_to_config():
    topic = st.session_state.get("task_description", "").strip()
    st.session_state.experiment_config["task"] = {"description": topic}
    st.session_state.experiment_config["instructions"] = {
        "base_instructions": st.session_state.get("base_instructions", ""),
        "guideline_notes": st.session_state.get("guideline_notes", ""),
    }
    st.session_state.experiment_config["agent_prompt_overrides"] = \
        st.session_state.get("agent_prompt_overrides", {})

    # Deliberation runs are single-topic by default. Keep a one-row internal
    # dataframe so the runner can reuse the existing item-processing pipeline
    # without a separate dataset page.
    if topic:
        st.session_state.dataset_df = pd.DataFrame([{"topic": topic}])
        st.session_state.column_mapping = {"topic": "topic"}
        st.session_state.experiment_config["dataset"] = {
            "source": "single_topic",
            "num_rows": 1,
            "columns": ["topic"],
        }


init_state()

main_col, summary_col = st.columns([2.2, 1], gap="large")

with main_col:

    with st.container(border=True):
        st.subheader("0. Deliberation topic / question")
        st.caption("This is the concrete question sent to every agent at runtime.")
        st.text_area(
            "Topic / question",
            placeholder="Should the city introduce congestion charges on its inner ring road?",
            height=90,
            key="task_description",
        )

    with st.container(border=True):
        st.subheader("1. Base instructions")
        st.caption(
            "These instructions are sent to every agent on every turn, before the topic. "
            "Write them in second person, addressed to the agent."
        )

        st.text_area(
            "Base instructions",
            placeholder=(
                "You are a policy advisor participating in a structured deliberation.\n"
                "Write exactly one paragraph (50–150 words) arguing from your assigned perspective.\n"
                "Be specific — cite mechanisms, consequences, or evidence. Do not summarise other agents' views."
            ),
            height=150,
            key="base_instructions",
        )

        st.text_area(
            "Guideline / definition notes (optional)",
            placeholder=(
                "Add definitions, scoring criteria, or factual context here.\n"
                "E.g.: A congestion charge is a fee levied on vehicles entering a defined urban zone."
            ),
            height=90,
            key="guideline_notes",
        )

    with st.container(border=True):
        st.subheader("2. Per-agent prompt overrides")
        st.caption(
            "Optionally give individual agents a custom emphasis. "
            "This text is prepended to the base instructions for that agent only."
        )

        agents = st.session_state.experiment_config.get("agents", [])
        if not agents:
            st.info("No agents configured yet — set up agents in Step 2 first.")
        else:
            overrides = st.session_state.agent_prompt_overrides
            for agent in agents:
                name = agent.get("name", "?")
                role = agent.get("role", "")
                custom_role = agent.get("custom_role", "")
                display_role = custom_role if role == "Custom" and custom_role else role
                with st.expander(f"{name} — {display_role}", expanded=False):
                    overrides[name] = st.text_area(
                        "Custom addition (optional)",
                        value=overrides.get(name, ""),
                        key=f"override_{name}",
                        height=80,
                        placeholder="E.g.: Focus especially on distributional effects.",
                    )

    with st.container(border=True):
        st.subheader("3. Prompt preview")
        st.caption(
            "Shows the system prompt plus the user message structure sent at runtime. "
            "The selected topic is shown below exactly as it will be injected."
        )

        agents = st.session_state.experiment_config.get("agents", [])
        agent_names = [a["name"] for a in agents] if agents else []

        preview_agent = None
        if agent_names:
            sel = st.selectbox("Preview for agent", ["(Base — no override)"] + agent_names)
            if sel != "(Base — no override)":
                preview_agent = sel

        preview_cfg = next((a for a in agents if a["name"] == preview_agent), {}) if preview_agent else {}
        raw_role = preview_cfg.get("role", "Participant")
        custom_role_text = preview_cfg.get("custom_role", "").strip()
        effective_role = custom_role_text if raw_role == "Custom" and custom_role_text else raw_role

        system_prompt = _build_system_prompt(
            agent_name=preview_agent or "Agent",
            agent_role=effective_role,
            base_instructions=st.session_state.get("base_instructions", ""),
            guideline_notes=st.session_state.get("guideline_notes", ""),
            agent_overrides=st.session_state.get("agent_prompt_overrides", {}),
            questions=[],
        )
        st.markdown("**System prompt:**")
        st.code(system_prompt, language=None)

        topic = st.session_state.get("task_description", "").strip() or "{deliberation topic/question}"
        st.markdown("**User message structure:**")
        st.code(
            f"topic: {topic}\n\n"
            "=== Previous round ===       ← shown from round 2 onwards\n"
            "Contributions:\n"
            "  [Agent 1]: ...\n"
            "  [Agent 2]: ...\n"
            "Peer review scores (average received):\n"
            "  Agent 1: depth_breadth: 0.xx, ...\n"
            "=== Your turn ===",
            language=None,
        )

    nav1, nav2, nav3 = st.columns([2, 2, 0.8])
    with nav1:
        if st.button("← Back"):
            st.switch_page("pages/2_Agent Setup.py")
    with nav2:
        if st.button("Save draft"):
            sync_to_config()
            st.success("Saved.")
    with nav3:
        if st.button("Next →"):
            sync_to_config()
            st.switch_page("pages/4_Review.py")


with summary_col:
    with st.container(border=True):
        st.subheader("Summary")
        topic = st.session_state.get("task_description", "").strip()
        base = st.session_state.get("base_instructions", "").strip()
        notes = st.session_state.get("guideline_notes", "").strip()
        overrides = st.session_state.get("agent_prompt_overrides", {})
        active = {k: v for k, v in overrides.items() if v.strip()}
        st.markdown(f"**Topic**  \n{'✅ Set' if topic else '—'}")
        st.markdown(f"**Base instructions**  \n{'✅ Set' if base else '—'}")
        st.markdown(f"**Guideline notes**  \n{'✅ Set' if notes else '—'}")
        st.markdown(f"**Agent overrides**  \n{len(active)} active")