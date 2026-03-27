"""Instructions and Prompts Setup Page"""

import streamlit as st
import json
from components.utils import output_fields_from_question_sets

st.set_page_config(page_title="Instructions Setup", page_icon="🧠", layout="wide")

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 3 of 5")
st.sidebar.progress(3 / 5)
st.sidebar.markdown(
    """
**Steps**
1. Task
2. Agents
3. Instructions
4. Dataset
5. Review
"""
)

st.title("Instructions and Prompts")
st.caption("Define base instructions, structured questions, and optional per-agent overrides.")


# ---------- helpers ----------
def init_instructions_state():
    if "experiment_config" not in st.session_state:
        st.session_state.experiment_config = {}

    if "base_instructions" not in st.session_state:
        saved = st.session_state.experiment_config.get("instructions", {})
        st.session_state.base_instructions = saved.get("base_instructions", "")

    if "guideline_notes" not in st.session_state:
        saved = st.session_state.experiment_config.get("instructions", {})
        st.session_state.guideline_notes = saved.get("guideline_notes", "")

    if "question_sets" not in st.session_state:
        st.session_state.question_sets = st.session_state.experiment_config.get("questions", [])

    if "agent_prompt_overrides" not in st.session_state:
        st.session_state.agent_prompt_overrides = {}


def sync_instructions_to_config():
    st.session_state.experiment_config["instructions"] = {
        "base_instructions": st.session_state.get("base_instructions", ""),
        "guideline_notes": st.session_state.get("guideline_notes", ""),
    }
    st.session_state.experiment_config["questions"] = st.session_state.get("question_sets", [])
    st.session_state.experiment_config["agent_prompt_overrides"] = st.session_state.get("agent_prompt_overrides", {})


def build_assembled_prompt(agent_name: str | None = None) -> str:
    """Build a preview of what gets sent to an agent."""
    lines = []

    base = st.session_state.get("base_instructions", "").strip()

    # Apply per-agent override if present
    if agent_name and agent_name in st.session_state.get("agent_prompt_overrides", {}):
        override = st.session_state.agent_prompt_overrides[agent_name].strip()
        if override:
            lines.append(f"[Agent override for {agent_name}]\n{override}")
            lines.append("")

    if base:
        lines.append("[Base instructions]")
        lines.append(base)
        lines.append("")

    notes = st.session_state.get("guideline_notes", "").strip()
    if notes:
        lines.append("[Guideline notes]")
        lines.append(notes)
        lines.append("")

    questions = st.session_state.get("question_sets", [])
    if questions:
        lines.append("[Questions]")
        for i, q in enumerate(questions, 1):
            lines.append(f"Q{i}. [{q.get('field_name', '?')}] {q.get('instruction', '')}")
            for opt in q.get("options", []):
                lines.append(f"   • {opt['code']}: {opt['description']}")
        lines.append("")

    if not lines:
        return "(No instructions or questions defined yet.)"
    return "\n".join(lines)


init_instructions_state()

# ---------- layout ----------
main_col, summary_col = st.columns([2.2, 1], gap="large")

with main_col:

    # ── Section 1: Base instructions ──────────────────────────────────────────
    with st.container(border=True):
        st.subheader("1. Base instructions")
        st.caption("These instructions are shared with all agents on every turn.")

        uploaded_instructions = st.file_uploader(
            "Upload instructions as a .txt file (optional)",
            type=["txt"],
            key="instructions_uploader",
        )
        if uploaded_instructions is not None:
            if st.button("Load instructions from file"):
                try:
                    text = uploaded_instructions.getvalue().decode("utf-8")
                    st.session_state.base_instructions = text
                    st.success("Instructions loaded.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load: {e}")

        st.text_area(
            "Base task instructions",
            placeholder="Write the main instructions all agents should follow.\n\nE.g.: You will be shown an advertisement image and its caption. Classify the ad according to the questions below. Be precise and concise in your reasoning.",
            height=160,
            key="base_instructions",
        )

        st.text_area(
            "Guideline / definition notes (optional)",
            placeholder="Add any term definitions, edge-case rules, or annotation guidelines here.",
            height=100,
            key="guideline_notes",
        )

    # ── Section 2: Question set ────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("2. Question set")
        st.caption("Structured questions define what agents are asked to classify. Upload a JSON file or inspect loaded questions.")

        uploaded_questions = st.file_uploader(
            "Upload question set JSON",
            type=["json"],
            help="Expected format: { \"questions\": [ { \"field_name\": ..., \"field_type\": ..., \"instruction\": ..., \"options\": [...] } ] }",
            key="question_set_uploader",
        )

        if uploaded_questions is not None:
            if st.button("Load questions from file"):
                try:
                    data = json.load(uploaded_questions)
                    if "questions" not in data or not isinstance(data["questions"], list):
                        st.error("Invalid JSON format. Expected a top-level 'questions' list.")
                    else:
                        st.session_state.question_sets = data["questions"]
                        st.session_state.output_fields = output_fields_from_question_sets(data["questions"])
                        st.success(f"Loaded {len(data['questions'])} question(s). Output schema updated.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to load questions: {e}")

        questions = st.session_state.get("question_sets", [])

        if not questions:
            st.info("No questions loaded yet. Upload a JSON file above.")
        else:
            st.success(f"{len(questions)} question(s) loaded.")
            st.divider()

            for i, q in enumerate(questions):
                field_name = q.get("field_name", f"question_{i}")
                field_type = q.get("field_type", "unknown")
                instruction = q.get("instruction", "")
                options = q.get("options", [])

                type_badge = {
                    "single_label": "🔘 Single-label",
                    "multi_label": "☑️ Multi-label",
                    "boolean": "✅ Boolean",
                    "score": "📊 Score",
                    "text": "📝 Text",
                    "ranking": "🔢 Ranking",
                }.get(field_type, field_type)

                with st.expander(f"`{field_name}`  -  {type_badge}", expanded=(i == 0)):
                    st.markdown(f"**Instruction:** {instruction}")
                    if options:
                        st.markdown("**Options:**")
                        for opt in options:
                            st.markdown(f"- `{opt['code']}` - {opt['description']}")

            if st.button("Clear all questions", type="secondary"):
                st.session_state.question_sets = []
                st.session_state.output_fields = []
                st.rerun()

    # ── Section 3: Per-agent overrides ─────────────────────────────────────────
    with st.container(border=True):
        st.subheader("3. Per-agent prompt overrides")
        st.caption(
            "Optionally customize instructions for individual agents. "
            "This is prepended to the base instructions when that agent is called."
        )

        agents = st.session_state.experiment_config.get("agents", [])

        if not agents:
            st.info("No agents configured yet. Set up agents in Step 2 first.")
        else:
            overrides = st.session_state.agent_prompt_overrides

            for agent in agents:
                name = agent.get("name", "?")
                role = agent.get("role", "")
                provider = agent.get("provider", "")
                model = agent.get("model", "")

                with st.expander(f"{name} - {provider} / {model} / {role}", expanded=False):
                    overrides[name] = st.text_area(
                        "Custom system prompt addition (optional)",
                        value=overrides.get(name, ""),
                        key=f"override_{name}",
                        height=100,
                        placeholder=f"E.g.: You are a {role}. Challenge previous classifications if you see strong evidence against them.",
                    )

    # ── Section 4: Assembled prompt preview ────────────────────────────────────
    with st.container(border=True):
        st.subheader("4. Assembled prompt preview")
        st.caption("Preview what will be sent to a selected agent.")

        agents = st.session_state.experiment_config.get("agents", [])
        agent_names = [a["name"] for a in agents] if agents else []

        preview_agent = None
        if agent_names:
            preview_agent = st.selectbox("Preview for agent", ["(Base - no override)"] + agent_names)
            if preview_agent == "(Base - no override)":
                preview_agent = None

        assembled = build_assembled_prompt(preview_agent)
        st.code(assembled, language=None)

    # ── Navigation ─────────────────────────────────────────────────────────────
    nav1, nav2, nav3 = st.columns([2, 2, 0.8])
    with nav1:
        if st.button("← Back"):
            st.switch_page("pages/2_Agent Setup.py")
    with nav2:
        if st.button("Save draft"):
            sync_instructions_to_config()
            st.success("Instructions saved.")
    with nav3:
        if st.button("Next →"):
            sync_instructions_to_config()
            st.switch_page("pages/4_Dataset.py")


with summary_col:
    with st.container(border=True):
        st.subheader("Live summary")

        base = st.session_state.get("base_instructions", "").strip()
        notes = st.session_state.get("guideline_notes", "").strip()
        questions = st.session_state.get("question_sets", [])
        overrides = st.session_state.get("agent_prompt_overrides", {})
        active_overrides = {k: v for k, v in overrides.items() if v.strip()}

        st.markdown(f"**Base instructions**  \n{'✅ Defined' if base else '-'}")
        st.markdown(f"**Guideline notes**  \n{'✅ Defined' if notes else '-'}")
        st.markdown(f"**Questions**  \n{len(questions)} loaded")
        st.markdown(f"**Agent overrides**  \n{len(active_overrides)} active")

        if questions:
            st.divider()
            st.markdown("**Question fields**")
            for q in questions:
                field_type = q.get("field_type", "")
                badge = {
                    "single_label": "🔘",
                    "multi_label": "☑️",
                    "boolean": "✅",
                    "score": "📊",
                    "text": "📝",
                    "ranking": "🔢",
                }.get(field_type, "•")
                st.markdown(f"{badge} `{q.get('field_name', '?')}`")

        if active_overrides:
            st.divider()
            st.markdown("**Overrides active for**")
            for name in active_overrides:
                st.markdown(f"- {name}")

    with st.container(border=True):
        st.subheader("Tips")
        st.markdown(
            """
- Keep base instructions short and task-focused
- Use guideline notes for definitions and edge cases
- Agent overrides let you study role-based bias
- The assembled preview shows exactly what the model receives
"""
        )
