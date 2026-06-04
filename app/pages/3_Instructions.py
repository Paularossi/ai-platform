"""Step 3 — Instructions, questions, and per-agent overrides."""

import sys
import json
from pathlib import Path

import streamlit as st

# Ensure core/ is importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.agent import _build_system_prompt, _is_classification_task

st.set_page_config(page_title="Instructions Setup", page_icon="🧠", layout="wide")

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 3 of 5")
st.sidebar.progress(3 / 5)
st.sidebar.markdown("""
**Steps**
1. Task
2. Agents
3. Instructions
4. Dataset
5. Review
""")

st.title("Instructions")
st.caption("Define what agents are told and, for classification tasks, what questions they answer.")


# ---------- helpers ----------
def init_state():
    if "experiment_config" not in st.session_state:
        st.session_state.experiment_config = {}

    instr = st.session_state.experiment_config.get("instructions", {})
    if "base_instructions" not in st.session_state:
        st.session_state.base_instructions = instr.get("base_instructions", "")
    if "guideline_notes" not in st.session_state:
        st.session_state.guideline_notes = instr.get("guideline_notes", "")
    if "question_sets" not in st.session_state:
        st.session_state.question_sets = st.session_state.experiment_config.get("questions", [])
    if "agent_prompt_overrides" not in st.session_state:
        st.session_state.agent_prompt_overrides = st.session_state.experiment_config.get(
            "agent_prompt_overrides", {}
        )


def sync_to_config():
    st.session_state.experiment_config["instructions"] = {
        "base_instructions": st.session_state.get("base_instructions", ""),
        "guideline_notes": st.session_state.get("guideline_notes", ""),
    }
    st.session_state.experiment_config["questions"] = st.session_state.get("question_sets", [])
    st.session_state.experiment_config["agent_prompt_overrides"] = st.session_state.get(
        "agent_prompt_overrides", {}
    )


init_state()

task_mode = st.session_state.experiment_config.get("task", {}).get("mode", "deliberation")
is_classification = task_mode == "classification"

main_col, summary_col = st.columns([2.2, 1], gap="large")

with main_col:

    # ── 1. Base instructions ──────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("1. Base instructions")
        st.caption("Shared with all agents on every turn, before any questions or history.")

        uploaded_txt = st.file_uploader(
            "Upload from .txt file (optional)", type=["txt"], key="instructions_uploader"
        )
        if uploaded_txt is not None:
            if st.button("Load from file"):
                try:
                    st.session_state.base_instructions = uploaded_txt.getvalue().decode("utf-8")
                    st.success("Instructions loaded.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load: {e}")

        if is_classification:
            placeholder = (
                "Write the main instructions all agents should follow.\n\n"
                "E.g.: You will be shown an advertisement image and its caption. "
                "Classify the ad according to the questions below. "
                "Be precise and concise in your reasoning."
            )
        else:
            placeholder = (
                "Write the main instructions all agents should follow.\n\n"
                "E.g.: You will be given the name of a country. "
                "Produce a single sentence that best describes this country. "
                "Be accurate and neutral."
            )

        st.text_area(
            "Base instructions",
            placeholder=placeholder,
            height=160,
            key="base_instructions",
        )

        st.text_area(
            "Guideline / definition notes (optional)",
            placeholder="Add term definitions, edge-case rules, or scoring criteria here.",
            height=100,
            key="guideline_notes",
        )

    # ── 2. Question set (classification only) ─────────────────────────────────
    if is_classification:
        with st.container(border=True):
            st.subheader("2. Question set")
            st.caption(
                "Upload a JSON file defining the classification fields and option codes. "
                "This determines the structured output format agents produce."
            )

            with st.expander("Expected JSON format", expanded=False):
                st.code(
                    '{\n'
                    '  "questions": [\n'
                    '    {\n'
                    '      "field_name": "target_age",\n'
                    '      "field_type": "single_label",\n'
                    '      "instruction": "Who is this ad primarily targeting?",\n'
                    '      "options": [\n'
                    '        {"code": "CHILD", "description": "Children under 12"},\n'
                    '        {"code": "ADOLESCENT", "description": "Teens 12-17"},\n'
                    '        {"code": "ADULT", "description": "Adults 18+"}\n'
                    '      ]\n'
                    '    }\n'
                    '  ]\n'
                    '}',
                    language="json",
                )
                st.caption(
                    "Supported field types: `single_label`, `multi_label`, "
                    "`boolean`, `score`, `text`"
                )

            uploaded_q = st.file_uploader(
                "Upload question set JSON",
                type=["json"],
                key="question_set_uploader",
            )

            if uploaded_q is not None:
                if st.button("Load questions", type="primary"):
                    try:
                        data = json.load(uploaded_q)
                        if not isinstance(data.get("questions"), list):
                            st.error("Expected a top-level 'questions' list.")
                        else:
                            st.session_state.question_sets = data["questions"]
                            st.success(f"Loaded {len(data['questions'])} question(s).")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load: {e}")

            questions = st.session_state.get("question_sets", [])

            if not questions:
                st.info("No questions loaded yet.")
            else:
                st.success(f"{len(questions)} question(s) loaded.")
                st.divider()

                TYPE_BADGE = {
                    "single_label": "🔘 Single-label",
                    "multi_label": "☑️ Multi-label",
                    "boolean": "✅ Boolean",
                    "score": "📊 Score",
                    "text": "📝 Text",
                }

                for i, q in enumerate(questions):
                    fname = q.get("field_name", f"q_{i}")
                    ftype = q.get("field_type", "single_label")
                    badge = TYPE_BADGE.get(ftype, ftype)
                    with st.expander(f"`{fname}`  —  {badge}", expanded=(i == 0)):
                        st.markdown(f"**Instruction:** {q.get('instruction', '')}")
                        opts = q.get("options", [])
                        if opts:
                            st.markdown("**Options:**")
                            for opt in opts:
                                st.markdown(f"- `{opt['code']}` — {opt['description']}")

                if st.button("Clear questions", type="secondary"):
                    st.session_state.question_sets = []
                    st.rerun()

    else:
        # Deliberation mode — no structured questions
        with st.container(border=True):
            st.subheader("2. Output format")
            st.info(
                "This is a **deliberation task** — agents produce free-text contributions. "
                "No question set is needed.\n\n"
                "Each agent turn produces: a **contribution** (statement), "
                "a **confidence** score (0–1), and **pros / cons** (reasoning).",
                icon="💬",
            )

    # ── 3. Per-agent overrides ────────────────────────────────────────────────
    with st.container(border=True):
        section_num = "3" if is_classification else "3"
        st.subheader(f"{section_num}. Per-agent prompt overrides")
        st.caption(
            "Give individual agents a different persona or emphasis. "
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
                model = agent.get("model", "")
                with st.expander(f"{name} — {model} / {role}", expanded=False):
                    overrides[name] = st.text_area(
                        "Custom prompt addition (optional)",
                        value=overrides.get(name, ""),
                        key=f"override_{name}",
                        height=90,
                        placeholder=(
                            "E.g.: You are a sceptical reviewer. "
                            "Challenge previous answers if you see strong evidence against them."
                        ),
                    )

    # ── 4. Prompt preview ─────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("4. Assembled prompt preview")
        st.caption(
            "Shows the exact system prompt that will be sent to the selected agent. "
            "Built directly from your instructions, questions, and overrides."
        )

        agents = st.session_state.experiment_config.get("agents", [])
        agent_names = [a["name"] for a in agents] if agents else []

        preview_agent = None
        if agent_names:
            sel = st.selectbox("Preview for agent", ["(Base — no override)"] + agent_names)
            if sel != "(Base — no override)":
                preview_agent = sel

        # Use agent.py's actual builder so the preview matches reality
        preview_agent_cfg = next(
            (a for a in agents if a["name"] == preview_agent), {}
        ) if preview_agent else {}

        # Resolve effective role (same logic as Agent.__init__)
        raw_role = preview_agent_cfg.get("role", "Participant")
        custom_role_text = preview_agent_cfg.get("custom_role", "").strip()
        effective_role = custom_role_text if raw_role == "Custom" and custom_role_text else raw_role

        # Build preview using actual _build_system_prompt so it matches runtime exactly
        assembled = _build_system_prompt(
            agent_name=preview_agent or "Agent",
            agent_role=effective_role,
            base_instructions=st.session_state.get("base_instructions", ""),
            guideline_notes=st.session_state.get("guideline_notes", ""),
            agent_overrides=st.session_state.get("agent_prompt_overrides", {}),
            questions=st.session_state.get("question_sets", []) if is_classification else [],
        )

        st.code(assembled, language=None)
        st.caption(
            "This is the **system prompt** sent to the agent on every turn. "
            "The actual item (e.g. the question or text to deliberate on) is injected "
            "separately at runtime from your dataset — it does not appear here."
        )

        # Show what the user message will look like (template only)
        st.markdown("**User message template (runtime):**")
        st.code(
            "{item_data}\n\n"
            "=== Previous round ===  ← shown from round 2 onwards\n"
            "\n"
            "Contributions:\n"
            "  [Agent 1]: ...\n"
            "  [Agent 2]: ...\n"
            "\n"
            "Peer review scores (average received):\n"
            "  Agent 1: depth_breadth: 0.xx, depth: 0.xx, ...\n"
            "\n"
            "=== Your turn ===",
            language=None,
        )

    # ── Navigation ─────────────────────────────────────────────────────────────
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
            st.switch_page("pages/4_Dataset.py")


with summary_col:
    with st.container(border=True):
        st.subheader("Live summary")
        base = st.session_state.get("base_instructions", "").strip()
        notes = st.session_state.get("guideline_notes", "").strip()
        questions = st.session_state.get("question_sets", [])
        overrides = st.session_state.get("agent_prompt_overrides", {})
        active_overrides = {k: v for k, v in overrides.items() if v.strip()}

        st.markdown(f"**Task mode**  \n{'Classification' if is_classification else 'Deliberation'}")
        st.markdown(f"**Base instructions**  \n{'✅ Defined' if base else '—'}")
        st.markdown(f"**Guideline notes**  \n{'✅ Defined' if notes else '—'}")
        if is_classification:
            st.markdown(f"**Questions**  \n{len(questions)} loaded")
        st.markdown(f"**Agent overrides**  \n{len(active_overrides)} active")

        if is_classification and questions:
            st.divider()
            st.markdown("**Fields**")
            TYPE_ICON = {
                "single_label": "🔘", "multi_label": "☑️", "boolean": "✅",
                "score": "📊", "text": "📝",
            }
            for q in questions:
                icon = TYPE_ICON.get(q.get("field_type", ""), "•")
                st.markdown(f"{icon} `{q.get('field_name', '?')}`")

        if active_overrides:
            st.divider()
            st.markdown("**Overrides active for**")
            for name in active_overrides:
                st.markdown(f"- {name}")