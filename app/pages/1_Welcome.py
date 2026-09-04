"""Home — landing page with two paths: new experiment or load draft."""

import json

import streamlit as st
from components.utils import restore_draft


def clear_experiment_state():
    keys_to_clear = [
        "experiment_config",
        "base_instructions", "guideline_notes", "agent_prompt_overrides",
        "agents", "num_agents", "interaction_setting",
        "run_mode", "visibility_mode", "order_type", "custom_order", "max_cycles",
        "stopping_rule", "initializer_agent", "exp_name", "author",
        "task_description", "dataset_df", "column_mapping", "run_results",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


st.title("🧠 Multi-Agent Lab")
st.caption("A configurable platform for studying multi-agent AI interaction, consensus formation, and bias in free-form debates.")

st.divider()

student_id = st.text_input(
    "Student ID",
    value=st.session_state.get("student_id", ""),
    placeholder="e.g. i6123456",
    key="student_id_input",
    help="Identifies your debates and tracks your usage.",
)
st.session_state.student_id = student_id.strip()
if not st.session_state.student_id:
    st.warning("Enter your student ID to continue.")
else:
    try:
        from core.db import COURSE_TOKEN_QUOTA, ensure_schema, get_total_token_usage
        ensure_schema()
        used = get_total_token_usage(st.session_state.student_id)
        st.progress(min(used / COURSE_TOKEN_QUOTA, 1.0))
        st.caption(f"Usage this course: {used:,} / {COURSE_TOKEN_QUOTA:,} tokens")
        if used >= COURSE_TOKEN_QUOTA:
            st.warning("You've used your full course quota. You can still run debates.")
    except Exception:
        # A student not yet in the database shows as 0 usage
        pass

st.divider()

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader("Start new experiment")
        st.markdown("Set up a new experiment from scratch using the step-by-step builder.")
        st.markdown("""
**Steps**
1. Agent setup
2. Instructions & topic
3. Review & launch
""")
        name_input = st.text_input(
            "Experiment name",
            placeholder="e.g. Congestion pricing deliberation — pilot",
            key="main_exp_name",
        )
        author_input = st.text_input(
            "Author(s)",
            placeholder="e.g. Paula, Freija",
            key="main_author",
        )
        st.write("")
        if st.button(
            "→ Start", type="primary", use_container_width=True,
            disabled=not st.session_state.student_id,
        ):
            clear_experiment_state()
            st.session_state.experiment_config = {
                "overview": {"name": name_input, "author": author_input},
                "task": {"description": ""},
                "instructions": {"base_instructions": "", "guideline_notes": ""},
            }
            st.session_state.exp_name = name_input
            st.session_state.author = author_input
            st.switch_page("pages/2_Agent Setup.py")

with right:
    with st.container(border=True):
        st.subheader("Load existing experiment")
        st.markdown(
            "Upload a previously saved experiment JSON to restore all settings "
            "and jump straight to the review page."
        )

        uploaded = st.file_uploader(
            "Upload experiment JSON",
            type=["json"],
            key="main_draft_uploader",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            try:
                raw = uploaded.read()
                loaded = json.loads(raw)
                ov = loaded.get("overview", {})
                exp_name = ov.get("name") or "Unnamed experiment"
                author = ov.get("author") or "—"
                protocol = loaded.get("protocol", {})
                n_agents = len(loaded.get("agents", []))
                saved_at = loaded.get("meta", {}).get("saved_at", "")

                st.info(
                    f"**{exp_name}**  \n"
                    f"Author: {author}  ·  "
                    f"{n_agents} agent(s)  ·  "
                    f"Protocol: {protocol.get('setting', '—')}  \n"
                    + (f"Saved: {saved_at[:10]}" if saved_at else ""),
                    icon="📋",
                )

                if st.button(
                    "→ Load & review", type="primary", use_container_width=True,
                    disabled=not st.session_state.student_id,
                ):
                    restore_draft(loaded)
                    st.switch_page("pages/4_Review.py")

            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            st.markdown("")
            st.markdown("")
            st.caption("No file selected yet.")
