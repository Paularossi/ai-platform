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

st.session_state.setdefault("id_valid", False)
st.session_state.setdefault("is_tutor", False)
st.session_state.setdefault("tutorial_group", None)

with st.form("login_form"):
    student_id_input = st.text_input(
        "Student or tutor ID",
        value=st.session_state.get("student_id", ""),
        placeholder="e.g. i6123456",
        help="Checked against the course roster. Ask your tutor if yours isn't recognized.",
    )
    logged_in = st.form_submit_button("Log in")

if logged_in:
    st.session_state.student_id = student_id_input.strip()
    st.session_state.id_valid = False
    st.session_state.is_tutor = False
    st.session_state.tutorial_group = None

    sid = st.session_state.student_id
    if not sid:
        st.warning("Enter your student ID to continue.")
    else:
        try:
            from core.db import ensure_schema, get_student_group, get_total_token_usage, is_tutor
            ensure_schema()
            if is_tutor(sid):
                st.session_state.id_valid = True
                st.session_state.is_tutor = True
                st.info("Tutor ID recognized. The tutor dashboard is coming soon — for now you can also run debates below.")
            else:
                group = get_student_group(sid)
                if group is not None:
                    st.session_state.id_valid = True
                    st.session_state.tutorial_group = group
                else:
                    st.error("ID not recognized. Check with your tutor if you think this is a mistake.")

            if st.session_state.id_valid:
                st.session_state.token_usage = get_total_token_usage(sid)
                # rerun to show the tutor page
                st.rerun()
        except Exception as exc:
            st.error(f"Can't verify your ID right now ({exc}). Please try again in a moment.")

elif st.session_state.get("id_valid"):
    st.success(f"Logged in as {st.session_state.student_id}" + (" (tutor)" if st.session_state.is_tutor else ""))

if st.session_state.get("id_valid"):
    from core.db import COURSE_TOKEN_QUOTA
    used = st.session_state.get("token_usage", 0)
    st.progress(min(used / COURSE_TOKEN_QUOTA, 1.0))
    st.caption(f"Usage this course: {used:,} / {COURSE_TOKEN_QUOTA:,} tokens")
    if used >= COURSE_TOKEN_QUOTA:
        st.warning("You've used your full course quota. You can still run debates.")

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
            disabled=not st.session_state.id_valid,
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
                    disabled=not st.session_state.id_valid,
                ):
                    restore_draft(loaded)
                    st.switch_page("pages/4_Review.py")

            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            st.markdown("")
            st.markdown("")
            st.caption("No file selected yet.")
