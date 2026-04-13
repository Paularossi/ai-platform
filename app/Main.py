"""Main entry point — landing page with two paths: new experiment or load draft."""

# select a simple case to show a demo - billboard? communication hub, and illustrate this first solution
# but in the true version we'll have all scenarios
# 1. demo
# 2. make it exciting
# 3. professional use for whatever purposes

import json

import streamlit as st
from components.utils import restore_draft

st.set_page_config(page_title="Multi-Agent Lab", page_icon="🧠", layout="wide")


# ── Clear state helper ────────────────────────────────────────────────────────
def clear_experiment_state():
    """Wipe all experiment-related keys from session state for a clean start."""
    keys_to_clear = [
        "experiment_config", "input_fields", "output_fields", "question_sets",
        "base_instructions", "guideline_notes", "agent_prompt_overrides",
        "agents", "num_agents", "interaction_setting", "platform_mode",
        "supervision_mode", "visibility_mode", "order_type", "max_cycles",
        "stopping_rule", "initializer_agent", "judge_agent",
        "exp_name", "author", "protocol_id", "task_category", "modalities",
        "task_description", "dataset_df", "dataset_filename", "dataset_image_bytes",
        "dataset_all_image_names", "dataset_image_dir", "dataset_image_lookup",
        "column_mapping", "dataset_ready",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


# ── Page ──────────────────────────────────────────────────────────────────────
st.title("🧠 Multi-Agent Lab")
st.caption("A configurable platform for studying multi-agent AI interaction, consensus formation, and bias in structured tasks.")

st.divider()

left, right = st.columns(2, gap="large")

# ── Path A: new experiment ────────────────────────────────────────────────────
with left:
    with st.container(border=True):
        st.subheader("Start new experiment")
        st.markdown(
            "Set up a new experiment from scratch using the step-by-step builder. "
            "Configure your task, agents, protocol, instructions, and dataset."
        )
        st.markdown(
            """
**Steps**
1. Task & schema
2. Agent setup
3. Instructions & prompts
4. Dataset upload
5. Review & launch
"""
        )
        st.write("")
        if st.button("→ New experiment", type="primary", width='stretch'):
            clear_experiment_state()
            st.switch_page("pages/1_Welcome.py")

# ── Path B: load draft ────────────────────────────────────────────────────────
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
            # Show a quick peek at what's inside before committing
            try:
                raw = uploaded.read()
                loaded = json.loads(raw)
                ov = loaded.get("overview", {})
                exp_name = ov.get("name") or "Unnamed experiment"
                author = ov.get("author") or "—"
                protocol = loaded.get("protocol", {})
                n_agents = len(loaded.get("agents", []))
                n_questions = len(loaded.get("questions", []))
                saved_at = loaded.get("meta", {}).get("saved_at", "")

                st.info(
                    f"**{exp_name}**  \n"
                    f"Author: {author}  ·  "
                    f"{n_agents} agent(s)  ·  "
                    f"{n_questions} question(s)  ·  "
                    f"Protocol: {protocol.get('setting', '—')}  \n"
                    + (f"Saved: {saved_at[:10]}" if saved_at else ""),
                    icon="📋",
                )

                if st.button("→ Load & review", type="primary", width='stretch'):
                    restore_draft(loaded)
                    st.switch_page("pages/5_Review.py")

            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            st.markdown("")
            st.markdown("")
            st.caption("No file selected yet.")
