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
        if st.button("→ Start", type="primary", use_container_width=True):
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

                if st.button("→ Load & review", type="primary", use_container_width=True):
                    restore_draft(loaded)
                    st.switch_page("pages/4_Review.py")

            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            st.markdown("")
            st.markdown("")
            st.caption("No file selected yet.")
