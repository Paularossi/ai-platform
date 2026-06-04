"""Step 1 — Experiment overview."""

import streamlit as st

st.set_page_config(page_title="Multi-Agent Platform", page_icon="🧠", layout="wide")


def init_experiment_config():
    if "experiment_config" not in st.session_state:
        st.session_state.experiment_config = {
            "overview": {"name": "", "author": "", "protocol_id": ""},
            "task": {"description": ""},
            "questions": [],
            "instructions": {"base_instructions": "", "guideline_notes": ""},
        }


def init_state():
    cfg = st.session_state.experiment_config
    if "exp_name" not in st.session_state:
        st.session_state.exp_name = cfg["overview"]["name"]
    if "author" not in st.session_state:
        st.session_state.author = cfg["overview"]["author"]
    if "protocol_id" not in st.session_state:
        st.session_state.protocol_id = cfg["overview"]["protocol_id"]
    if "task_description" not in st.session_state:
        st.session_state.task_description = cfg["task"]["description"]


def sync_to_config():
    st.session_state.experiment_config["overview"] = {
        "name": st.session_state.get("exp_name", ""),
        "author": st.session_state.get("author", ""),
        "protocol_id": st.session_state.get("protocol_id", ""),
    }
    st.session_state.experiment_config["task"] = {
        "description": st.session_state.get("task_description", ""),
    }


init_experiment_config()
init_state()

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 1 of 4")
st.sidebar.progress(1 / 4)
st.sidebar.markdown("""
**Steps**
1. Overview
2. Agents
3. Instructions & topic
4. Review
""")

st.title("Experiment Overview")
st.caption("Name your deliberation experiment.")

main_col, summary_col = st.columns([2.1, 1], gap="large")

with main_col:

    with st.container(border=True):
        st.subheader("1. Overview")
        st.text_input(
            "Experiment name",
            placeholder="e.g. Congestion pricing deliberation — pilot",
            key="exp_name",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Author(s)", placeholder="e.g. Paula, Freija", key="author")
        with c2:
            st.text_input("Version", placeholder="e.g. v0.1", key="protocol_id")

    nav1, nav2, _ = st.columns([2, 2, 4])
    with nav1:
        if st.button("Save draft"):
            sync_to_config()
            st.switch_page("pages/4_Review.py")
    with nav2:
        if st.button("Next →"):
            sync_to_config()
            st.switch_page("pages/2_Agent Setup.py")


with summary_col:
    with st.container(border=True):
        st.subheader("Summary")
        st.markdown(f"**Name**  \n{st.session_state.get('exp_name') or '—'}")
        st.markdown(f"**Author**  \n{st.session_state.get('author') or '—'}")
        st.markdown(f"**Version**  \n{st.session_state.get('protocol_id') or '—'}")
        st.divider()
        st.markdown("**Topic**")
        st.write(st.session_state.get("task_description") or "—")