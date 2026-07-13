"""History page - browse past Agent 0 debates saved to the shared log."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.title("Past debates")
st.caption("Every debate run from this app is logged here, shared across everyone using it.")

try:
    from core.db import ensure_schema, get_debate_brief, list_debates
    ensure_schema()
    debates = list_debates(limit=100)
except Exception as exc:
    st.error(f"Couldn't reach the shared log: {exc}")
    st.stop()

if not debates:
    st.info("No debates logged yet. Run one from the Setup page.")
    st.stop()

if "history_open_id" not in st.session_state:
    st.session_state.history_open_id = None

for debate in debates:
    when = debate["created_at"].strftime("%Y-%m-%d %H:%M")
    with st.container(border=True):
        st.markdown(f"**{debate['topic']}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("When", when)
        c2.metric("Model", debate["az_model"] or "-")
        c3.metric("Rounds", debate["num_rounds"] or "-")
        c4.metric("Ended", debate["ended_reason"] or "-")

        is_open = st.session_state.history_open_id == debate["id"]
        if is_open:
            if st.button("Hide brief", key=f"hide_{debate['id']}"):
                st.session_state.history_open_id = None
                st.rerun()
            brief = get_debate_brief(debate["id"])
            st.markdown(brief or "_No brief stored for this run._")
        else:
            if st.button("View brief", key=f"view_{debate['id']}"):
                st.session_state.history_open_id = debate["id"]
                st.rerun()
