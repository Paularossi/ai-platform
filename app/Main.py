"""Main entry point — navigation router only."""

import streamlit as st

st.set_page_config(page_title="Agent 0 Lab", page_icon="🧠", layout="wide")

pg = st.navigation([
    st.Page("pages/1_Welcome.py", title="Setup", icon="🧠", default=True),
    st.Page("pages/5_Run.py", title="Run", icon="▶️"),
], position="sidebar")
pg.run()
