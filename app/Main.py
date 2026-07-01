"""Main entry point — navigation router only."""

import streamlit as st

st.set_page_config(page_title="Multi-Agent Lab", page_icon="🧠", layout="wide")

pg = st.navigation([
    st.Page("pages/1_Welcome.py", title="Home", icon="🏠", default=True),
    st.Page("pages/2_Agent Setup.py", title="Agent Setup", icon="🤖"),
    st.Page("pages/3_Instructions.py", title="Instructions & topic", icon="📝"),
    st.Page("pages/4_Review.py", title="Review & Launch", icon="🚀"),
    st.Page("pages/5_Run.py", title="Run", icon="▶️"),
], position="sidebar")
pg.run()
