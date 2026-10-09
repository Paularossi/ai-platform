"""Main entry point — navigation router only."""

import streamlit as st

st.set_page_config(page_title="Multi-Agent Lab", page_icon="🧠", layout="wide")

pages = [
    st.Page("pages/1_Welcome.py", title="Home", icon="🏠", default=True),
    st.Page("pages/2_Agent Setup.py", title="Topic & Agents", icon="🤖"),
    st.Page("pages/3_Instructions.py", title="Protocol & Evaluation", icon="📝"),
    st.Page("pages/4_Review.py", title="Review & Launch", icon="🚀"),
    st.Page("pages/5_Run.py", title="Run", icon="▶️"),
]

pg = st.navigation(pages, position="sidebar")
pg.run()
