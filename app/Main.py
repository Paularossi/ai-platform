"""Main entry point — navigation router only."""

import streamlit as st

st.set_page_config(page_title="Multi-Agent Lab", page_icon="🧠", layout="wide")

pages = [
    st.Page("pages/1_Welcome.py", title="Home", icon="🏠", default=True),
    st.Page("pages/2_Agent Setup.py", title="Agent Setup", icon="🤖"),
    st.Page("pages/3_Instructions.py", title="Instructions & topic", icon="📝"),
    st.Page("pages/4_Review.py", title="Review & Launch", icon="🚀"),
    st.Page("pages/5_Run.py", title="Run", icon="▶️"),
]
# Only in the sidebar once logged in as a tutor
# if st.session_state.get("is_tutor"):
#     pages.append(st.Page("pages/6_Tutor.py", title="Tutor", icon="🎓"))

pg = st.navigation(pages, position="sidebar")
pg.run()
