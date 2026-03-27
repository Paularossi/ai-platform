"""Review Page"""

import streamlit as st

st.set_page_config(page_title="Experiment Setup Review", page_icon="🧠", layout="wide")

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 5 of 5")
st.sidebar.progress(5 / 5)
st.sidebar.markdown(
    """
**Steps**
1. Task
2. Agents
3. Instructions
4. Dataset
5. Review
"""
)

st.title("Experiment Setup Review")
st.caption("Review and finalize your experiment setup before deployment.")
