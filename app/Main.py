"""Main entry point for the Streamlit app."""

import streamlit as st

st.set_page_config(page_title="Multi-Agent Platform", page_icon="🧠", layout="wide")
st.title("Multi-Agent Platform")
st.write("Use the sidebar to navigate through the experiment builder or click the button below to start a new experiment.")
if st.button("Start New Experiment"):
    st.switch_page("pages/1_Welcome.py")


# select a simple case to show a demo - billboard? communication hub, and illustrate this first solution
# but in the true version we'll have all scenarios
# 1. demo
# 2. make it exciting
# 3. professional use for whatever purposes
