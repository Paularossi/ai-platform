"""Main entry point for the Streamlit app."""

import streamlit as st

st.set_page_config(page_title="Multi-Agent Platform", page_icon="🧠", layout="wide")
st.title("Multi-Agent Platform")
st.write("Use the sidebar to navigate through the experiment builder or click the button below to start a new experiment.")
if st.button("Start New Experiment"):
    st.switch_page("pages/1_Welcome.py")