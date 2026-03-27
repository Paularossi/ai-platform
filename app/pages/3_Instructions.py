"""Instructions and Prompts Setup Page"""

import streamlit as st
import json
from components.utils import output_fields_from_question_sets

st.set_page_config(page_title="Instructions Setup", page_icon="🧠", layout="wide")

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 3 of 5")
st.sidebar.progress(3 / 5)
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

st.title("Instructions and Prompts Setup")
st.caption("This page allows you to set up the instructions and prompts for your multi-agent experiment.")

# ---------- layout ----------
main_col, summary_col = st.columns([2.2, 1], gap="large")

with main_col:
    with st.container(border=True):
        st.subheader("Instruction specification")
        uploaded_instructions = st.file_uploader(
            "Upload instructions as a text file",
            type=["txt"],
            key="instructions_uploader"
        )
        if uploaded_instructions is not None:
            if st.button("Load instructions from file"):
                # add here code to save the instructions in the state
                try:
                    instructions_text = uploaded_instructions.getvalue().decode("utf-8")
                    st.session_state.base_instructions = instructions_text
                    st.success("Instructions loaded successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load instructions: {e}")

        base_instructions = st.text_area(
            "Base task instructions",
            placeholder="Write the main instructions all agents should follow.",
            height=140,
            key="base_instructions"
        )
    
    with st.container(border=True):
        st.subheader("Questions / prompt setup")
        st.caption("Provide the questions that will be used as prompts for the task, as a JSON file.")
        uploaded_questions = st.file_uploader(
            "Upload question set JSON",
            type=["json"],
            help="Upload structured label definitions for each output field.",
            key="question_set_uploader"
        )
        
        if uploaded_questions is not None:
            if st.button("Load questions from file"):
                try:
                    data = json.load(uploaded_questions)

                    if "questions" not in data or not isinstance(data["questions"], list):
                        st.error("Invalid JSON format. Expected a top-level 'questions' list.")
                    else:
                        st.session_state.question_sets = data["questions"]
                        st.session_state.output_fields = output_fields_from_question_sets(data["questions"])
                        st.success("Question sets loaded and output schema updated.")

                except Exception as e:
                    st.error(f"Failed to load JSON: {e}")

    nav1, nav2, nav3 = st.columns([2, 2, 0.8])
    with nav1:
        if st.button("<- Back"):
            st.switch_page("pages/2_Agent Setup.py")
    with nav2:
        if st.button("Save draft"):
            st.success("Instructions setup saved.")
    with nav3:
        if st.button("Next ->"):
            st.switch_page("pages/4_Dataset.py")