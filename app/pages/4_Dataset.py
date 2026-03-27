"""Dataset Upload Page"""

import streamlit as st

st.set_page_config(page_title="Dataset Upload", page_icon="🧠", layout="wide")

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 4 of 5")
st.sidebar.progress(4 / 5)
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

st.title("Dataset Upload")
st.caption("Upload and manage your dataset for the experiment.")


# ---------- layout ----------
main_col, summary_col = st.columns([2.2, 1], gap="large")

with main_col:
    with st.container(border=True):
        st.subheader("Dataset specification")
        uploaded_dataset = st.file_uploader(
            "Upload dataset as a CSV/Excel file",
            type=["csv", "xlsx"],
            key="dataset_uploader"
        )
        if uploaded_dataset is not None:
            if st.button("Load dataset from file"):
                # add here code to save the dataset in the state
                try:
                    data = uploaded_dataset.getvalue().decode("utf-8")
                    st.session_state.dataset = data
                    st.success("Dataset loaded successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")

    nav1, nav2, nav3 = st.columns([2, 2, 0.8])
    with nav1:
        if st.button("<- Back"):
            st.switch_page("pages/3_Instructions.py")
    with nav2:
        if st.button("Save draft"):
            st.success("Dataset setup saved.")
    with nav3:
        if st.button("Next ->"):
            st.switch_page("pages/5_Review.py")