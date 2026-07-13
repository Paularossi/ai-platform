"""Main entry point - navigation router and shared styling."""

import streamlit as st

st.set_page_config(page_title="OMAS Deliberation", page_icon=":material/balance:", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 16px;
    }
    h1 { font-size: 2.1rem; }
    h2 { font-size: 1.5rem; }
    h3 { font-size: 1.2rem; }
    h1, h2, h3 {
        font-family: 'Lora', serif;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    p, li, .stMarkdown {
        font-size: 1rem;
        line-height: 1.55;
    }
    [data-testid="stCaptionContainer"], .stCaption {
        font-size: 0.95rem !important;
    }
    code, pre, .stCode, [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        color: #2E5945;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
    button[kind="primary"] {
        background-color: #2E5945 !important;
        border-color: #2E5945 !important;
    }
    button[kind="primary"]:hover {
        background-color: #234636 !important;
        border-color: #234636 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

pg = st.navigation([
    st.Page("pages/1_Welcome.py", title="Setup", icon=":material/tune:", default=True),
    st.Page("pages/5_Run.py", title="Debate", icon=":material/forum:"),
    st.Page("pages/6_History.py", title="History", icon=":material/history:"),
], position="sidebar")
pg.run()

# TODO:
# - add cost estimate?
# - show progress bar for deliberation?
# - show full summary at the end? e.g. final weights, etc.