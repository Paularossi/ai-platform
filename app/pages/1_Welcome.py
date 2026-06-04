"""Step 1 — Task definition."""

import streamlit as st

st.set_page_config(page_title="Multi-Agent Platform", page_icon="🧠", layout="wide")


# ---------- helpers ----------
def init_experiment_config():
    if "experiment_config" not in st.session_state:
        st.session_state.experiment_config = {
            "overview": {"name": "", "author": "", "protocol_id": ""},
            "task": {"category": "Deliberation", "modalities": ["Text"], "description": "", "mode": "deliberation"},
            "schemas": {"input_fields": []},
            "questions": [],
            "instructions": {"base_instructions": "", "guideline_notes": ""},
        }


def init_state():
    cfg = st.session_state.experiment_config

    if "exp_name" not in st.session_state:
        st.session_state.exp_name = cfg["overview"]["name"]
    if "author" not in st.session_state:
        st.session_state.author = cfg["overview"]["author"]
    if "protocol_id" not in st.session_state:
        st.session_state.protocol_id = cfg["overview"]["protocol_id"]
    if "task_category" not in st.session_state:
        st.session_state.task_category = cfg["task"]["category"]
    if "modalities" not in st.session_state:
        st.session_state.modalities = cfg["task"]["modalities"]
    if "task_description" not in st.session_state:
        st.session_state.task_description = cfg["task"]["description"]
    if "task_mode" not in st.session_state:
        st.session_state.task_mode = cfg["task"].get("mode", "deliberation")
    if "input_fields" not in st.session_state:
        st.session_state.input_fields = cfg["schemas"].get("input_fields", [])


def sync_to_config():
    st.session_state.experiment_config["overview"] = {
        "name": st.session_state.get("exp_name", ""),
        "author": st.session_state.get("author", ""),
        "protocol_id": st.session_state.get("protocol_id", ""),
    }
    st.session_state.experiment_config["task"] = {
        "category": st.session_state.get("task_category", "Deliberation"),
        "modalities": st.session_state.get("modalities", ["Text"]),
        "description": st.session_state.get("task_description", ""),
        "mode": st.session_state.get("task_mode", "deliberation"),
    }
    st.session_state.experiment_config["schemas"] = {
        "input_fields": st.session_state.get("input_fields", []),
    }


init_experiment_config()
init_state()

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 1 of 5")
st.sidebar.progress(1 / 5)
st.sidebar.markdown("""
**Steps**
1. Task
2. Agents
3. Instructions
4. Dataset
5. Review
""")

st.title("Task Definition")
st.caption("Define what the experiment is about and what agents receive as input.")

main_col, summary_col = st.columns([2.1, 1], gap="large")

with main_col:

    # ── Overview ──────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("1. Overview")
        st.text_input(
            "Experiment name",
            placeholder="e.g. Gossip pilot — Italy description task",
            key="exp_name",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Author", placeholder="e.g. Paula / Freija", key="author")
        with c2:
            st.text_input("Protocol version", placeholder="e.g. v0.1", key="protocol_id")

    # ── Task specification ────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("2. Task specification")

        c1, c2 = st.columns(2)
        with c1:
            st.selectbox(
                "Task category",
                [
                    "Deliberation",
                    "Classification / annotation",
                    "Extraction",
                    "Summarization",
                    "Evaluation / judging",
                    "Generation",
                    "Comparison / ranking",
                    "Custom",
                ],
                key="task_category",
            )
        with c2:
            st.multiselect(
                "Modality",
                ["Text", "Image", "Audio", "Video", "Tabular / structured data"],
                default=st.session_state.modalities,
                key="modalities",
            )

        st.text_area(
            "Task description",
            placeholder=(
                "Describe the task in natural language. This is shown to you as reference "
                "— the actual instructions sent to agents are set in Step 3.\n\n"
                "E.g.: Agents are given a country name and must produce a one-sentence "
                "description. They interact over multiple rounds to reach a shared description."
            ),
            height=130,
            key="task_description",
        )

        # Task mode — determines how agent.py builds prompts and parses outputs
        mode_options = ["deliberation", "classification"]
        mode_labels = {
            "deliberation": "Deliberation — agents produce free-text contributions",
            "classification": "Classification — agents answer structured questions with option codes",
        }
        current_mode = st.session_state.get("task_mode", "deliberation")
        selected_mode = st.radio(
            "Task mode",
            mode_options,
            format_func=lambda m: mode_labels[m],
            index=mode_options.index(current_mode),
            key="task_mode",
            help=(
                "Deliberation: agents write free-text statements (e.g. describe Italy). "
                "Classification: agents choose from predefined option codes (e.g. target_age: ADULT)."
            ),
        )

        if selected_mode == "classification":
            st.info(
                "In Step 3 you will upload a question set JSON defining the fields and option codes.",
                icon="ℹ️",
            )

    # ── Input schema ──────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("3. Input schema")
        st.caption(
            "Define what each item in your dataset contains. "
            "These fields are used to build the context shown to agents."
        )

        input_type_options = ["Text", "Long text", "Image", "Audio", "Video", "JSON", "Numeric", "Category"]

        for i, field in enumerate(st.session_state.input_fields):
            cols = st.columns([1.4, 1, 0.4])
            with cols[0]:
                field["name"] = st.text_input(
                    "Field name", value=field["name"], key=f"in_name_{i}",
                    placeholder="e.g. country, caption, image"
                )
            with cols[1]:
                idx = input_type_options.index(field["type"]) if field["type"] in input_type_options else 0
                field["type"] = st.selectbox(
                    "Type", input_type_options, index=idx, key=f"in_type_{i}"
                )
            with cols[2]:
                st.write("")
                st.write("")
                if st.button("✕", key=f"rm_in_{i}"):
                    st.session_state.input_fields.pop(i)
                    st.rerun()

        if st.button("+ Add input field"):
            st.session_state.input_fields.append({"name": "", "type": "Text"})
            st.rerun()

    st.info("Next: configure agents and interaction protocol →")

    nav1, nav2, _ = st.columns([2, 2, 4])
    with nav1:
        if st.button("Save draft"):
            sync_to_config()
            st.switch_page("pages/5_Review.py")
    with nav2:
        if st.button("Next →"):
            sync_to_config()
            st.switch_page("pages/2_Agent Setup.py")


with summary_col:
    with st.container(border=True):
        st.subheader("Live summary")
        st.markdown(f"**Name**  \n{st.session_state.get('exp_name') or '—'}")
        st.markdown(f"**Author**  \n{st.session_state.get('author') or '—'}")
        st.markdown(f"**Category**  \n{st.session_state.get('task_category', '—')}")
        st.markdown(f"**Mode**  \n{st.session_state.get('task_mode', '—')}")
        st.markdown(f"**Modality**  \n{', '.join(st.session_state.get('modalities', [])) or '—'}")
        st.divider()
        st.markdown("**Task description**")
        st.write(st.session_state.get("task_description") or "—")
        st.divider()
        st.markdown("**Input fields**")
        for f in st.session_state.get("input_fields", []):
            st.markdown(f"- `{f['name'] or 'unnamed'}` ({f['type']})")