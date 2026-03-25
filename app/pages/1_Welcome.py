"""Welcome page for setting up a new experiment in the Multi-Agent Platform app."""

import streamlit as st
import json

st.set_page_config(page_title="Multi-Agent Platform", page_icon="🧠", layout="wide")


# ---------- helpers ----------
def default_input_field():
    return {"name": "", "type": "Text"}


def default_output_field():
    return {"name": "", "type": "Single-label"}

def output_fields_from_question_sets(question_sets):
    fields = []
    for q in question_sets:
        field_type_map = {
            "single_label": "Single-label",
            "multi_label": "Multi-label",
            "boolean": "Boolean",
            "score": "Score",
            "text": "Text",
            "ranking": "Ranking",
        }

        fields.append(
            {
                "name": q.get("field_name", ""),
                "type": field_type_map.get(q.get("field_type", "single_label"), "Single-label")
            }
        )
    return fields


def init_state():
    if "input_fields" not in st.session_state:
        st.session_state.input_fields = [
            {"name": "image", "type": "Image"},
            {"name": "caption", "type": "Text"},
        ]
        
    if "output_fields" not in st.session_state:
        st.session_state.output_fields = [
            {"name": "label", "type": "Single-label"},
            {"name": "multi_label", "type": "Multi-label"},
            {"name": "confidence", "type": "Score"},
            {"name": "rationale", "type": "Text"},
        ]

    if "question_sets" not in st.session_state:
        st.session_state.question_sets = []

    if "base_instructions" not in st.session_state:
        st.session_state.base_instructions = ""

    if "exp_name" not in st.session_state:
        st.session_state.exp_name = st.session_state.experiment_config["overview"]["name"]

    if "author" not in st.session_state:
        st.session_state.author = st.session_state.experiment_config["overview"]["author"]

    if "protocol_id" not in st.session_state:
        st.session_state.protocol_id = st.session_state.experiment_config["overview"]["protocol_id"]

    if "task_category" not in st.session_state:
        st.session_state.task_category = st.session_state.experiment_config["task"]["category"]

    if "modalities" not in st.session_state:
        st.session_state.modalities = st.session_state.experiment_config["task"]["modalities"]

    if "task_description" not in st.session_state:
        st.session_state.task_description = st.session_state.experiment_config["task"]["description"]


def init_experiment_config():
    if "experiment_config" not in st.session_state:
        st.session_state.experiment_config = {
            "overview": {
                "name": "",
                "author": "",
                "protocol_id": "",
            },
            "task": {
                "category": "Classification / annotation",
                "modalities": ["Image", "Text"],
                "description": "",
            },
            "schemas": {
                "input_fields": [],
                "output_fields": [],
            },
            "questions": [],
            "instructions": {
                "base_instructions": "",
                "guideline_notes": "",
            },
        }


def sync_welcome_to_config():
    st.session_state.experiment_config["overview"] = {
        "name": st.session_state.get("exp_name", ""),
        "author": st.session_state.get("author", ""),
        "protocol_id": st.session_state.get("protocol_id", ""),
    }

    st.session_state.experiment_config["task"] = {
        "category": st.session_state.get("task_category", "Classification / annotation"),
        "modalities": st.session_state.get("modalities", ["Image", "Text"]),
        "description": st.session_state.get("task_description", ""),
    }

    st.session_state.experiment_config["schemas"] = {
        "input_fields": st.session_state.get("input_fields", []),
        "output_fields": st.session_state.get("output_fields", []),
    }

    st.session_state.experiment_config["questions"] = st.session_state.get("question_sets", [])

    st.session_state.experiment_config["instructions"] = {
        "base_instructions": st.session_state.get("base_instructions", ""),
        "guideline_notes": st.session_state.get("guideline_notes", ""),
    }


init_experiment_config()
init_state()

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 1 of 6")
st.sidebar.progress(1 / 6)
st.sidebar.markdown(
    """
**Steps**
1. Task
2. Agents
3. Schema
4. Instructions
5. Evaluation
6. Review
"""
)


# ---------- header ----------
st.title("Create Experiment")
st.caption("Define a reusable task setup for multi-agent workflows.")


# ---------- layout ----------
main_col, summary_col = st.columns([2.1, 1], gap="large")

with main_col:
    with st.container(border=True):
        st.subheader("1. Experiment overview")
        exp_name = st.text_input("Experiment name", placeholder="e.g. Gossip pilot for multimodal ad annotation", key="exp_name")

        c1, c2 = st.columns(2)
        with c1:
            author = st.text_input("Author", placeholder="e.g. Paula / Freija", key="author")
        with c2:
            protocol_id = st.text_input("Protocol version", placeholder="e.g. v0.1", key="protocol_id")

    with st.container(border=True):
        st.subheader("2. Task specification")

        c1, c2 = st.columns(2)
        with c1:
            task_category = st.selectbox(
                "Task category",
                [
                    "Classification / annotation",
                    "Extraction",
                    "Summarization",
                    "Evaluation / judging",
                    "Generation",
                    "Comparison / ranking",
                    "Review / revision",
                    "Custom",
                ], key="task_category"
            )
        with c2:
            modalities = st.multiselect(
                "Modality",
                ["Text", "Image", "Audio", "Video", "Tabular / structured data", "Multimodal"],
                default=["Image", "Text"], key="modalities"
            )

        task_description = st.text_area(
            "Task description",
            placeholder="Describe the task in natural language.",
            height=120, key="task_description"
        )

    with st.container(border=True):
        st.subheader("3. Input schema")
        st.caption("Define what each task item contains.")

        for i, field in enumerate(st.session_state.input_fields):
            cols = st.columns([1.4, 1, 0.5])
            with cols[0]:
                field["name"] = st.text_input("Field name", value=field["name"], key=f"in_name_{i}")
            with cols[1]:
                field["type"] = st.selectbox(
                    "Type",
                    ["Text", "Long text", "Image", "Audio", "Video", "JSON", "Numeric", "Category"],
                    index=["Text", "Long text", "Image", "Audio", "Video", "JSON", "Numeric", "Category"].index(field["type"]),
                    key=f"in_type_{i}",
                )
            with cols[2]:
                st.write("")
                st.write("")
                if st.button("✕", key=f"remove_input_{i}"):
                    st.session_state.input_fields.pop(i)
                    st.rerun()

        if st.button("+ Add input field"):
            st.session_state.input_fields.append(default_input_field())
            st.rerun()

    with st.container(border=True):
        st.subheader("4. Output schema")
        st.caption("Define what the agents should return.")

        # add option to upload a json file
        uploaded_output_schema = st.file_uploader(
            "Upload output schema JSON",
            type=["json"],
            key="output_schema_uploader",
            help="Upload a JSON file describing the output fields."
        )

        if uploaded_output_schema is not None:
            if st.button("Load output schema from file"):
                try:
                    schema_data = json.load(uploaded_output_schema)

                    if isinstance(schema_data, dict) and "output_fields" in schema_data:
                        new_fields = schema_data["output_fields"]
                    elif isinstance(schema_data, list):
                        new_fields = schema_data
                    else:
                        st.error("Invalid JSON format. Use a list of fields or {'output_fields': [...]}.")

                    if "new_fields" in locals():
                        validated_fields = []
                        for field in new_fields:
                            validated_fields.append(
                                {
                                    "name": field.get("name", ""),
                                    "type": field.get("type", "Single-label")
                                }
                            )

                        st.session_state.output_fields = validated_fields
                        st.success("Output schema loaded successfully.")

                except Exception as e:
                    st.error(f"Failed to load JSON: {e}")


        for i, field in enumerate(st.session_state.output_fields):
            cols = st.columns([1.4, 1, 0.5])
            with cols[0]:
                field["name"] = st.text_input("Field name", value=field["name"], key=f"out_name_{i}")
            with cols[1]:
                output_type_options = ["Single-label", "Multi-label", "Score", "Text", "Boolean", "Ranking"]
                field_type = field["type"] if field["type"] in output_type_options else "Single-label"

                field["type"] = st.selectbox(
                    "Type",
                    output_type_options,
                    index=output_type_options.index(field_type),
                    key=f"out_type_{i}",
                )
            with cols[2]:
                st.write("")
                st.write("")
                if st.button("✕", key=f"remove_output_{i}"):
                    st.session_state.output_fields.pop(i)
                    st.rerun()

        add_col, clear_col, _ = st.columns([3, 3, 2])

        with add_col:
            if st.button("+ Add output field"):
                st.session_state.output_fields.append(default_output_field())
                st.rerun()

        with clear_col:
            if st.button("Clear all output fields", type="secondary"):
                st.session_state.output_fields = []
                st.rerun()

    with st.container(border=True):
        st.subheader("5. Questions / prompt setup")
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


    with st.container(border=True):
        st.subheader("6. Instruction specification")
        uploaded_instructions = st.file_uploader(
            "Upload instructions as a text file",
            type=["txt"],
            key="instructions_uploader"
        )
        if uploaded_instructions is not None:
            if st.button("Load instructions from file"):
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

        guideline_notes = st.text_area(
            "Guidelines / protocol notes",
            placeholder="Paste coding rules, rubric notes, or protocol constraints.",
            height=120,
        )

    st.info("Next: Configure agents and interaction protocol ->")

    nav1, nav2, nav3 = st.columns([2, 2, 5]) # the numbers represent the relative width of the columns, so nav3 is wider to push the buttons to the left
    with nav1:
        if st.button("Save draft"):
            sync_welcome_to_config()
            with open("experiment_draft.json", "w", encoding="utf-8") as f:
                json.dump(st.session_state.experiment_config, f, indent=2, ensure_ascii=False)
            st.success("Draft saved.")
    with nav2:
        if st.button("Next ->"):
            sync_welcome_to_config()  # make sure to save all the info in the session state before moving to the next page
            st.switch_page("pages/2_Agent Setup.py")

with summary_col:
    with st.container(border=True):
        st.subheader("Live summary")
        st.markdown(f"**Name**  \n{exp_name or '—'}")
        st.markdown(f"**Task category**  \n{task_category}")
        st.markdown(f"**Modality**  \n{', '.join(modalities) if modalities else '—'}")
        st.markdown(f"**Author**  \n{author or '—'}")
        st.markdown(f"**Protocol**  \n{protocol_id or '—'}")

        st.divider()
        st.markdown("**Task description**")
        st.write(task_description or "—")

        st.divider()
        st.markdown("**Input fields**")
        for field in st.session_state.input_fields:
            st.markdown(f"- `{field['name'] or 'unnamed'}` ({field['type']})")

        st.markdown("**Output fields**")
        current_output_fields = st.session_state.get("output_fields", [])

        for field in current_output_fields:
            st.markdown(f"- `{field['name'] or 'unnamed'}` ({field['type']})")

# everything works, next step to save all this info in the session state and make it available in the next pages, and ideally also save it as a draft to a file so we can retrieve it later when we want to run the experiment