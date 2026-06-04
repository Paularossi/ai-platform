import streamlit as st


def restore_draft(loaded: dict) -> None:
    """
    Restore a full experiment config dict (previously saved as JSON) into
    session state. Called from both Main.py (on upload) and the Review page.
    After calling this, st.switch_page("pages/5_Review.py") to show the summary.
    """
    # Store the raw config so Review page can read it
    st.session_state.experiment_config = {
        k: v for k, v in loaded.items() if k != "meta"
    }

    ov = loaded.get("overview", {})
    st.session_state.exp_name = ov.get("name", "")
    st.session_state.author = ov.get("author", "")
    st.session_state.protocol_id = ov.get("protocol_id", "")

    task = loaded.get("task", {})
    st.session_state.task_category = task.get("category", "")
    st.session_state.modalities = task.get("modalities", [])
    st.session_state.task_description = task.get("description", "")
    st.session_state.task_mode = task.get("mode", "deliberation")

    schemas = loaded.get("schemas", {})
    st.session_state.input_fields = schemas.get("input_fields", [])

    st.session_state.question_sets = loaded.get("questions", [])

    instr = loaded.get("instructions", {})
    st.session_state.base_instructions = instr.get("base_instructions", "")
    st.session_state.guideline_notes = instr.get("guideline_notes", "")

    st.session_state.agent_prompt_overrides = loaded.get("agent_prompt_overrides", {})
    st.session_state.agents = loaded.get("agents", [])
    st.session_state.num_agents = len(st.session_state.agents)

    proto = loaded.get("protocol", {})
    st.session_state.interaction_setting = proto.get("setting", "Gossip (sequential)")
    st.session_state.platform_mode = proto.get("platform_mode", "Multi-platform")
    st.session_state.supervision_mode = proto.get("supervision_mode", "Unsupervised")
    st.session_state.visibility_mode = proto.get("visibility_mode", "Current state only")
    st.session_state.order_type = proto.get("order_type", "Fixed")
    st.session_state.max_cycles = proto.get("max_cycles", 5)
    st.session_state.stopping_rule = proto.get("stopping_rule", "Either")
    st.session_state.initializer_agent = proto.get("initializer_agent", "")
    st.session_state.judge_agent = proto.get("judge_agent", "")

    ds = loaded.get("dataset", {})
    st.session_state.column_mapping = ds.get("column_mapping", {})


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