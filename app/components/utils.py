import sys
from pathlib import Path
import streamlit as st
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def restore_draft(loaded: dict) -> None:
    """Restore a saved experiment JSON into session state."""
    # Clear any stale per-page session keys so every page re-initialises
    # from the freshly loaded experiment_config rather than old values.
    stale_keys = [
        "task_description", "base_instructions", "guideline_notes",
        "agent_prompt_overrides", "agents", "num_agents",
        "interaction_setting", "supervision_mode", "visibility_mode",
        "review_depth", "order_type", "max_cycles", "stopping_rule",
        "initializer_agent", "judge_agent",
        "ecu_enabled", "ecu_info_condition", "ecu_self_assessment",
        "ecu_coalition_threshold", "ecu_dimensions",
        "ecu_orchestrator_enabled", "ecu_orchestrator_step_size",
        "ecu_orchestrator_every",
    ]
    for key in stale_keys:
        st.session_state.pop(key, None)

    st.session_state.experiment_config = {
        k: v for k, v in loaded.items() if k != "meta"
    }

    ov = loaded.get("overview", {})
    st.session_state.exp_name = ov.get("name", "")
    st.session_state.author = ov.get("author", "")
    st.session_state.protocol_id = ov.get("protocol_id", "")

    task = loaded.get("task", {})
    st.session_state.task_description = task.get("description", "")
    if st.session_state.task_description:
        st.session_state.dataset_df = pd.DataFrame([{"topic": st.session_state.task_description}])
        st.session_state.column_mapping = {"topic": "topic"}

    instr = loaded.get("instructions", {})
    st.session_state.base_instructions = instr.get("base_instructions", "")
    st.session_state.guideline_notes = instr.get("guideline_notes", "")

    st.session_state.agent_prompt_overrides = loaded.get("agent_prompt_overrides", {})
    st.session_state.agents = loaded.get("agents", [])
    st.session_state.num_agents = len(st.session_state.agents)

    proto = loaded.get("protocol", {})
    st.session_state.interaction_setting = proto.get("setting", "Simultaneous")
    st.session_state.supervision_mode = proto.get("supervision_mode", "Unsupervised")
    st.session_state.visibility_mode = proto.get("visibility_mode", "Previous round")
    st.session_state.review_depth = proto.get("review_depth", "previous_round")
    st.session_state.order_type = proto.get("order_type", "Fixed")
    st.session_state.max_cycles = proto.get("max_cycles", 5)
    st.session_state.stopping_rule = proto.get("stopping_rule", "Either")
    st.session_state.initializer_agent = proto.get("initializer_agent", "")
    st.session_state.judge_agent = proto.get("judge_agent", "")

    ecu = loaded.get("ecu", {})
    st.session_state.ecu_enabled = ecu.get("enabled", True)
    st.session_state.ecu_info_condition = ecu.get("info_condition", "opaque")
    st.session_state.ecu_self_assessment = ecu.get("include_self_assessment", False)
    st.session_state.ecu_coalition_threshold = float(ecu.get("coalition_threshold", 0.6))
    if "dimensions" in ecu:
        st.session_state.ecu_dimensions = ecu["dimensions"]
    st.session_state.ecu_orchestrator_enabled = ecu.get("orchestrator_enabled", False)
    st.session_state.ecu_orchestrator_step_size = float(ecu.get("orchestrator_step_size", 0.1))
    st.session_state.ecu_orchestrator_every = int(ecu.get("orchestrator_every", 2))

    ds = loaded.get("dataset", {})
    st.session_state.column_mapping = ds.get("column_mapping", {})