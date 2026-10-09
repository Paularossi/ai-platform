"""Shared setup state for the topic and protocol pages."""
import streamlit as st


def init_agent_config():
    if "experiment_config" not in st.session_state:
        st.session_state.experiment_config = {}

    if "agents" not in st.session_state.experiment_config:
        st.session_state.experiment_config["agents"] = []

    if "protocol" not in st.session_state.experiment_config:
        st.session_state.experiment_config["protocol"] = {
            "setting": "Simultaneous",
            "run_mode": "Automatic",
            "visibility_mode": "Previous round",
            "review_depth": "Previous Round",
            "order_type": "Fixed",
            "custom_order": [],
            "initializer_agent": "Agent 1",
            "max_cycles": 5,
            "stopping_rule": "Either",
        }
    # ensure overview is preserved if coming from Main
    if "overview" not in st.session_state.experiment_config:
        st.session_state.experiment_config["overview"] = {
            "name": st.session_state.get("exp_name", ""),
            "author": st.session_state.get("author", ""),
        }

def sync_agents_to_config():
    st.session_state.experiment_config["agents"] = st.session_state.agents

    st.session_state.experiment_config["protocol"] = {
        "setting": st.session_state.interaction_setting,
        "run_mode": st.session_state.run_mode,
        "visibility_mode": st.session_state.visibility_mode,
        "review_depth": st.session_state.get("review_depth", "Previous Round"),
        "order_type": st.session_state.order_type,
        "custom_order": st.session_state.get("custom_order", []),
        "initializer_agent": st.session_state.initializer_agent,
        "max_cycles": st.session_state.max_cycles,
        "stopping_rule": st.session_state.stopping_rule,
    }


def init_agent_state(reload_from_draft: bool = False):
    protocol = st.session_state.experiment_config.get("protocol", {})
    saved_agents = st.session_state.experiment_config.get("agents", [])

    if "num_agents" not in st.session_state or reload_from_draft:
        st.session_state.num_agents = len(saved_agents) if saved_agents else 3

    if "agents" not in st.session_state or reload_from_draft:
        if saved_agents:
            st.session_state.agents = saved_agents
        else:
            st.session_state.agents = [
                {
                    "name": "Agent 1",
                    "provider": "OpenAI",
                    "model": "gpt-6-luna",
                    "temperature": 0.0,
                    "role": "Custom",
                    "custom_role": "",
                },
                {
                    "name": "Agent 2",
                    "provider": "Anthropic",
                    "model": "claude-sonnet-5-5",
                    "temperature": 0.0,
                    "role": "Custom",
                    "custom_role": "",
                },
                {
                    "name": "Agent 3",
                    "provider": "Google",
                    "model": "gemini-3.8-flash",
                    "temperature": 0.0,
                    "role": "Custom",
                    "custom_role": "",
                },
            ]

    if "interaction_setting" not in st.session_state or reload_from_draft:
        st.session_state.interaction_setting = protocol.get("setting", "Simultaneous")

    if "run_mode" not in st.session_state or reload_from_draft:
        st.session_state.run_mode = protocol.get("run_mode", "Automatic")

    if "visibility_mode" not in st.session_state or reload_from_draft:
        st.session_state.visibility_mode = protocol.get("visibility_mode", "Previous round")

    if "review_depth" not in st.session_state or reload_from_draft:
        st.session_state.review_depth = protocol.get("review_depth", "Previous Round")

    if "order_type" not in st.session_state or reload_from_draft:
        st.session_state.order_type = protocol.get("order_type", "Fixed")

    if "custom_order" not in st.session_state or reload_from_draft:
        st.session_state.custom_order = protocol.get("custom_order", [])

    if "initializer_agent" not in st.session_state or reload_from_draft:
        st.session_state.initializer_agent = protocol.get("initializer_agent", "Agent 1")

    if "max_cycles" not in st.session_state or reload_from_draft:
        st.session_state.max_cycles = protocol.get("max_cycles", 5)

    if "stopping_rule" not in st.session_state or reload_from_draft:
        st.session_state.stopping_rule = protocol.get("stopping_rule", "Either")
