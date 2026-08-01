"""Setup page for Agent 0 mode.

Agent 0 designs the debate: the panel of agents, the instructions they get,
the criteria they are scored on, and the visibility rules. The person using
this page only supplies the topic, picks which model runs Agent 0, and sets
hard limits on how large or long the debate can run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.providers import API_KEY_ENV_VARS, PROVIDER_MODELS, PROVIDERS


def init_state() -> None:
    defaults = {
        "az_topic": "",
        "az_brief_instructions": "",
        "az_provider": "Anthropic",
        "az_model": "claude-sonnet-4-6",
        "az_temperature": 0.0,
        "az_max_rounds": 10,
        "az_max_agents": 6,
        "az_max_total_spawns": 8,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

st.title("New Debate")
st.caption(
    "Enter a topic. Agent 0 assembles the panel, sets the ground rules, and runs the "
    "debate to a policy brief."
)

st.divider()

with st.container(border=True):
    st.subheader("Topic")
    st.session_state.az_topic = st.text_area(
        "Topic",
        value=st.session_state.az_topic,
        placeholder="Should the city introduce congestion charges on its inner ring road?",
        height=100,
        label_visibility="collapsed",
        key="widget_topic",
    )

with st.container(border=True):
    st.subheader("Final brief instructions (optional)")
    st.caption(
        "How Agent 0 should structure, format, or shape the brief it writes once the debate ends "
        " - e.g. section order, length, language (only Agent 0 sees this). "
    )
    st.session_state.az_brief_instructions = st.text_area(
        "Final brief instructions",
        value=st.session_state.az_brief_instructions,
        placeholder="e.g. Open with a two-sentence introduction, then a dedicated section on X, then a conclusion.",
        height=80,
        label_visibility="collapsed",
        key="widget_brief_instructions",
    )

with st.container(border=True):
    st.subheader("Moderator")
    st.caption(
        "Agent 0 runs the debate: it picks who takes part, decides what they are scored on, "
        "and calls the debate when it's done. Panel members are drawn from OpenAI's and Anthropic's models. "
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.az_provider = st.selectbox(
            "Provider", PROVIDERS,
            index=PROVIDERS.index(st.session_state.az_provider),
            key="widget_provider",
        )
    with c2:
        model_options = PROVIDER_MODELS.get(st.session_state.az_provider, [])
        current_model = st.session_state.az_model if st.session_state.az_model in model_options else model_options[0]
        st.session_state.az_model = st.selectbox(
            "Model", model_options,
            index=model_options.index(current_model),
            key="widget_model",
        )
    with c3:
        st.session_state.az_temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0,
            value=st.session_state.get("az_temperature", 0.0), step=0.01,
            help="0 keeps Agent 0's decisions consistent. Higher values add variety. "
                 "A few models don't support this and will run at their own default.",
            key="widget_temperature",
        )

with st.expander("Limits"):
    st.caption(
        "These cap how long and how large the debate can grow, regardless of what "
        "Agent 0 decides."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.az_max_rounds = st.number_input(
            "Max rounds", min_value=1, max_value=50, value=st.session_state.az_max_rounds, step=1,
            key="widget_max_rounds",
        )
    with c2:
        st.session_state.az_max_agents = st.number_input(
            "Max panel size", min_value=1, max_value=10, value=st.session_state.az_max_agents, step=1,
            key="widget_max_agents",
        )
    with c3:
        st.session_state.az_max_total_spawns = st.number_input(
            "Max agents created", min_value=1, max_value=20, value=st.session_state.az_max_total_spawns, step=1,
            help="Counts the starting panel plus every agent added later.",
            key="widget_max_total_spawns",
        )

with st.container(border=True):
    st.subheader("API keys")
    used_providers = sorted({"OpenAI", "Anthropic", st.session_state.az_provider})
    api_keys: dict[str, str] = {}
    for provider in used_providers:
        env_var = API_KEY_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")
        if os.environ.get(env_var):
            st.success(f"{provider} key set", icon=":material/check_circle:")
            api_keys[provider] = ""
        else:
            api_keys[provider] = st.text_input(
                f"{provider} API key", type="password", key=f"az_api_key_{provider}",
            )

st.divider()

topic_ready = bool(st.session_state.az_topic.strip())
if not topic_ready:
    st.warning("Enter a topic to start.")

if st.button("Start debate", type="primary", disabled=not topic_ready, use_container_width=True):
    for provider, key in api_keys.items():
        if key:
            os.environ[API_KEY_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")] = key

    st.session_state.experiment_config = {
        "mode": "agent_zero",
        "task": {
            "description": st.session_state.az_topic.strip(),
            "brief_instructions": st.session_state.az_brief_instructions.strip(),
        },
        # "instructions" and "ecu" are left unset here on purpose: Agent 0
        # designs them in its initialization call (see core/agent_zero.py).
        "agent_zero": {
            "provider": st.session_state.az_provider,
            "model": st.session_state.az_model,
            "temperature": float(st.session_state.get("az_temperature", 0.0)),
            "max_rounds": int(st.session_state.az_max_rounds),
            "max_agents": int(st.session_state.az_max_agents),
            "max_total_spawns": int(st.session_state.az_max_total_spawns),
        },
    }
    st.session_state.pop("az_result", None)
    st.switch_page("pages/5_Run.py")
