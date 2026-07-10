"""Agent 0 mode — Setup: topic, Agent 0's own model, and hard stopping bounds.

This is the only configuration page. Agent 0 designs everything about how the
debate runs — the roster, the shared instructions given to agents, the
quality dimensions and weights, the coalition threshold — and adapts, steers,
and ends the debate autonomously. The human only supplies the topic, which
LLM orchestrates, and the hard safety bounds that cap cost/runtime.
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

st.title("🧠 Agent 0 — Autonomous Deliberation")
st.caption(
    "Give it a topic. Agent 0 designs the debating roster, the instructions agents receive, "
    "and the quality dimensions used to score them — then adapts, steers, and ends the debate "
    "with a final policy brief, autonomously."
)

st.divider()

with st.container(border=True):
    st.subheader("Deliberation topic")
    st.session_state.az_topic = st.text_area(
        "Topic / question",
        value=st.session_state.az_topic,
        placeholder="Should the city introduce congestion charges on its inner ring road?",
        height=100,
        label_visibility="collapsed",
    )

with st.container(border=True):
    st.subheader("Agent 0's model")
    st.caption(
        "Which LLM orchestrates the debate — designs the roster and evaluation criteria, "
        "adds/removes agents, steers them, and decides when to end. (The debating agents "
        "themselves are drawn from OpenAI/gpt-4o and Anthropic/claude-sonnet-4-6.)"
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.az_provider = st.selectbox(
            "Provider", PROVIDERS,
            index=PROVIDERS.index(st.session_state.az_provider),
        )
    with c2:
        model_options = PROVIDER_MODELS.get(st.session_state.az_provider, [])
        current_model = st.session_state.az_model if st.session_state.az_model in model_options else model_options[0]
        st.session_state.az_model = st.selectbox(
            "Model", model_options,
            index=model_options.index(current_model),
        )
    with c3:
        st.session_state.az_temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0,
            value=st.session_state.get("az_temperature", 0.0), step=0.1,
            help="0 = deterministic. Higher values increase variety in Agent 0's own decisions. "
                 "Ignored for some Anthropic models on this call path.",
        )

with st.expander("Advanced: safety bounds"):
    st.caption(
        "Enforced by the loop, independent of Agent 0's own judgement — these cap runtime "
        "and cost regardless of what Agent 0 decides."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.az_max_rounds = st.number_input(
            "Max rounds", min_value=1, max_value=50, value=st.session_state.az_max_rounds, step=1,
        )
    with c2:
        st.session_state.az_max_agents = st.number_input(
            "Max agents on roster", min_value=1, max_value=10, value=st.session_state.az_max_agents, step=1,
        )
    with c3:
        st.session_state.az_max_total_spawns = st.number_input(
            "Max total spawns", min_value=1, max_value=20, value=st.session_state.az_max_total_spawns, step=1,
            help="Caps the cumulative number of agents ever created, including the initial roster.",
        )

with st.container(border=True):
    st.subheader("API keys")
    used_providers = sorted({"OpenAI", "Anthropic", st.session_state.az_provider})
    api_keys: dict[str, str] = {}
    for provider in used_providers:
        env_var = API_KEY_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")
        if os.environ.get(env_var):
            st.success(f"{provider} API key configured ✓", icon="🔑")
            api_keys[provider] = ""
        else:
            api_keys[provider] = st.text_input(
                f"{provider} API key", type="password", key=f"az_api_key_{provider}",
            )

st.divider()

topic_ready = bool(st.session_state.az_topic.strip())
if not topic_ready:
    st.warning("Enter a deliberation topic to continue.")

if st.button("▶ Launch debate", type="primary", disabled=not topic_ready, use_container_width=True):
    for provider, key in api_keys.items():
        if key:
            os.environ[API_KEY_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")] = key

    st.session_state.experiment_config = {
        "mode": "agent_zero",
        "task": {"description": st.session_state.az_topic.strip()},
        # "instructions" and "ecu" are intentionally left for Agent 0 to
        # design in its initialization call — see core/agent_zero.py.
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
