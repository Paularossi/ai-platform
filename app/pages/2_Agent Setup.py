"""Step 1 — Deliberation topic, instructions, and agent roster."""

import streamlit as st

from components.utils import require_login
from components.setup_state import init_agent_config, init_agent_state, sync_agents_to_config
import pandas as pd

require_login() # checks for valid student/tutor id


# ---------- helpers ----------


def sync_agents_to_count(n: int):
    current = st.session_state.agents
    if len(current) < n:
        for i in range(len(current), n):
            current.append(
                {
                    "name": f"Agent {i+1}",
                    "provider": "OpenAI",
                    "model": "gpt-4o",
                    "temperature": 0.0,
                    "role": "Custom",
                    "custom_role": "",
                }
            )
    elif len(current) > n:
        st.session_state.agents = current[:n]


def change_agent_provider(index: int):
    """Do not carry a model ID from one provider into another provider's API."""
    from core.providers import PROVIDER_MODELS
    provider = st.session_state[f"agent_provider_{index}"]
    agent = st.session_state.agents[index]
    agent["provider"] = provider
    agent["model"] = next(iter(PROVIDER_MODELS.get(provider, [])), "")
    st.session_state.pop(f"agent_model_{index}", None)


init_agent_config()
# a freshly loaded draft re-syncs session_state below even if the keys already exist from an earlier visit to this page
_reload_from_draft = st.session_state.get("_draft_loaded_token") != st.session_state.get("_agent_setup_synced_token")
init_agent_state(_reload_from_draft)
st.session_state["_agent_setup_synced_token"] = st.session_state.get("_draft_loaded_token")


# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 1 of 3")
st.sidebar.progress(1 / 3)
st.sidebar.markdown("""
**Steps**
1. **Topic & Agents ← you are here**
2. Protocol & Evaluation
3. Review
4. Run
""")

st.title("Topic & Agents")
st.caption("Define the topic, instructions, and agent roster for the deliberation.")


def init_state():
    if "experiment_config" not in st.session_state:
        st.session_state.experiment_config = {}
    cfg = st.session_state.experiment_config
    instr = cfg.get("instructions", {})
    task = cfg.get("task", {})

    # a freshly loaded draft re-syncs these even if the keys already exist from an earlier visit to this page
    reload_from_draft = st.session_state.get("_draft_loaded_token") != st.session_state.get("_instructions_synced_token")

    if "task_description" not in st.session_state or reload_from_draft:
        st.session_state.task_description = task.get("description", "")
    if "base_instructions" not in st.session_state or reload_from_draft:
        st.session_state.base_instructions = instr.get("base_instructions", "")
    if "guideline_notes" not in st.session_state or reload_from_draft:
        st.session_state.guideline_notes = instr.get("guideline_notes", "")
    if "agent_prompt_overrides" not in st.session_state or reload_from_draft:
        st.session_state.agent_prompt_overrides = cfg.get("agent_prompt_overrides", {})

    st.session_state["_instructions_synced_token"] = st.session_state.get("_draft_loaded_token")


def sync_to_config():
    topic = st.session_state.get("task_description", "").strip()
    st.session_state.experiment_config["task"] = {"description": topic}
    st.session_state.experiment_config["instructions"] = {
        "base_instructions": st.session_state.get("base_instructions", ""),
        "guideline_notes": st.session_state.get("guideline_notes", ""),
    }
    st.session_state.experiment_config["agent_prompt_overrides"] = \
        st.session_state.get("agent_prompt_overrides", {})

    # deliberation runs are single-topic by default
    if topic:
        st.session_state.dataset_df = pd.DataFrame([{"topic": topic}])
        st.session_state.column_mapping = {"topic": "topic"}
        st.session_state.experiment_config["dataset"] = {
            "source": "single_topic",
            "num_rows": 1,
            "columns": ["topic"],
        }


init_state()

main_col, summary_col = st.columns([2.2, 1], gap="large")

with main_col:
    with st.container(border=True):
        st.subheader("1. Deliberation topic / question")
        st.caption("This is the concrete question sent to every agent at runtime.")
        st.text_area(
            "Topic / question",
            placeholder="Should the city introduce congestion charges on its inner ring road?",
            height=90,
            key="task_description",
        )

    with st.container(border=True):
        st.subheader("2. Base instructions")
        st.caption(
            "These instructions are sent to every agent on every turn, before the topic. "
            "Write them in second person, addressed to the agent."
        )

        st.text_area(
            "Base instructions",
            placeholder=(
                "You are a policy advisor participating in a structured deliberation.\n"
                "Write exactly one paragraph (50–150 words) arguing from your assigned perspective.\n"
                "Be specific - cite mechanisms, consequences, or evidence. Do not summarise other agents' views."
            ),
            height=150,
            key="base_instructions",
        )

        st.text_area(
            "Guideline / definition notes (optional)",
            placeholder=(
                "Add definitions, scoring criteria, or factual context here.\n"
                "E.g.: A congestion charge is a fee levied on vehicles entering a defined urban zone."
            ),
            height=90,
            key="guideline_notes",
        )

    with st.container(border=True):
        st.subheader("3. Agent roster")

        num_agents = st.number_input(
            "Number of agents",
            min_value=1,
            max_value=10,
            value=st.session_state.num_agents,
            step=1,
        )
        st.session_state.num_agents = num_agents
        sync_agents_to_count(num_agents)
        sync_agents_to_config()


        for i in range(st.session_state.num_agents):
            agent = st.session_state.agents[i]
            # ensure role is always "Custom" going forward
            agent["role"] = "Custom"

            with st.expander(f"Agent {i+1}", expanded=True if i < 3 else False):
                c1, c2, c3, c4 = st.columns([3, 2, 3, 2])
                with c1:
                    agent["name"] = st.text_input(
                        "Agent name",
                        value=agent["name"],
                        key=f"agent_name_{i}",
                        placeholder=f"e.g. Economist, Historian, Agent {i+1}",
                    )
                with c2:
                    from core.providers import PROVIDERS, PROVIDER_MODELS
                    # Only one agent can be "Human" at a time
                    other_human_taken = any(
                        st.session_state.agents[j].get("provider") == "Human"
                        for j in range(st.session_state.num_agents) if j != i
                    )
                    provider_options = PROVIDERS if other_human_taken else PROVIDERS + ["Human"]
                    agent["provider"] = st.selectbox(
                        "Provider",
                        provider_options,
                        index=provider_options.index(agent["provider"])
                        if agent["provider"] in provider_options else 0,
                        key=f"agent_provider_{i}",
                        on_change=change_agent_provider,
                        args=(i,),
                    )
                is_human = agent["provider"] == "Human"
                with c3:
                    if is_human:
                        st.text_input("Model", value="— you'll write the contributions —",
                                      disabled=True, key=f"agent_model_disabled_{i}")
                        agent["model"] = ""
                    else:
                        model_options = PROVIDER_MODELS.get(agent["provider"], [])
                        if not agent["model"]:
                            # Switched away from "Human" so fall back to this provider's first model
                            agent["model"] = model_options[0] if model_options else ""
                        # If stored model not in list keep it as free-text fallback
                        if agent["model"] not in model_options:
                            model_options = [agent["model"]] + model_options
                        agent["model"] = st.selectbox(
                            "Model",
                            model_options,
                            index=model_options.index(agent["model"]),
                            key=f"agent_model_{i}",
                        )
                with c4:
                    from core.providers import uses_default_temperature
                    if is_human:
                        agent["temperature"] = 0.0
                    elif uses_default_temperature(agent["provider"], agent["model"]):
                        st.text_input("Temperature", value="Model default", disabled=True,
                                      key=f"agent_temperature_default_{i}",
                                      help="This model uses its default sampling settings.")
                    else:
                        agent["temperature"] = st.number_input(
                            "Temperature",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(agent.get("temperature", 0.0)),
                            step=0.01,
                            key=f"agent_temperature_{i}",
                            help="0 = deterministic. Higher values increase randomness. 1 = very random.",
                        )
                if is_human:
                    st.caption(
                        "You'll be prompted to write this agent's contribution yourself "
                        "each round when the debate runs. It won't do peer review yet but "
                        "it can still be reviewed by the other agents."
                    )

                agent["custom_role"] = st.text_area(
                    "Role description",
                    value=agent.get("custom_role", ""),
                    key=f"agent_custom_role_{i}",
                    height=80,
                    placeholder=(
                        "Describe this agent's perspective, mandate, or persona.\n"
                        "E.g.: You are an economist. Argue from welfare economics, "
                        "prioritise quantitative rigour and cite empirical evidence."
                    ),
                )

    if st.button("Save and continue to Protocol & evaluation →", type="primary", use_container_width=True):
        sync_to_config()
        sync_agents_to_config()
        st.switch_page("pages/3_Instructions.py")

with summary_col:
    with st.container(border=True):
        st.subheader("Summary")
        st.markdown(f"**Topic**  \n{'✅ Set' if st.session_state.get('task_description', '').strip() else '—'}")
        st.markdown(f"**Instructions**  \n{'✅ Set' if st.session_state.get('base_instructions', '').strip() else '—'}")
        st.markdown(f"**Agents**  \n{len(st.session_state.get('agents', []))}")
