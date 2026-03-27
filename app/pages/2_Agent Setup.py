"""AI Agent Setup Page"""

import streamlit as st

st.set_page_config(page_title="Agent Setup", page_icon="🧠", layout="wide")


# ---------- helpers ----------
def init_agent_config():
    if "experiment_config" not in st.session_state:
        st.session_state.experiment_config = {}

    if "agents" not in st.session_state.experiment_config:
        st.session_state.experiment_config["agents"] = []

    if "protocol" not in st.session_state.experiment_config:
        st.session_state.experiment_config["protocol"] = {
            "setting": "Gossip (sequential)",
            "supervision_mode": "Unsupervised",
            "visibility_mode": "Current state only",
            "order_type": "Fixed",
            "initializer_agent": "Agent 1",
            "judge_agent": "Agent 3",
            "max_cycles": 5,
            "stopping_rule": "Either",
        }

def sync_agents_to_config():
    st.session_state.experiment_config["agents"] = st.session_state.agents

    st.session_state.experiment_config["protocol"] = {
        "setting": st.session_state.interaction_setting,
        "supervision_mode": st.session_state.supervision_mode,
        "visibility_mode": st.session_state.visibility_mode,
        "order_type": st.session_state.order_type,
        "initializer_agent": st.session_state.initializer_agent,
        "judge_agent": st.session_state.judge_agent,
        "max_cycles": st.session_state.max_cycles,
        "stopping_rule": st.session_state.stopping_rule,
    }


def init_agent_state():
    protocol = st.session_state.experiment_config.get("protocol", {})
    saved_agents = st.session_state.experiment_config.get("agents", [])

    if "num_agents" not in st.session_state:
        st.session_state.num_agents = len(saved_agents) if saved_agents else 3

    if "agents" not in st.session_state:
        if saved_agents:
            st.session_state.agents = saved_agents
        else:
            st.session_state.agents = [
                {
                    "name": "Agent 1",
                    "provider": "OpenAI",
                    "model": "gpt-4o",
                    "role": "Neutral / Initializer",
                },
                {
                    "name": "Agent 2",
                    "provider": "Anthropic",
                    "model": "claude",
                    "role": "Skeptical reviewer",
                },
                {
                    "name": "Agent 3",
                    "provider": "Google",
                    "model": "gemini",
                    "role": "Conservative validator",
                },
            ]

    if "interaction_setting" not in st.session_state:
        st.session_state.interaction_setting = protocol.get("setting", "Gossip (sequential)")

    if "supervision_mode" not in st.session_state:
        st.session_state.supervision_mode = protocol.get("supervision_mode", "Unsupervised")

    if "visibility_mode" not in st.session_state:
        st.session_state.visibility_mode = protocol.get("visibility_mode", "Current state only")

    if "order_type" not in st.session_state:
        st.session_state.order_type = protocol.get("order_type", "Fixed")

    if "initializer_agent" not in st.session_state:
        st.session_state.initializer_agent = protocol.get("initializer_agent", "Agent 1")

    if "max_cycles" not in st.session_state:
        st.session_state.max_cycles = protocol.get("max_cycles", 5)

    if "stopping_rule" not in st.session_state:
        st.session_state.stopping_rule = protocol.get("stopping_rule", "Either")

    if "judge_agent" not in st.session_state:
        st.session_state.judge_agent = protocol.get("judge_agent", "Agent 3")


def sync_agents_to_count(n: int):
    current = st.session_state.agents
    if len(current) < n:
        for i in range(len(current), n):
            current.append(
                {
                    "name": f"Agent {i+1}",
                    "provider": "OpenAI",
                    "model": "gpt-4o",
                    "role": "Neutral / Initializer" if i == 0 else "Skeptical reviewer",
                }
            )
    elif len(current) > n:
        st.session_state.agents = current[:n]


init_agent_config()
init_agent_state()


# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 2 of 5")
st.sidebar.progress(2 / 5)
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

st.title("Agent Setup")
st.caption("Configure the agents and the communication protocol.")


main_col, summary_col = st.columns([2.2, 1], gap="large")

with main_col:
    with st.container(border=True):
        st.subheader("1. Protocol configuration")

        c1, c2 = st.columns(2)
        with c1:
            st.session_state.interaction_setting = st.selectbox(
                "Interaction setting",
                ["Gossip (sequential)", "Crowd (parallel)", "Duel (debate)", "Court (judge-based)"],
                index=["Gossip (sequential)", "Crowd (parallel)", "Duel (debate)", "Court (judge-based)"].index(
                    st.session_state.interaction_setting
                ),
            )
        with c2:
            st.session_state.supervision_mode = st.selectbox(
                "Supervision mode",
                ["Unsupervised", "Supervised"],
                index=["Unsupervised", "Supervised"].index(st.session_state.supervision_mode),
            )

    with st.container(border=True):
        st.subheader("2. Agent roster")

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

        role_options = [
            "Neutral / Initializer",
            "Skeptical reviewer",
            "Conservative validator",
            "Judge",
            "Custom",
        ]

        provider_options = ["OpenAI", "Anthropic", "Google", "Mistral", "Local", "Custom"]

        for i in range(st.session_state.num_agents):
            agent = st.session_state.agents[i]

            with st.expander(f"Agent {i+1}", expanded=True if i < 3 else False):
                c1, c2 = st.columns(2)
                with c1:
                    agent["name"] = st.text_input(
                        "Agent name",
                        value=agent["name"],
                        key=f"agent_name_{i}",
                    )
                with c2:
                    agent["role"] = st.selectbox(
                        "Role",
                        role_options,
                        index=role_options.index(agent["role"]) if agent["role"] in role_options else 0,
                        key=f"agent_role_{i}",
                    )

                c3, c4 = st.columns(2)
                with c3:
                    agent["provider"] = st.selectbox(
                        "Provider",
                        provider_options,
                        index=provider_options.index(agent["provider"]) if agent["provider"] in provider_options else 0,
                        key=f"agent_provider_{i}",
                    )
                with c4:
                    agent["model"] = st.text_input(
                        "Model",
                        value=agent["model"],
                        key=f"agent_model_{i}",
                    )

                if agent["role"] == "Custom":
                    agent["custom_role"] = st.text_area(
                        "Custom role description",
                        value=agent.get("custom_role", ""),
                        key=f"agent_custom_role_{i}",
                        height=80,
                        placeholder="Describe how this agent should behave.",
                    )

    with st.container(border=True):
        st.subheader("3. Information flow")

        c1, c2 = st.columns(2)
        with c1:
            visibility_options = [
                "Current state only",
                "Previous agent only",
                "Full history",
                "Summary only",
            ]
            st.session_state.visibility_mode = st.selectbox(
                "Information visibility",
                visibility_options,
                index=visibility_options.index(st.session_state.visibility_mode),
            )

        with c2:
            order_options = ["Fixed", "Randomized each cycle"]
            st.session_state.order_type = st.selectbox(
                "Execution order",
                order_options,
                index=order_options.index(st.session_state.order_type),
            )

        agent_names = [agent["name"] for agent in st.session_state.agents]

        if st.session_state.interaction_setting in ["Gossip (sequential)", "Duel (debate)", "Court (judge-based)"]:
            st.session_state.initializer_agent = st.selectbox(
                "Initializer / first agent",
                agent_names,
                index=agent_names.index(st.session_state.initializer_agent)
                if st.session_state.initializer_agent in agent_names
                else 0,
            )

        if st.session_state.interaction_setting == "Court (judge-based)":
            st.session_state.judge_agent = st.selectbox(
                "Judge agent",
                agent_names,
                index=agent_names.index(st.session_state.judge_agent)
                if st.session_state.judge_agent in agent_names
                else min(2, len(agent_names) - 1),
            )

    with st.container(border=True):
        st.subheader("4. Stopping rules")

        c1, c2 = st.columns(2)
        with c1:
            st.session_state.max_cycles = st.number_input(
                "Max cycles / rounds",
                min_value=1,
                max_value=50,
                value=st.session_state.max_cycles,
                step=1,
            )
        with c2:
            stopping_options = ["Convergence", "Max cycles", "Either"]
            st.session_state.stopping_rule = st.selectbox(
                "Stopping rule",
                stopping_options,
                index=stopping_options.index(st.session_state.stopping_rule),
            )

        st.info("Next: define evaluation rules and run settings →")

    nav1, nav2, nav3 = st.columns([2, 2, 0.8])
    with nav1:
        if st.button("<- Back"):
            st.switch_page("pages/1_Welcome.py")
    with nav2:
        if st.button("Save draft"):
            sync_agents_to_config()
            st.success("Agent setup saved.")
    with nav3:
        if st.button("Next ->"):
            sync_agents_to_config()
            st.switch_page("pages/3_Instructions.py")
    

with summary_col:
    with st.container(border=True):
        st.subheader("Live summary")
        st.markdown(f"**Setting**  \n{st.session_state.interaction_setting}")
        st.markdown(f"**Platform mode**  \n{st.session_state.platform_mode}")
        st.markdown(f"**Supervision**  \n{st.session_state.supervision_mode}")
        st.markdown(f"**Visibility**  \n{st.session_state.visibility_mode}")
        st.markdown(f"**Order**  \n{st.session_state.order_type}")
        st.markdown(f"**Max cycles**  \n{st.session_state.max_cycles}")
        st.markdown(f"**Stopping rule**  \n{st.session_state.stopping_rule}")

        if st.session_state.interaction_setting in ["Gossip (sequential)", "Duel (debate)", "Court (judge-based)"]:
            st.markdown(f"**Initializer**  \n{st.session_state.initializer_agent}")

        if st.session_state.interaction_setting == "Court (judge-based)":
            st.markdown(f"**Judge**  \n{st.session_state.judge_agent}")

        st.divider()
        st.markdown("**Agents**")
        for agent in st.session_state.agents:
            st.markdown(
                f"- **{agent['name']}** — {agent['provider']} / {agent['model']} / {agent['role']}"
            )

    with st.container(border=True):
        st.subheader("Notes")
        st.markdown(
            """
- roles define behavior, not personality
- visibility controls what each agent can see
- stopping rules define when the protocol ends
"""
        )

