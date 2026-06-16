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
            "setting": "Simultaneous",
            "supervision_mode": "Unsupervised",
            "visibility_mode": "Previous round",
            "review_depth": "previous_round",
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
        "review_depth": st.session_state.get("review_depth", "previous_round"),
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
                    "role": "Custom",
                    "custom_role": "",
                },
                {
                    "name": "Agent 2",
                    "provider": "Anthropic",
                    "model": "claude",
                    "role": "Custom",
                    "custom_role": "",
                },
                {
                    "name": "Agent 3",
                    "provider": "Google",
                    "model": "gemini",
                    "role": "Custom",
                    "custom_role": "",
                },
            ]

    if "interaction_setting" not in st.session_state:
        st.session_state.interaction_setting = protocol.get("setting", "Simultaneous")

    if "supervision_mode" not in st.session_state:
        st.session_state.supervision_mode = protocol.get("supervision_mode", "Unsupervised")

    if "visibility_mode" not in st.session_state:
        st.session_state.visibility_mode = protocol.get("visibility_mode", "Previous round")
    if "review_depth" not in st.session_state:
        st.session_state.review_depth = protocol.get("review_depth", "previous_round")

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
                    "role": "Custom",
                    "custom_role": "",
                }
            )
    elif len(current) > n:
        st.session_state.agents = current[:n]


init_agent_config()
init_agent_state()


# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 2 of 4")
st.sidebar.progress(2 / 4)
st.sidebar.markdown("""
**Steps**
1. Overview
2. Agents
3. Instructions & topic
4. Review
""")

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
                ["Simultaneous", "Sequential", "Duel (debate)", "Court (judge-based)"],
                index=["Simultaneous", "Sequential", "Duel (debate)", "Court (judge-based)"].index(
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

        provider_options = ["OpenAI", "Anthropic", "Google", "Mistral", "Local"]

        for i in range(st.session_state.num_agents):
            agent = st.session_state.agents[i]
            # Ensure role is always "Custom" going forward
            agent["role"] = "Custom"

            with st.expander(f"Agent {i+1}", expanded=True if i < 3 else False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    agent["name"] = st.text_input(
                        "Agent name",
                        value=agent["name"],
                        key=f"agent_name_{i}",
                        placeholder=f"e.g. Economist, Historian, Agent {i+1}",
                    )
                with c2:
                    agent["provider"] = st.selectbox(
                        "Provider",
                        provider_options,
                        index=provider_options.index(agent["provider"])
                        if agent["provider"] in provider_options else 0,
                        key=f"agent_provider_{i}",
                    )
                with c3:
                    agent["model"] = st.text_input(
                        "Model",
                        value=agent["model"],
                        key=f"agent_model_{i}",
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

    with st.container(border=True):
        st.subheader("3. Information flow")
        st.caption(
            "Two separate settings control what agents see — one for each phase of a round."
        )

        c1, c2 = st.columns(2)
        with c1:
            phi1_options = ["Blind", "Previous round", "Full history"]
            phi1_labels = {
                "Blind": "Blind — agents write without seeing anyone",
                "Previous round": "Previous round — each agent sees everyone's last contribution",
                "Full history": "Full history — each agent sees all contributions across all rounds",
            }
            # Backwards compatibility: map old labels
            current_vis = st.session_state.visibility_mode
            if current_vis in ("Current state only", "Summary only"):
                st.session_state.visibility_mode = "Previous round"
            elif current_vis == "Previous agent only":
                st.session_state.visibility_mode = "Blind"

            st.session_state.visibility_mode = st.selectbox(
                "φ₁ — Phase 1 visibility (before writing contribution)",
                phi1_options,
                format_func=lambda x: phi1_labels[x],
                index=phi1_options.index(st.session_state.visibility_mode)
                if st.session_state.visibility_mode in phi1_options else 1,
                help=(
                    "Controls what each agent sees about other agents' contributions "
                    "before writing their own position statement."
                ),
            )

        with c2:
            phi2_options = ["current_only", "previous_round", "full_history"]
            phi2_labels = {
                "current_only": "Current only — see only this round's contribution",
                "previous_round": "Previous round — see current + previous round side-by-side",
                "full_history": "Full history — see the full contribution trajectory",
            }
            st.session_state.review_depth = st.selectbox(
                "φ₂ — Phase 2 review depth (during peer review)",
                phi2_options,
                format_func=lambda x: phi2_labels[x],
                index=phi2_options.index(st.session_state.review_depth)
                if st.session_state.review_depth in phi2_options else 1,
                help=(
                    "Controls how much of an agent's contribution history a reviewer "
                    "sees when scoring that agent. 'Previous round' enables meaningful "
                    "consensus scoring — the reviewer can assess whether the agent "
                    "changed their position in response to others."
                ),
            )



        with c2:
            order_options = ["Fixed", "Randomized each cycle"]
            st.session_state.order_type = st.selectbox(
                "Execution order",
                order_options,
                index=order_options.index(st.session_state.order_type),
            )

        agent_names = [agent["name"] for agent in st.session_state.agents]

        if st.session_state.interaction_setting in ["Sequential", "Duel (debate)", "Court (judge-based)"]:
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

        st.info("Next: define agent instructions →")

    with st.container(border=True):
        st.subheader("5. ECU quality dimensions")
        st.caption(
            "Define the quality dimensions used to score each agent contribution. "
            "SW weights w^SW are fixed social-planner valuations; ECU weights w^ECU are "
            "variable incentive weights used for agent payouts and Orchestrator updates."
        )

        # Load defaults from ecu module
        import sys
        from pathlib import Path
        _ROOT = Path(__file__).resolve().parents[2]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        from core.ecu import DEFAULT_DIMENSIONS

        if "ecu_dimensions" not in st.session_state:
            saved = st.session_state.experiment_config.get("ecu", {})
            if saved.get("dimensions"):
                st.session_state.ecu_dimensions = saved["dimensions"]
            else:
                st.session_state.ecu_dimensions = [
                    {"name": d["name"], "label": d["label"], "weight": 1.0, "sw_weight": 1.0}
                    for d in DEFAULT_DIMENSIONS
                ]
        if "ecu_enabled" not in st.session_state:
            st.session_state.ecu_enabled = st.session_state.experiment_config.get("ecu", {}).get("enabled", False)
        if "ecu_info_condition" not in st.session_state:
            st.session_state.ecu_info_condition = st.session_state.experiment_config.get("ecu", {}).get("info_condition", "opaque")
        if "ecu_self_assessment" not in st.session_state:
            st.session_state.ecu_self_assessment = st.session_state.experiment_config.get("ecu", {}).get("include_self_assessment", False)
        if "ecu_coalition_threshold" not in st.session_state:
            st.session_state.ecu_coalition_threshold = float(st.session_state.experiment_config.get("ecu", {}).get("coalition_threshold", 0.6))

        st.session_state.ecu_enabled = st.toggle(
            "Enable ECU scoring",
            value=st.session_state.ecu_enabled,
            help="When enabled, agents score each other after each round and ECU balances are updated.",
        )

        if st.session_state.ecu_enabled:
            c1, c2 = st.columns(2)
            with c1:
                info_options = ["opaque", "semi-transparent", "transparent"]
                st.session_state.ecu_info_condition = st.selectbox(
                    "T/S/O information condition",
                    info_options,
                    format_func=lambda x: {
                        "transparent": "T — Transparent (full weights + all balances)",
                        "semi-transparent": "S — Semi-transparent (noisy weights + own balance)",
                        "opaque": "O — Opaque (no ECU/weight information)",
                    }[x],
                    index=info_options.index(st.session_state.ecu_info_condition),
                    help=(
                        "Controls what agents know about the ECU mechanism during Phase 1 and peer review.\n\n"
                        "T (Transparent): agents see the full weight vector w and all agents' balances.\n"
                        "S (Semi-transparent): agents see noisy weight estimates (±20%) and only their own balance.\n"
                        "O (Opaque): agents receive no ECU or weight information — they only see contributions."
                    ),
                )
            with c2:
                st.number_input(
                    "Coalition threshold τ",
                    min_value=0.0, max_value=1.0,
                    value=st.session_state.ecu_coalition_threshold,
                    step=0.05,
                    key="ecu_coalition_threshold",
                    help="Minimum mutual agreement score for two agents to be in the same coalition.",
                )

            st.session_state.ecu_self_assessment = st.toggle(
                "Include self-assessment",
                value=st.session_state.ecu_self_assessment,
                help="Agents also score themselves. Self-score contributes with weight λ=0.5.",
            )

            st.divider()
            st.markdown("**Orchestrator (Social Welfare)**")
            st.caption(
                "The Orchestrator adjusts only the ECU incentive weights w^ECU. "
                "Social welfare is computed with fixed SW weights w^SW."
            )

            if "ecu_orchestrator_enabled" not in st.session_state:
                st.session_state.ecu_orchestrator_enabled = st.session_state.experiment_config.get("ecu", {}).get("orchestrator_enabled", False)
            if "ecu_orchestrator_step_size" not in st.session_state:
                st.session_state.ecu_orchestrator_step_size = float(st.session_state.experiment_config.get("ecu", {}).get("orchestrator_step_size", 0.1))
            if "ecu_orchestrator_every" not in st.session_state:
                st.session_state.ecu_orchestrator_every = int(st.session_state.experiment_config.get("ecu", {}).get("orchestrator_every", 2))

            st.session_state.ecu_orchestrator_enabled = st.toggle(
                "Enable Orchestrator weight-updating",
                value=st.session_state.ecu_orchestrator_enabled,
                help=(
                    "Each round, agents vote on which quality dimensions matter most "
                    "(given the topic and their role). Their votes guide gradient-ascent "
                    "updates to the ECU weights w^ECU. Uses a 1/t learning rate — "
                    "larger updates early, diminishing over time."
                ),
            )
            if st.session_state.ecu_orchestrator_enabled:
                st.number_input(
                    "Update every K rounds", min_value=1, max_value=10,
                    value=st.session_state.ecu_orchestrator_every,
                    step=1,
                    key="ecu_orchestrator_every",
                    help="Collect importance votes and update weights every K rounds (default: 1 = every round).",
                )

            st.markdown("**Dimensions and weights**")
            dims = st.session_state.ecu_dimensions
            for dim in dims:
                dim.setdefault("sw_weight", 1.0)
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    st.markdown(f"**{dim['label']}** (`{dim['name']}`)")
                with c2:
                    dim["sw_weight"] = st.number_input(
                        "SW weight",
                        min_value=0.0, max_value=10.0,
                        value=float(dim.get("sw_weight", 1.0)),
                        step=0.1,
                        key=f"sw_w_{dim['name']}",
                        help="Fixed social-planner valuation w^SW_q.",
                    )
                with c3:
                    dim["weight"] = st.number_input(
                        "ECU weight",
                        min_value=0.0, max_value=10.0,
                        value=float(dim.get("weight", 1.0)),
                        step=0.1,
                        key=f"ecu_w_{dim['name']}",
                        help="Variable incentive weight w^ECU_q used for ECU payouts.",
                    )

            total_sw = sum(float(d.get("sw_weight", 1.0)) for d in dims)
            total_ecu = sum(float(d.get("weight", 1.0)) for d in dims)
            st.caption(f"Total SW weight: {total_sw:.1f} · Total ECU weight: {total_ecu:.1f}")

        # Persist to experiment_config
        st.session_state.experiment_config["ecu"] = {
            "enabled": st.session_state.ecu_enabled,
            "info_condition": st.session_state.get("ecu_info_condition", "opaque"),
            "include_self_assessment": st.session_state.get("ecu_self_assessment", False),
            "coalition_threshold": st.session_state.get("ecu_coalition_threshold", 0.6),
            "orchestrator_enabled": st.session_state.get("ecu_orchestrator_enabled", False),
            "orchestrator_step_size": st.session_state.get("ecu_orchestrator_step_size", 0.1),
            "orchestrator_every": st.session_state.get("ecu_orchestrator_every", 2),
            "dimensions": st.session_state.ecu_dimensions,
        }

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
        st.markdown(f"**Supervision**  \n{st.session_state.supervision_mode}")
        st.markdown(f"**φ₁ Visibility**  \n{st.session_state.visibility_mode}")
        st.markdown(f"**φ₂ Review depth**  \n{st.session_state.get('review_depth', 'previous_round')}")
        st.markdown(f"**Order**  \n{st.session_state.order_type}")
        st.markdown(f"**Max cycles**  \n{st.session_state.max_cycles}")
        st.markdown(f"**Stopping rule**  \n{st.session_state.stopping_rule}")

        if st.session_state.interaction_setting in ["Sequential", "Duel (debate)", "Court (judge-based)"]:
            st.markdown(f"**Initializer**  \n{st.session_state.initializer_agent}")

        if st.session_state.interaction_setting == "Court (judge-based)":
            st.markdown(f"**Judge**  \n{st.session_state.judge_agent}")

        st.divider()
        st.markdown("**Agents**")
        for agent in st.session_state.agents:
            st.markdown(
                f"- **{agent['name']}** - {agent['provider']} / {agent['model']} / {agent['role']}"
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