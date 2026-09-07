"""AI Agent Setup Page"""

import streamlit as st

from components.utils import require_login

require_login() # checks for valid student/tutor id


# ---------- helpers ----------
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
    # Ensure overview is preserved if coming from Main
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
                    "model": "gpt-4o",
                    "temperature": 0.0,
                    "role": "Custom",
                    "custom_role": "",
                },
                {
                    "name": "Agent 2",
                    "provider": "Anthropic",
                    "model": "claude-sonnet-4-6",
                    "temperature": 0.0,
                    "role": "Custom",
                    "custom_role": "",
                },
                {
                    "name": "Agent 3",
                    "provider": "Anthropic",
                    "model": "claude-opus-4-8",
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


init_agent_config()
# A freshly loaded draft re-syncs session_state below even if the keys already
# exist from an earlier visit to this page.
_reload_from_draft = st.session_state.get("_draft_loaded_token") != st.session_state.get("_agent_setup_synced_token")
init_agent_state(_reload_from_draft)
st.session_state["_agent_setup_synced_token"] = st.session_state.get("_draft_loaded_token")


# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 1 of 3")
st.sidebar.progress(1 / 3)
st.sidebar.markdown("""
**Steps**
1. **Agents ← you are here**
2. Instructions & topic
3. Review
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
                ["Simultaneous", "Sequential"],
                index=["Simultaneous", "Sequential"].index(
                    st.session_state.interaction_setting
                ) if st.session_state.interaction_setting in ["Simultaneous", "Sequential"] else 0,
            )
        with c2:
            has_human_agent = any(
                st.session_state.get(f"agent_provider_{i}") == "Human" or a.get("provider") == "Human"
                for i, a in enumerate(st.session_state.agents)
            )
            if has_human_agent:
                st.session_state.run_mode = "Manual (step-through)"
                st.selectbox(
                    "Run mode",
                    ["Automatic", "Manual (step-through)"],
                    index=1,
                    disabled=True,
                    help="Forced to Manual (step-through) — a Human agent is configured "
                         "below, and a human turn always needs to pause for input.",
                )
            else:
                st.session_state.run_mode = st.selectbox(
                    "Run mode",
                    ["Automatic", "Manual (step-through)"],
                    index=["Automatic", "Manual (step-through)"].index(st.session_state.run_mode),
                    help=(
                        "Automatic plays the debate straight through using the stopping "
                        "rule below. Manual pauses before every agent turn so you can "
                        "review and edit the prompt, then approve it to send"
                    ),
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


        for i in range(st.session_state.num_agents):
            agent = st.session_state.agents[i]
            # Ensure role is always "Custom" going forward
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
                    if is_human:
                        agent["temperature"] = 0.0
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
            phi2_options = ["Current Only", "Previous Round", "Full History"]
            phi2_labels = {
                "Current Only": "Current Only — see only this round's contribution",
                "Previous Round": "Previous Round — see current + previous round side-by-side",
                "Full History": "Full History — see the full contribution trajectory",
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



        agent_names = [agent["name"] for agent in st.session_state.agents]

        if st.session_state.interaction_setting == "Sequential":
            with c2:
                order_options = ["Fixed", "Randomized each cycle", "Custom order"]
                st.session_state.order_type = st.selectbox(
                    "Execution order",
                    order_options,
                    index=order_options.index(st.session_state.order_type)
                    if st.session_state.order_type in order_options else 0,
                )

            if st.session_state.order_type == "Custom order":
                st.caption(
                    "Set each agent's position in the speaking order (1 = goes first). "
                    "Ties are broken by the order agents are listed above."
                )
                current_order = [n for n in st.session_state.get("custom_order", []) if n in agent_names]
                current_order += [n for n in agent_names if n not in current_order]
                ranks: dict[str, int] = {}
                rank_cols = st.columns(min(len(agent_names), 4) or 1)
                for i, name in enumerate(agent_names):
                    with rank_cols[i % len(rank_cols)]:
                        ranks[name] = st.number_input(
                            name,
                            min_value=1,
                            max_value=len(agent_names),
                            value=current_order.index(name) + 1,
                            step=1,
                            key=f"custom_order_rank_{name}",
                        )
                st.session_state.custom_order = sorted(agent_names, key=lambda n: (ranks[n], agent_names.index(n)))
                st.caption("Order: " + " → ".join(st.session_state.custom_order))
            else:
                st.session_state.initializer_agent = st.selectbox(
                    "Initializer / first agent",
                    agent_names,
                    index=agent_names.index(st.session_state.initializer_agent)
                    if st.session_state.initializer_agent in agent_names
                    else 0,
                )

    with st.container(border=True):
        st.subheader("4. Stopping rules")

        if st.session_state.run_mode == "Manual (step-through)":
            st.caption(
                "In Manual (step-through) mode you decide when the debate ends — "
                "click 'Stop here' during the run. There's no round limit or "
                "stopping rule to set here."
            )
        else:
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

    with st.container(border=True):
        st.subheader("5. ECU quality dimensions")
        st.caption(
            "Define the quality dimensions used to score each agent contribution. "
            "SW weights $w^{SW}$ are fixed social-planner valuations; ECU weights $w^{ECU}$ are "
            "variable incentive weights used for agent payouts and Orchestrator updates."
        )

        # Load defaults from ecu module
        import sys
        from pathlib import Path
        _ROOT = Path(__file__).resolve().parents[2]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        from core.ecu import DEFAULT_DIMENSIONS

        if "ecu_dimensions" not in st.session_state or _reload_from_draft:
            saved = st.session_state.experiment_config.get("ecu", {})
            if saved.get("dimensions"):
                st.session_state.ecu_dimensions = saved["dimensions"]
            else:
                st.session_state.ecu_dimensions = [
                    {"name": d["name"], "label": d["label"], "weight": 1.0, "sw_weight": 1.0}
                    for d in DEFAULT_DIMENSIONS
                ]
        if "ecu_enabled" not in st.session_state or _reload_from_draft:
            st.session_state.ecu_enabled = st.session_state.experiment_config.get("ecu", {}).get("enabled", False)
        if "ecu_info_condition" not in st.session_state or _reload_from_draft:
            st.session_state.ecu_info_condition = st.session_state.experiment_config.get("ecu", {}).get("info_condition", "opaque")
        if "ecu_self_assessment" not in st.session_state or _reload_from_draft:
            st.session_state.ecu_self_assessment = st.session_state.experiment_config.get("ecu", {}).get("include_self_assessment", False)
        if "ecu_coalition_threshold" not in st.session_state or _reload_from_draft:
            st.session_state.ecu_coalition_threshold = float(st.session_state.experiment_config.get("ecu", {}).get("coalition_threshold", 0.6))

        st.session_state.ecu_enabled = st.toggle(
            "Enable peer review",
            value=st.session_state.ecu_enabled,
            help="Agents score each other's contributions after each round; ECU balances are derived from those scores automatically.",
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
                "The Orchestrator adjusts only the ECU incentive weights $w^{ECU}$. "
                "Social welfare is computed with fixed SW weights $w^{SW}$."
            )

            if "ecu_orchestrator_enabled" not in st.session_state or _reload_from_draft:
                st.session_state.ecu_orchestrator_enabled = st.session_state.experiment_config.get("ecu", {}).get("orchestrator_enabled", False)
            if "ecu_orchestrator_step_size" not in st.session_state or _reload_from_draft:
                st.session_state.ecu_orchestrator_step_size = float(st.session_state.experiment_config.get("ecu", {}).get("orchestrator_step_size", 0.1))
            if "ecu_orchestrator_every" not in st.session_state or _reload_from_draft:
                st.session_state.ecu_orchestrator_every = int(st.session_state.experiment_config.get("ecu", {}).get("orchestrator_every", 2))

            st.session_state.ecu_orchestrator_enabled = st.toggle(
                "Enable Orchestrator weight-updating",
                value=st.session_state.ecu_orchestrator_enabled,
                help=(
                    "Each round, agents vote on which quality dimensions matter most "
                    "(given the topic and their role). Their votes guide gradient-ascent "
                    "updates to the ECU weights w^ECU. Uses a 1/t learning rate — larger updates early, diminishing over time."
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
                        help="Fixed social-planner valuation $w^{SW}_q$.",
                    )
                with c3:
                    dim["weight"] = st.number_input(
                        "ECU weight",
                        min_value=0.0, max_value=10.0,
                        value=float(dim.get("weight", 1.0)),
                        step=0.1,
                        key=f"ecu_w_{dim['name']}",
                        help="Variable incentive weight $w^{ECU}_q$ used for ECU payouts.",
                    )

            total_sw = sum(float(d.get("sw_weight", 1.0)) for d in dims)
            total_ecu = sum(float(d.get("weight", 1.0)) for d in dims)
            st.caption(f"Total $w^{{SW}}$: {total_sw:.1f} · Total $w^{{ECU}}$: {total_ecu:.1f}")

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

    nav1, nav2, nav3 = st.columns([2, 2, 2])
    with nav1:
        if st.button("← Back"):
            st.switch_page("pages/1_Welcome.py")
    with nav2:
        if st.button("Save draft", use_container_width=True):
            sync_agents_to_config()
            st.toast("Agent setup saved.")
    with nav3:
        if st.button("Next →", type="primary", use_container_width=True):
            sync_agents_to_config()
            st.switch_page("pages/3_Instructions.py")
    

with summary_col:
    with st.container(border=True):
        st.subheader("Live summary")
        st.markdown(f"**Setting**  \n{st.session_state.interaction_setting}")
        st.markdown(f"**Run mode**  \n{st.session_state.run_mode}")
        st.markdown(f"**φ₁ Visibility**  \n{st.session_state.visibility_mode}")
        st.markdown(f"**φ₂ Review depth**  \n{st.session_state.get('review_depth', 'previous_round')}")
        if st.session_state.interaction_setting == "Sequential":
            st.markdown(f"**Order**  \n{st.session_state.order_type}")
            if st.session_state.order_type == "Custom order":
                st.markdown(f"**Custom order**  \n{' → '.join(st.session_state.get('custom_order', [])) or '—'}")
            else:
                st.markdown(f"**Initializer**  \n{st.session_state.initializer_agent}")
        if st.session_state.run_mode == "Manual (step-through)":
            st.markdown("**Stopping**  \nYou decide (Manual mode)")
        else:
            st.markdown(f"**Max cycles**  \n{st.session_state.max_cycles}")
            st.markdown(f"**Stopping rule**  \n{st.session_state.stopping_rule}")

        st.divider()
        st.markdown("**Agents**")
        for agent in st.session_state.agents:
            st.markdown(
                f"- **{agent['name']}** - {agent['provider']} / {agent['model']} / {agent['role']}"
            )

