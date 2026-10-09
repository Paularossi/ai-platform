"""Step 2 — Protocol, information flow, stopping rules, ECU, and prompt preview."""

import streamlit as st

from components.utils import require_login
from components.setup_state import init_agent_config, init_agent_state, sync_agents_to_config
import pandas as pd
from core.agent import _build_system_prompt

require_login() # checks for valid student/tutor id


def sync_to_config():
    """Rebuild dataset from saved topic; page 1 owns the instruction fields."""
    topic = st.session_state.experiment_config.get("task", {}).get("description", "").strip()
    if topic:
        st.session_state.dataset_df = pd.DataFrame([{"topic": topic}])
        st.session_state.column_mapping = {"topic": "topic"}
        st.session_state.experiment_config["dataset"] = {
            "source": "single_topic", "num_rows": 1, "columns": ["topic"],
        }


init_agent_config()
# a freshly loaded draft re-syncs session_state below even if the keys already exist from an earlier visit to this page
_reload_from_draft = st.session_state.get("_draft_loaded_token") != st.session_state.get("_agent_setup_synced_token")
init_agent_state(_reload_from_draft)
st.session_state["_agent_setup_synced_token"] = st.session_state.get("_draft_loaded_token")


# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("Step 2 of 3")
st.sidebar.progress(2 / 3)
st.sidebar.markdown("""
**Steps**
1. Topic & Agents
2. **Protocol & Evaluation ← you are here**
3. Review
4. Run
""")
st.title("Protocol & Evaluation")
st.caption("Configure protocol, information flow, stopping rules, ECU dimensions, and prompt preview.")

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
                st.selectbox(
                    "Run mode",
                    ["Automatic", "Manual (step-through)"],
                    key="run_mode",
                    help=(
                        "Automatic plays the debate straight through using the stopping "
                        "rule below. Manual pauses before every agent turn so you can "
                        "review and edit the prompt, then approve it to send"
                    ),
                )

    with st.container(border=True):
        st.subheader("2. Information flow")
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
        st.subheader("3. Stopping rules")

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
        st.subheader("4. ECU quality dimensions")
        st.caption(
            "Define the quality dimensions used to score each agent contribution. "
            "SW weights $w^{SW}$ are fixed social-planner valuations; ECU weights $w^{ECU}$ are "
            "variable incentive weights used for agent payouts and Orchestrator updates."
        )

        # load defaults from ecu module
        import sys
        from pathlib import Path
        _ROOT = Path(__file__).resolve().parents[2]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        from core.ecu import DEFAULT_DIMENSIONS, dimensions_for_mode
        from uuid import uuid4
        automatic = st.session_state.run_mode == "Automatic"

        if "ecu_dimensions" not in st.session_state or _reload_from_draft:
            saved = st.session_state.experiment_config.get("ecu", {})
            if "dimensions" in saved:
                st.session_state.ecu_dimensions = saved["dimensions"]
            else:
                st.session_state.ecu_dimensions = [
                    {**d, "weight": 0.25, "sw_weight": 0.25}
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

        dimension_errors = False
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
            if automatic:
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
            dims = dimensions_for_mode(st.session_state.ecu_dimensions, automatic)
            st.session_state.ecu_dimensions = dims
            st.caption("Rename each dimension and edit its scoring question. Define what scores 0 and 1 mean.")
            if not automatic:
                st.caption("Coalition detection is off. You decide when to stop the debate.")
            for dim in dims:
                editor_id = dim.setdefault("_editor_id", uuid4().hex)
                fixed_consensus = automatic and dim["name"] == "consensus"
                dim.setdefault("weight", 0.25)
                dim.setdefault("sw_weight", 0.25)
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    if fixed_consensus:
                        st.markdown("**Consensus**", help=(
                            "Automatic mode requires Consensus for coalition detection. "
                        ))
                    else:
                        dim["label"] = st.text_input(
                            "Dimension name", value=dim["label"],
                            key=f"dim_label_{automatic}_{editor_id}_{st.session_state.get('_draft_loaded_token', 0)}",
                        )
                with c2:
                    dim["sw_weight"] = st.number_input(
                        "SW weight",
                        min_value=0.0, max_value=1.00,
                        value=min(1.0, max(0.0, float(dim.get("sw_weight", 0.25)))),
                        step=0.05,
                        key=f"sw_w_{editor_id}",
                        help="Fixed social-planner valuation $w^{SW}_q$.",
                    )
                with c3:
                    dim["weight"] = st.number_input(
                        "ECU weight",
                        min_value=0.0, max_value=1.00,
                        value=min(1.0, max(0.0, float(dim.get("weight", 0.25)))),
                        step=0.05,
                        key=f"ecu_w_{editor_id}",
                        help="Variable incentive weight $w^{ECU}_q$ used for ECU payouts.",
                    )

                if fixed_consensus:
                    st.write(dim["rubric"])
                else:
                    dim["rubric"] = st.text_area(
                        "Scoring question / rubric", value=dim["rubric"], height=110,
                        key=f"dim_rubric_{automatic}_{editor_id}_{st.session_state.get('_draft_loaded_token', 0)}",
                    )
                    if st.button("Remove dimension", key=f"remove_dim_{dim['name']}"):
                        st.session_state.ecu_dimensions = [d for d in dims if d["name"] != dim["name"]]
                        st.rerun()
            if st.button("Add dimension"):
                dims.append({"name": "custom_" + uuid4().hex, "label": "", "rubric": "",
                             "weight": 0.25, "sw_weight": 0.25})
                st.rerun()
            dimension_errors = not dims or any(not d["label"].strip() or not d["rubric"].strip() for d in dims)
            if dimension_errors:
                st.error("Give every dimension a name and scoring rubric before continuing.")
            if len({d["label"].strip().casefold() for d in dims}) != len(dims):
                dimension_errors = True
                st.error("Use a different name for each quality dimension.")

            if not dims:
                st.error("Add at least one dimension, or turn off peer review.")

            total_sw = sum(float(d.get("sw_weight", 0.25)) for d in dims)
            total_ecu = sum(float(d.get("weight", 0.25)) for d in dims)
            st.caption(f"Total $w^{{SW}}$: {total_sw:.1f} · Total $w^{{ECU}}$: {total_ecu:.1f}")

        # persist to experiment_config
        st.session_state.experiment_config["ecu"] = {
            "enabled": st.session_state.ecu_enabled,
            "info_condition": st.session_state.get("ecu_info_condition", "opaque"),
            "include_self_assessment": st.session_state.get("ecu_self_assessment", False),
            "coalition_threshold": st.session_state.get("ecu_coalition_threshold", 0.6),
            "orchestrator_enabled": st.session_state.get("ecu_orchestrator_enabled", False),
            "orchestrator_every": st.session_state.get("ecu_orchestrator_every", 2),
            "dimensions": st.session_state.ecu_dimensions,
        }

    with st.expander("Prompt preview", expanded=False):
        st.caption(
            "Shows the system prompt plus the user message structure sent at runtime. "
            "The selected topic is shown below exactly as it will be injected."
        )

        agents = st.session_state.experiment_config.get("agents", [])
        agent_names = [a["name"] for a in agents] if agents else []

        preview_agent = None
        if agent_names:
            sel = st.selectbox("Preview for agent", ["(Base)"] + agent_names)
            if sel != "(Base)":
                preview_agent = sel

        preview_cfg = next((a for a in agents if a["name"] == preview_agent), {}) if preview_agent else {}
        raw_role = preview_cfg.get("role", "Participant")
        custom_role_text = preview_cfg.get("custom_role", "").strip()
        effective_role = custom_role_text if raw_role == "Custom" and custom_role_text else raw_role

        ecu_cfg = st.session_state.experiment_config.get("ecu", {})
        system_prompt = _build_system_prompt(
            agent_name=preview_agent or "Agent",
            agent_role=effective_role,
            base_instructions=st.session_state.experiment_config.get("instructions", {}).get("base_instructions", ""),
            guideline_notes=st.session_state.experiment_config.get("instructions", {}).get("guideline_notes", ""),
            agent_overrides=st.session_state.experiment_config.get("agent_prompt_overrides", {}),
            ecu_info_condition=ecu_cfg.get("info_condition", "opaque"),
            ecu_dimensions=ecu_cfg.get("dimensions", []),
        )
        st.markdown("**System prompt:**")
        st.code(system_prompt, language=None)

        topic = st.session_state.experiment_config.get("task", {}).get("description", "").strip() or "{deliberation topic/question}"
        phi1 = st.session_state.get("visibility_mode", "Previous round")
        if phi1 == "Full history":
            history_header = "=== Contribution history ==="
            history_note = "← all rounds shown per agent"
        else:
            history_header = "=== Previous round ==="
            history_note = "← shown from round 2 onwards (φ₁: Previous round)"

        peer_review_on = st.session_state.experiment_config.get("ecu", {}).get("enabled", False)
        peer_review_lines = (
            "Peer review scores (average received, last round):\n"
            "  Agent 1: depth_breadth: 0.xx, ...\n"
            if peer_review_on else ""
        )
        st.markdown("**User message structure:**")
        st.code(
            f"topic: {topic}\n\n"
            f"{history_header}       {history_note}\n"
            "Contributions:\n"
            "  [Agent 1]: ...\n"
            "  [Agent 2]: ...\n"
            f"{peer_review_lines}"
            "=== Your turn ===",
            language=None,
        )
        if not peer_review_on:
            st.caption("Peer review is off (Agent Setup → ECU quality dimensions), so no peer review scores are sent to agents.")

    nav1, nav2, nav3 = st.columns([2, 2, 2])
    with nav1:
        if st.button("← Back"):
            st.switch_page("pages/2_Agent Setup.py")
    with nav2:
        if st.button("Save setup", use_container_width=True):
            sync_agents_to_config()
            sync_to_config()
            st.toast("Setup saved.")
    with nav3:
        if st.button("Next →", type="primary", use_container_width=True,
                     disabled=st.session_state.ecu_enabled and dimension_errors):
            sync_agents_to_config()
            sync_to_config()
            st.switch_page("pages/4_Review.py")

with summary_col:
    with st.container(border=True):
        st.subheader("Live summary")
        st.markdown(f"**Protocol**  \n{st.session_state.get('interaction_setting', '—')}")
        st.markdown(f"**Run mode**  \n{st.session_state.get('run_mode', '—')}")
        st.markdown(f"**Peer review**  \n{'✅ Enabled' if st.session_state.get('ecu_enabled') else '—'}")
