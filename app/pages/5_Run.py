"""
5_Run.py  -  Experiment runner

Streams the gossip protocol live, item by item, agent by agent.
Shows a live feed of each agent turn, then a per-item results table,
and finally lets you download the full results as CSV + JSON.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

_AMS = ZoneInfo("Europe/Amsterdam")
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── make sure core/ is importable when running from app/ ──────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ecu import PeerReviewRound
from core.runner import build_agents, build_ecu_components, build_hub, build_item_data, build_peer_reviewer, build_protocol, collect_result, results_to_df

st.set_page_config(page_title="Run Experiment", page_icon="🚀", layout="wide")

# ---------- sidebar ----------
st.sidebar.title("Experiment Builder")
st.sidebar.caption("▶ Running")
st.sidebar.progress(1.0)
st.sidebar.markdown("""
**Steps**
1. Overview
2. Agents
3. Instructions & topic
4. Review
5. **Run ← you are here**
"""
)


# ---------- helpers ----------

def get_config() -> dict:
    return st.session_state.get("experiment_config", {})


# ---------- page ----------

st.title("🚀 Run Experiment")
cfg = get_config()

# ── Guard: check experiment is configured ─────────────────────────────────────
agents_cfg = cfg.get("agents", [])
df: pd.DataFrame | None = st.session_state.get("dataset_df")
column_mapping: dict = st.session_state.get("column_mapping", {})

missing = []
if not agents_cfg:
    missing.append("No agents configured — go to Step 2")
if df is None:
    missing.append("No deliberation topic set — go to Step 3")

if missing:
    st.error("Cannot run - please complete setup first:")
    for m in missing:
        st.markdown(f"- {m}")
    if st.button("← Back to Review"):
        st.switch_page("pages/4_Review.py")
    st.stop()

# ── Run configuration panel ───────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Run configuration")

    col1, col2 = st.columns([2, 3])

    with col1:
        n_items = st.number_input(
            "Items to process",
            min_value=1,
            max_value=len(df),
            value=min(5, len(df)),
            step=1,
            help=f"Dataset has {len(df)} rows total.",
        )

    with col2:
        from core.providers import API_KEY_ENV_VARS, reset_provider
        used_providers = sorted({a.get("provider", "OpenAI") for a in agents_cfg})
        api_keys: dict[str, str] = {}
        for provider in used_providers:
            env_var = API_KEY_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")
            api_keys[provider] = st.text_input(
                f"{provider} API key",
                type="password",
                value=os.environ.get(env_var, ""),
                key=f"api_key_{provider}",
                help=f"Leave blank if {env_var} is already set in your environment.",
            )

proto = cfg.get("protocol", {})
c1, c2, c3, c4 = st.columns([5, 2, 2, 3])
c1.metric("Setting", proto.get("setting", "-"))
c2.metric("Agents", len(agents_cfg))
c3.metric("Max cycles", proto.get("max_cycles", "-"))
c4.metric("Stopping rule", proto.get("stopping_rule", "-"))

# ── Results renderer (called both after run and on download rerun) ────────────
def _show_results(results: list[dict], experiment_cfg: dict) -> None:
    st.divider()
    st.subheader("Results")

    col1, col2 = st.columns(2)
    col1.metric("Items processed", len(results))
    col2.metric("Total turns", sum(r["num_turns"] for r in results))

    results_df = results_to_df(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    timestamp = datetime.now(_AMS).strftime("%Y%m%d_%H%M")
    exp_name = experiment_cfg.get("overview", {}).get("name", "experiment").replace(" ", "_").lower()

    dl1, dl2 = st.columns(2)
    with dl1:
        csv_bytes = results_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Results CSV",
            data=csv_bytes,
            file_name=f"{exp_name}_{timestamp}_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        full_json = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "⬇ Full log JSON",
            data=full_json.encode(),
            file_name=f"{exp_name}_{timestamp}_log.json",
            mime="application/json",
            use_container_width=True,
            help="Includes contributions, peer reviews, ECU history, Orchestrator updates, and all prompts.",
        )


def _show_nav_buttons():
    st.divider()
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("🔄 Restart debate", use_container_width=True,
                     help="Clear results and run the same experiment again."):
            st.session_state.pop("run_results", None)
            st.rerun()
    with nav2:
        if st.button("✨ Start new experiment", use_container_width=True, type="primary",
                     help="Clear all settings and start from scratch."):
            keys_to_clear = [
                "experiment_config", "question_sets",
                "base_instructions", "guideline_notes", "agent_prompt_overrides",
                "agents", "num_agents", "interaction_setting", "supervision_mode",
                "visibility_mode", "review_depth", "order_type", "max_cycles",
                "stopping_rule", "initializer_agent", "judge_agent",
                "exp_name", "author", "protocol_id", "task_description",
                "dataset_df", "column_mapping", "run_results",
                "ecu_enabled", "ecu_info_condition", "ecu_self_assessment",
                "ecu_coalition_threshold", "ecu_dimensions",
                "ecu_orchestrator_enabled", "ecu_orchestrator_every",
            ]
            for key in keys_to_clear:
                st.session_state.pop(key, None)
            st.switch_page("pages/1_Welcome.py")


# ── Launch button ─────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    launch = st.button("▶ Launch", type="primary", width="stretch")

if not launch:
    # If a previous run's results are stored, show them even without relaunching
    if "run_results" in st.session_state:
        _show_results(st.session_state["run_results"], cfg)
        _show_nav_buttons()
    st.stop()

# Clear any previous results so a fresh run always starts clean
st.session_state.pop("run_results", None)

# ── Set API keys if provided ──────────────────────────────────────────────────
for _provider, _key in api_keys.items():
    if _key:
        _env = API_KEY_ENV_VARS.get(_provider, f"{_provider.upper()}_API_KEY")
        os.environ[_env] = _key
        reset_provider(_provider)

# ── Build agents ──────────────────────────────────────────────────────────────
agents = build_agents(cfg)
agent_names = [a.name for a in agents]
visibility_mode = proto.get("visibility_mode", "Previous round")
review_depth = proto.get("review_depth", "Previous Round")

# ── Diagnostics ───────────────────────────────────────────────────────────────
with st.expander("🔍 Pre-run diagnostics", expanded=False):
    st.markdown("**Agents:**")
    for a in agents:
        st.caption(f"• **{a.name}** — {a.provider} / {a.model} / temp={a.temperature}")
    st.markdown(f"**Visibility mode: ()** {visibility_mode}")
    st.markdown(f"**Peer review depth: ()** {review_depth}")

    if agents:
        st.markdown("**System prompt preview (Agent 1):**")
        from core.agent import _build_system_prompt
        preview = _build_system_prompt(
            agent_name=agents[0].name,
            agent_role=agents[0].role,
            base_instructions=cfg.get("instructions", {}).get("base_instructions", ""),
            guideline_notes=cfg.get("instructions", {}).get("guideline_notes", ""),
            agent_overrides=cfg.get("agent_prompt_overrides", {}),
            questions=[],
        )
        st.code(preview, language=None)

# ── Build shared peer reviewer (reused across items, one per experiment) ─────
peer_reviewer: PeerReviewRound | None = build_peer_reviewer(cfg, review_depth)

# ── Slice dataset ─────────────────────────────────────────────────────────────
subset = df.head(int(n_items)).reset_index(drop=True)

# ── Results accumulator ───────────────────────────────────────────────────────
all_results: list[dict] = []

st.divider()
st.subheader(f"Running on {len(subset)} item(s)…")

progress_bar = st.progress(0.0, text="Starting…")
results_placeholder = st.empty()

# ── Per-item status table (updates live) ──────────────────────────────────────
status_rows: list[dict] = []

# ── Main loop ─────────────────────────────────────────────────────────────────
for item_idx, row in subset.iterrows():
    id_col = column_mapping.get("id", None)
    if id_col and id_col in df.columns:
        item_id = str(row[id_col])
    elif len(df.columns) > 0:
        item_id = str(row[df.columns[0]])
    else:
        item_id = f"item_{item_idx}"

    item_data = build_item_data(row, column_mapping)

    item_ledger, item_coalition, item_orchestrator = build_ecu_components(
        cfg, agent_names, review_depth
    )
    protocol = build_protocol(
        agents, cfg,
        peer_reviewer=peer_reviewer,
        coalition_tracker=item_coalition,
        orchestrator=item_orchestrator,
    )
    hub = build_hub(item_id, item_data, cfg, agent_names, item_ledger)

    # ── Live turn feed ─────────────────────────────────────────────────────
    with st.expander(f"📄 Item {item_idx + 1} / {len(subset)}  -  `{item_id}`", expanded=True):
        # Each event appends a new permanent container rather than rebuilding
        # one giant markdown string. This ensures tables render correctly.
        feed_container = st.container()

        t_start = time.time()

        is_crowd = cfg.get("protocol", {}).get("setting", "") in ("Simultaneous", "Crowd (parallel)")
        _last_crowd_cycle_rendered = -1

        for event in protocol.run_iter(hub):

            if event.kind == "dispatch":
                if is_crowd:
                    if event.cycle > 0 and event.cycle != _last_crowd_cycle_rendered:
                        _last_crowd_cycle_rendered = event.cycle
                        packet = event.packet
                        with feed_container:
                            st.markdown(f"#### 📬 Round {event.cycle + 1} · Context sent to all agents")
                            if packet.visible_history:
                                round_labels = ", ".join(
                                    f"{h.agent_name} (round {h.cycle + 1})"
                                    for h in packet.visible_history
                                )
                                st.caption(f"Previous contributions visible: {round_labels}")
                            else:
                                st.caption("Round 1 — agents contribute blind (no prior context).")
                    continue

                # Sequential dispatch
                packet = event.packet
                with feed_container:
                    st.markdown(f"#### 🔀 Hub → **{event.agent_name}**  ·  Round {event.cycle + 1}")
                    if packet.visible_history:
                        lines = [f"- Round {h.cycle+1} · **{h.agent_name}**: {str(h.contribution)[:120]}"
                                 for h in packet.visible_history]
                        st.markdown("\n".join(lines))
                    else:
                        st.caption(f"Round 1 — contributing blind (φ₁: {visibility_mode})")

            elif event.kind == "submission":
                output = event.output
                changed_fields = [f for f, c in output.changed.items() if c]
                change_note = f"✏️ revised: {', '.join(changed_fields)}" if changed_fields else "✔ no changes"
                with feed_container:
                    st.markdown(f"#### 📨 **{event.agent_name}** → Hub  ·  Round {event.cycle + 1}  ·  {change_note}")
                    contrib = str(output.contribution) if output.contribution else "*(empty)*"
                    st.markdown(contrib)
                    if not output.contribution and output.raw_response:
                        st.code(output.raw_response, language=None)

            elif event.kind == "peer_review":
                round_reviews = [
                    r for r in hub.peer_review_log
                    if r.cycle == event.cycle and r.reviewer_name == event.agent_name
                ]
                if round_reviews:
                    review = round_reviews[-1]
                    with feed_container:
                        st.markdown(f"#### 📋 **{event.agent_name}** peer review  ·  Round {event.cycle + 1}")

                        if review.scores:
                            import pandas as _pd
                            dim_names = list(next(iter(review.scores.values())).keys())
                            rows = []
                            for reviewed, dim_scores in review.scores.items():
                                row_d = {"Agent": reviewed}
                                for d in dim_names:
                                    row_d[d] = round(dim_scores.get(d, 0), 2)
                                row_d["Justification"] = review.justifications.get(reviewed, "")
                                rows.append(row_d)
                            if review.self_scores:
                                self_row = {"Agent": "*(self)*"}
                                for d in dim_names:
                                    self_row[d] = round(review.self_scores.get(d, 0), 2)
                                self_row["Justification"] = ""
                                rows.append(self_row)

                            st.dataframe(
                                _pd.DataFrame(rows),
                                hide_index=True,
                                use_container_width=True,
                                column_config={
                                    "Justification": st.column_config.TextColumn(width="large"),
                                },
                            )

                        if review.importance_votes:
                            vote_parts = "  ·  ".join(
                                f"**{d}**: {round(v)}" for d, v in review.importance_votes.items()
                            )
                            st.caption(f"Dimension importance votes (out of 100): {vote_parts}")

            elif event.kind == "ecu_update":
                balances = hub.ecu_balances
                with feed_container:
                    st.markdown(f"#### 💰 ECU update  ·  after Round {event.cycle + 1}")

                    if item_ledger:
                        round_records = [r for r in item_ledger.history if r.cycle == event.cycle]
                        if round_records:
                            dim_names = list(round_records[0].aggregated_scores.keys())
                            score_rows = []
                            for rec in round_records:
                                r_d = {"Agent": rec.agent_name}
                                for d in dim_names:
                                    r_d[d] = round(rec.aggregated_scores.get(d, 0), 2)
                                r_d["ECU earned"] = round(rec.ecu_earned, 3)
                                score_rows.append(r_d)
                            st.dataframe(pd.DataFrame(score_rows), hide_index=True, width="content")

                    # Balances, SW, coalition, orchestrator as metric row + captions
                    if balances:
                        bal_cols = st.columns(len(balances))
                        for col, (name, bal) in zip(bal_cols, balances.items()):
                            col.metric(f"{name} (cumulative)", f"{bal:.3f} ecus")

                    sw_entry = None
                    if item_ledger and item_ledger.social_welfare_history:
                        sw_entry = next(
                            (e for e in reversed(item_ledger.social_welfare_history)
                             if e["cycle"] == event.cycle), None,
                        )
                    if sw_entry:
                        sw_w = "  ,  ".join(f"{k}={v:.2f}" for k, v in sw_entry["sw_weights"].items())
                        st.caption(f"SW = {sw_entry['social_welfare']:.4f}  |  w^SW (fixed): {sw_w}")

                    if item_coalition and item_coalition.history:
                        last_c = item_coalition.history[-1]
                        c = last_c.get("coalition", [])
                        coalition_label = ", ".join(c) if len(c) >= 2 else "none"
                        st.caption(f"Coalition (τ={item_coalition.threshold}): {coalition_label}")

                    if item_orchestrator:
                        cycle_updates = [u for u in item_orchestrator.history if u.cycle == event.cycle]
                        if cycle_updates:
                            u = cycle_updates[-1]
                            dim_names = list(u.mean_votes.keys())
                            orch_rows = [
                                {"": "votes (out of 100)", **{d: round(u.mean_votes[d], 1) for d in dim_names}},
                                {"": "old ECU weights",  **{d: round(u.old_weights[d], 3) for d in dim_names}},
                                {"": "new ECU weights",  **{d: round(u.new_weights[d], 3) for d in dim_names}},
                            ]
                            st.markdown(f"**Orchestrator** — learning rate: {u.learning_rate:.4f}")
                            st.dataframe(pd.DataFrame(orch_rows), hide_index=True, width="content")

        elapsed = time.time() - t_start

        st.success(f"✅ Done  ·  {hub.num_submissions} turn(s)  ·  {elapsed:.1f}s")

        # ECU balances summary
        if hub.ecu_balances:
            bal_cols = st.columns(len(hub.ecu_balances))
            for col, (name, bal) in zip(bal_cols, hub.ecu_balances.items()):
                col.metric(name, f"{bal:.3f} ecus")

        # Orchestrator weight trajectory (debug)
        if item_orchestrator and item_orchestrator.history:
            with st.expander("📊 Orchestrator weight trajectory", expanded=False):
                if item_ledger and item_ledger.social_welfare_history:
                    st.markdown("**Social welfare trajectory**")
                    st.dataframe(
                        pd.DataFrame(item_ledger.social_welfare_history),
                        hide_index=True, use_container_width=True,
                    )
                st.markdown("**Importance-vote weight updates**")
                update_rows = []
                for u in item_orchestrator.history:
                    row = {
                        "Round": u.cycle + 1,
                        "lr": round(u.learning_rate, 4),
                    }
                    for d, v in u.mean_votes.items():
                        row[f"vote_{d}"] = round(v, 1)
                    for d, v in u.new_weights.items():
                        row[f"w_{d}"] = round(v, 3)
                    update_rows.append(row)
                st.dataframe(
                    pd.DataFrame(update_rows), hide_index=True, use_container_width=True
                )
                st.caption(
                    "vote_* = mean importance points (sum ≈ 100).  "
                    "w_* = updated ECU weights after renormalisation.  "
                    "SW weights w^SW are fixed and never shown here."
                )

    all_results.append(collect_result(hub, item_ledger, item_coalition, item_orchestrator))

    status_row: dict = {
        "Item": item_id,
        "Converged": "✅" if hub.converged else "⏹",
        "Turns": hub.num_submissions,
        "Time (s)": f"{elapsed:.1f}",
    }
    if hub.ecu_balances:
        for name, bal in hub.ecu_balances.items():
            status_row[f"ECU {name}"] = f"{bal:.2f}"
    status_rows.append(status_row)

    # Update progress
    progress = (item_idx + 1) / len(subset)
    progress_bar.progress(progress, text=f"{item_idx + 1} / {len(subset)} items done")

    # Update live summary table
    results_placeholder.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True,
    )

progress_bar.progress(1.0, text="Done!")

# Save results so they survive the download-button rerun
st.session_state["run_results"] = all_results

# ── Summary + download ────────────────────────────────────────────────────────
_show_results(all_results, cfg)
_show_nav_buttons()

# TODO:
# - check for convergence -> if coalition is formed then stop ?
# - gemini not working properly, either fix or remove (importance votes and justifications are not returned)