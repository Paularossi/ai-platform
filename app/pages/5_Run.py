"""
5_Run.py  -  Experiment runner

Runs the deliberation on the single configured topic, agent by agent,
round by round, rendered as a plain chat transcript (or, in manual mode,
pauses before every agent turn for review/edit/approval). The debate ends
when the stopping condition defined in the protocol is met for automatic runs,
or manually in manual mode by setting an outcome.
"""

from __future__ import annotations

import html
import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

_AMS = ZoneInfo("Europe/Amsterdam")
from pathlib import Path

import pandas as pd
import streamlit as st

# ── make sure core/ is importable when running from app/ ──────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.utils import require_login
from core.agent import unescape_literal_whitespace
from core.ecu import PeerReviewRound
from core.runner import (
    build_agents, build_ecu_components, build_hub, build_item_data,
    build_peer_reviewer, build_protocol, build_transcript_md, collect_result,
)

require_login() # checks for valid student/tutor id

_ACCENT = "#2E5945"

# The outcome is classified by hand from the log after the debate has finished.
OUTCOME_OPTIONS = [
    "Full consensus",
    "Static equilibrium",
    "Dynamic equilibrium",
    "Chaotic state",
]
OUTCOME_DESCRIPTIONS = {
    "Full consensus": "Full consensus - agents converge to a single stable position.",
    "Static equilibrium": "Static equilibrium - a stable distribution of positions (e.g. 20% vs 80%).",
    "Dynamic equilibrium": "Dynamic equilibrium - a partially stable distribution with periodic shifts between positions.",
    "Chaotic state": "Chaotic state - unstable, non-converging and highly variable outputs.",
}

# In manual mode the debate is over when explicitly stopped (via "Stop here")
_MANUAL_MAX_CYCLES = 999


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
if df is None or len(df) == 0:
    missing.append("No deliberation topic set — go to Step 3")

if missing:
    st.error("Cannot run - please complete setup first:")
    for m in missing:
        st.markdown(f"- {m}")
    if st.button("← Back to Review"):
        st.switch_page("pages/4_Review.py")
    st.stop()

# The platform runs one ongoing deliberation on one topic — the row built from Step 3's topic field.
topic_row = df.iloc[0]
topic_text = str(topic_row.get("topic", "")).strip()

st.markdown(
    f"""
    <div style="border-left: 4px solid {_ACCENT}; padding: 0.85rem 1.25rem;
                margin-bottom: 1rem; background: rgba(46,89,69,0.06); border-radius: 4px;">
        <div style="font-size:0.75rem; letter-spacing:0.08em; text-transform:uppercase;
                    color:{_ACCENT}; font-weight:600; margin-bottom:0.3rem;">Topic</div>
        <div style="font-size:1.15rem; font-style:italic; line-height:1.4;">{html.escape(topic_text)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Run configuration panel ───────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Run configuration")

    from core.providers import API_KEY_ENV_VARS, reset_provider
    used_providers = sorted({
        a.get("provider", "OpenAI") for a in agents_cfg if a.get("provider") != "Human"
    })
    api_keys: dict[str, str] = {}
    for provider in used_providers:
        env_var = API_KEY_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")
        if os.environ.get(env_var):
            st.success(f"{provider} API key configured ✓", icon="🔑")
            api_keys[provider] = ""
        else:
            api_keys[provider] = st.text_input(
                f"{provider} API key",
                type="password",
                key=f"api_key_{provider}",
            )

proto = cfg.get("protocol", {})
# Run mode is decided upfront on Agent Setup, not re-toggled here
# A human agent forces the run to pause for its own turns regardless of run mode
human_agent_name = next(
    (a.get("name") for a in agents_cfg if a.get("provider") == "Human"), None
)
has_human = human_agent_name is not None
# A human agent always runs in Manual mode: there's no auto-computed stopping
# condition to fall back on, the student decides when the debate is over and
# what its outcome was, same as any other manual run.
manual_mode = proto.get("run_mode", "Automatic") == "Manual (step-through)" or has_human

if manual_mode and has_human:
    st.caption(
        f"**Manual (step-through)** run, with **{human_agent_name}** played by you — "
        "pause before every turn: yours to write, others to review and approve. "
        "You decide when the debate ends."
    )
elif manual_mode:
    st.caption(
        "**Manual (step-through)** run — pause before every agent turn to review and "
        "edit the prompt, then approve it to send. You decide when the debate ends."
    )
else:
    st.caption(
        "**Automatic** run — plays straight through using the stopping rule below."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Setting", proto.get("setting", "-"))
c2.metric("Agents", len(agents_cfg))
if manual_mode:
    c3.metric("Max cycles", "— (you decide)")
    c4.metric("Stopping rule", "Manual: 'Stop here'")
else:
    c3.metric("Max cycles", proto.get("max_cycles", "-"))
    c4.metric("Stopping rule", proto.get("stopping_rule", "-"))


# ── Outcome + downloads (shared by both run paths) ────────────────────────────

def _sync_debate_to_db(result: dict, experiment_cfg: dict) -> None:
    """
    Save (or, on a later call, update) this debate in the shared Supabase
    log. Called after the outcome or the reflection is saved locally, so
    the DB always reflects whatever's currently in `result`.
    """
    student_id = st.session_state.get("student_id", "").strip()
    if not student_id:
        return
    try:
        from core.db import ensure_schema, save_debate, update_debate_fields
        ensure_schema()
        if "_debate_id" not in result:
            result["_debate_id"] = save_debate(
                result, experiment_cfg, student_id,
                tutorial_group=st.session_state.get("tutorial_group"),
                logged_in_as_tutor=st.session_state.get("is_tutor", False),
            )
        else:
            outcome = result.get("outcome") or {}
            update_debate_fields(
                result["_debate_id"],
                outcome_label=outcome.get("label"),
                outcome_notes=outcome.get("notes"),
                reflection=result.get("reflection"),
            )
    except Exception as exc:
        st.caption(f"Note: this debate wasn't saved to the shared log ({exc}).")


def _render_outcome_and_downloads(results: list[dict], experiment_cfg: dict) -> None:
    result = results[0]
    saved_outcome = result.get("outcome")

    st.divider()
    st.subheader("Outcome")
    st.caption(
        "Record how the deliberation ended. Saving an outcome finishes the debate."
    )

    oc1, oc2 = st.columns([1, 2])
    with oc1:
        default_label = saved_outcome["label"] if saved_outcome else OUTCOME_OPTIONS[0]
        outcome_label = st.selectbox(
            "Outcome",
            OUTCOME_OPTIONS,
            index=OUTCOME_OPTIONS.index(default_label) if default_label in OUTCOME_OPTIONS else 0,
            format_func=lambda x: OUTCOME_DESCRIPTIONS.get(x, x),
            key="outcome_label_select",
        )
        st.caption(
            "These four types describe how a deliberation can settle — all are valid outcomes."
        )
    with oc2:
        outcome_notes = st.text_area(
            "Notes / justification (optional)",
            value=saved_outcome.get("notes", "") if saved_outcome else "",
            height=80,
            key="outcome_notes_input",
        )
    save_label = "💾 Update outcome" if saved_outcome else "🏁 Save outcome & close debate"
    if st.button(save_label, type="primary" if not saved_outcome else "secondary"):
        results[0]["outcome"] = {"label": outcome_label, "notes": outcome_notes}
        st.session_state["run_results"] = results
        _sync_debate_to_db(results[0], experiment_cfg)
        st.toast("Outcome saved.")
        st.rerun()

    st.divider()
    st.subheader("Reflection")
    st.caption(
        "Your own read on why the debate settled this way — saved alongside the transcript."
    )
    saved_reflection = result.get("reflection", "")
    reflection_text = st.text_area(
        "Reflection (optional)",
        value=saved_reflection,
        height=120,
        key="reflection_input",
    )
    if st.button("💾 Update reflection" if saved_reflection else "💾 Save reflection"):
        results[0]["reflection"] = reflection_text
        st.session_state["run_results"] = results
        _sync_debate_to_db(results[0], experiment_cfg)
        st.toast("Reflection saved.")
        st.rerun()

    st.divider()
    st.subheader("Downloads")

    if not saved_outcome:
        st.info("Save an outcome above to finish the debate.")
        return

    timestamp = datetime.now(_AMS).strftime("%Y%m%d_%H%M")
    exp_name = experiment_cfg.get("overview", {}).get("name", "experiment").replace(" ", "_").lower()

    dl1, dl2 = st.columns(2)
    with dl1:
        transcript_md = build_transcript_md(result, experiment_cfg)
        st.download_button(
            "⬇ Transcript (.md)",
            data=transcript_md.encode(),
            file_name=f"{exp_name}_{timestamp}_transcript.md",
            mime="text/markdown",
            use_container_width=True,
            help="A readable, round-by-round record of what was said.",
        )
    with dl2:
        full_json = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "⬇ Full log (.json)",
            data=full_json.encode(),
            file_name=f"{exp_name}_{timestamp}_log.json",
            mime="application/json",
            use_container_width=True,
            help="Everything: contributions, peer reviews, ECU history, Orchestrator updates, and all prompts.",
        )


def _show_run_summary(hub, elapsed: float, show_converged: bool = True) -> None:
    # Tokens spent on LLM calls this debate (contributions + peer review)
    if show_converged:
        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.metric("Turns", hub.num_submissions)
        scol2.metric("Converged", "Yes" if hub.converged else "No")
        scol3.metric("Time", f"{elapsed:.1f}s")
        scol4.metric("Tokens", f"{hub.total_tokens:,}")
    else:
        # Manual mode: the user decides when to stop
        scol1, scol2, scol3 = st.columns(3)
        scol1.metric("Turns", hub.num_submissions)
        scol2.metric("Time", f"{elapsed:.1f}s")
        scol3.metric("Tokens", f"{hub.total_tokens:,}")

    if hub.ecu_balances:
        bal_cols = st.columns(len(hub.ecu_balances))
        for col, (name, bal) in zip(bal_cols, hub.ecu_balances.items()):
            col.metric(name, f"{bal:.3f} ecus")


def _show_nav_buttons():
    st.divider()
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("🔄 Restart debate", use_container_width=True,
                     help="Clear results and run the same experiment again."):
            st.session_state.pop("run_results", None)
            st.session_state.pop("manual_run", None)
            st.rerun()
    with nav2:
        if st.button("✨ Start new experiment", use_container_width=True, type="primary",
                     help="Clear all settings and start from scratch."):
            keys_to_clear = [
                "experiment_config",
                "base_instructions", "guideline_notes", "agent_prompt_overrides",
                "agents", "num_agents", "interaction_setting", "run_mode",
                "visibility_mode", "review_depth", "order_type", "custom_order", "max_cycles",
                "stopping_rule", "initializer_agent",
                "exp_name", "author", "task_description",
                "dataset_df", "column_mapping", "run_results", "manual_run",
                "ecu_enabled", "ecu_info_condition", "ecu_self_assessment",
                "ecu_coalition_threshold", "ecu_dimensions",
                "ecu_orchestrator_enabled", "ecu_orchestrator_every",
            ]
            for key in keys_to_clear:
                st.session_state.pop(key, None)
            st.switch_page("pages/1_Welcome.py")


# ── Shared feed rendering (used by both manual and auto-run) ─────────────────
#
# Rendered as a plain chat transcript — each contribution is one static chat bubble. Peer review / ECU updates are compact
# collapsed blocks rather than bubbles, since they're metadata, not turns.

def _feed_entry_from_event(event, hub) -> dict | None:
    if event.kind == "submission":
        output = event.output
        changed_fields = [f for f, c in output.changed.items() if c]
        return {
            "kind": "submission",
            "cycle": event.cycle,
            "agent_name": event.agent_name,
            "contribution": str(output.contribution) if output.contribution else "*(empty)*",
            "changed_note": f"revised: {', '.join(changed_fields)}" if changed_fields else None,
        }
    if event.kind == "peer_review":
        reviews = [
            r for r in hub.peer_review_log
            if r.cycle == event.cycle and r.reviewer_name == event.agent_name
        ]
        if not reviews:
            return None
        return {"kind": "peer_review", "cycle": event.cycle, "agent_name": event.agent_name, "review": reviews[-1]}
    if event.kind == "ecu_update":
        return {"kind": "ecu_update", "cycle": event.cycle, "balances": dict(hub.ecu_balances)}
    return None


def _render_feed_entry(entry: dict) -> None:
    if entry["kind"] == "submission":
        with st.chat_message(entry["agent_name"]):
            st.markdown(f"**{entry['agent_name']}**")
            st.markdown(entry["contribution"])
            if entry["changed_note"]:
                st.caption(f"✏️ {entry['changed_note']}")
    elif entry["kind"] == "peer_review":
        review = entry["review"]
        with st.expander(f"📋 {entry['agent_name']} — peer review", expanded=False):
            for reviewed, dim_scores in review.scores.items():
                scores_str = ", ".join(f"{d}: {round(s, 2)}" for d, s in dim_scores.items())
                justification = review.justifications.get(reviewed, "")
                st.markdown(f"**{reviewed}** — {scores_str}")
                if justification:
                    st.caption(justification)
            if review.importance_votes:
                vote_parts = "  ·  ".join(f"**{d}**: {round(v)}" for d, v in review.importance_votes.items())
                st.caption(f"Dimension importance votes (out of 100): {vote_parts}")
    elif entry["kind"] == "ecu_update":
        balances = entry["balances"]
        bal_str = ", ".join(f"{k}: {v:.3f}" for k, v in balances.items()) if balances else "—"
        st.caption(f"💰 ECU update — balances: {bal_str}")


def _unescape_chunk_stream(chunks):
    """
    Wrap a chunk stream so literal "\\n"/"\\t" sequences a model may emit
    render as real line breaks live, not as literal backslash-n text on the
    page. A trailing lone backslash is held back to the next chunk in case
    it's the first half of an escape sequence split across a chunk boundary.
    """
    pending = ""
    for chunk in chunks:
        text = pending + chunk
        pending = ""
        if text.endswith("\\") and not text.endswith("\\\\"):
            pending = "\\"
            text = text[:-1]
        yield unescape_literal_whitespace(text)
    if pending:
        yield pending


def _drain_stream_and_render(gen, first_chunk_event):
    """
    Render one agent's response live, ChatGPT-style, as it's generated.

    `first_chunk_event` is a "stream_chunk" RunEvent already pulled from
    `gen`. This pulls the rest of that turn's "stream_chunk" events directly
    off `gen` and feeds them to st.write_stream for a live typing effect,
    stopping at (and returning) the first non-"stream_chunk" event.
    """
    agent_name = first_chunk_event.agent_name
    terminal: dict = {}

    def _raw_chunks():
        yield first_chunk_event.chunk
        while True:
            nxt = next(gen)
            if nxt.kind == "stream_chunk":
                yield nxt.chunk
            else:
                terminal["event"] = nxt
                return

    with st.chat_message(agent_name):
        st.markdown(f"**{agent_name}**")
        st.write_stream(_unescape_chunk_stream(_raw_chunks()))

    return terminal["event"]


# ── Manual step-through mode ────────────────────────────────────────────────
#
# The auto-run loop below drains protocol.run_iter(hub) in a single Streamlit script execution. This block instead keeps the generator alive in
# st.session_state across reruns, pausing on every "dispatch" event so a user can review/edit the exact prompt (or write it themselves) before
# it's sent, and can stop the run whenever they decide the debate is over.

def _advance_manual_run(resume_value: dict | None = None) -> None:
    """
    Resume the stored generator until it either pauses again on a "dispatch"
    event (stored as `state["pending"]`) or finishes (`state["finished"]`).

    resume_value, if given, is delivered via generator.send() to the
    suspended `dispatch` yield in the protocol — see RunEvent's docstring.
    """
    state = st.session_state["manual_run"]
    gen = state["gen"]
    hub = state["hub"]
    auto_ai = state.get("auto_ai", False)
    human_agent_name = state.get("human_agent_name")
    # Render each finished entry to the page as soon as it's produced, instead
    # of only appending to `feed` for the next rerun to draw — otherwise a
    # human's own submission (no stream_chunk to render it) stays invisible
    # while the next AI turn starts streaming right after it.
    last_cycle = state["feed"][-1]["cycle"] if state["feed"] else None
    first = True
    while True:
        try:
            event = gen.send(resume_value if first else None)
        except StopIteration:
            state["finished"] = True
            state["pending"] = None
            return
        first = False
        if event.kind == "dispatch":
            if not auto_ai or event.agent_name == human_agent_name:
                state["pending"] = event
                return
            # auto_ai: AI turn s play it straight through, same as a plain Automatic run
            continue
        if event.cycle != last_cycle:
            st.markdown(f"### Round {event.cycle + 1}")
            last_cycle = event.cycle
        if event.kind == "stream_chunk":
            # Live-render this turn as it's generated; it then becomes a normal static entry in `feed`,
            # same as every other turn, once the response is complete.
            event = _drain_stream_and_render(gen, event)
            entry = _feed_entry_from_event(event, hub)
            if entry:
                state["feed"].append(entry)
            continue
        entry = _feed_entry_from_event(event, hub)
        if entry:
            _render_feed_entry(entry)
            state["feed"].append(entry)


def _start_manual_run(cfg, agents, agent_names, peer_reviewer, item_id, item_data, review_depth,
                       auto_ai: bool = False, human_agent_name: str | None = None) -> None:
    """
    auto_ai=False (true Manual mode): self-paced, pause before every turn —
    no automatic stopping rule and no round cap, only "Stop here" ends the
    debate. See _MANUAL_MAX_CYCLES.

    auto_ai=True (Automatic mode with a human agent aboard): keep the real
    stopping rule and cycle cap — the run only pauses for human_agent_name's
    own turns, everything else plays straight through same as a plain
    Automatic run.
    """
    if auto_ai:
        run_cfg = cfg
    else:
        run_cfg = {
            **cfg,
            "protocol": {**proto, "max_cycles": _MANUAL_MAX_CYCLES, "stopping_rule": "Max cycles"},
        }
    item_ledger, item_coalition, item_orchestrator = build_ecu_components(run_cfg, agent_names, review_depth)
    protocol = build_protocol(
        agents, run_cfg,
        peer_reviewer=peer_reviewer,
        coalition_tracker=item_coalition,
        orchestrator=item_orchestrator,
    )
    hub = build_hub(item_id, item_data, run_cfg, agent_names, item_ledger)

    st.session_state["manual_run"] = {
        "gen": protocol.run_iter(hub),
        "hub": hub,
        "item_ledger": item_ledger,
        "item_coalition": item_coalition,
        "item_orchestrator": item_orchestrator,
        "item_id": item_id,
        "feed": [],
        "pending": None,
        "finished": False,
        "t_start": time.time(),
        "auto_ai": auto_ai,
        "human_agent_name": human_agent_name,
    }
    _advance_manual_run()


def _render_manual_run() -> None:
    state = st.session_state["manual_run"]
    hub = state["hub"]
    human_agent_name = state.get("human_agent_name")

    st.divider()
    last_cycle = None
    for entry in state["feed"]:
        if entry["cycle"] != last_cycle:
            st.markdown(f"### Round {entry['cycle'] + 1}")
            last_cycle = entry["cycle"]
        _render_feed_entry(entry)

    if state["finished"]:
        elapsed = time.time() - state["t_start"]
        # Pure Manual mode is self-paced ("Stop here" decides). Automatic mode with a
        # human aboard still ends via a stopping rule, same as a plain Automatic run.
        _show_run_summary(hub, elapsed, show_converged=state.get("auto_ai", False))
        if "run_results" not in st.session_state:
            result = collect_result(
                hub, state["item_ledger"], state["item_coalition"], state["item_orchestrator"]
            )
            st.session_state["run_results"] = [result]
        return

    pending = state["pending"]
    is_human_turn = human_agent_name is not None and pending.agent_name == human_agent_name
    
    pending_box = st.empty()
    with pending_box.container():
        key_prefix = f"manual_{pending.cycle}_{pending.agent_name}"

        if is_human_turn:
            st.markdown(f"### ⏸ Your turn — **{pending.agent_name}**  ·  Round {pending.cycle + 1}")
            st.caption("Write your contribution for this round, then submit it.")

            history = pending.packet.visible_history
            if history:
                with st.expander("What's been said so far", expanded=True):
                    for h in history:
                        st.markdown(f"**{h.agent_name}** (round {h.cycle + 1}):")
                        st.markdown(str(h.contribution) if h.contribution else "*(empty)*")
            else:
                st.caption("No prior context is visible for this turn — you're contributing blind.")

            human_text = st.text_area(
                "Your contribution", height=200, key=f"{key_prefix}_human_input",
                placeholder="Write what you contribute to the debate this round.",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                submit_clicked = st.button(
                    "✅ Submit", type="primary", key=f"{key_prefix}_submit",
                    use_container_width=True, disabled=not human_text.strip(),
                )
            with col_b:
                stop_clicked = st.button(
                    "⏹ Stop here", key=f"{key_prefix}_stop", use_container_width=True,
                    help="End the debate now instead of submitting this turn.",
                )
        else:
            st.markdown(f"### ⏸ Paused — **{pending.agent_name}**'s turn  ·  Round {pending.cycle + 1}")
            st.caption("Review and edit the exact prompt about to be sent, then approve to send it.")

            system_edit = st.text_area(
                "System prompt", value=pending.prompt["system"], height=150, key=f"{key_prefix}_sys"
            )
            user_edit = st.text_area(
                "User message", value=pending.prompt["user"], height=200, key=f"{key_prefix}_user"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                submit_clicked = st.button(
                    "✅ Approve & send", type="primary", key=f"{key_prefix}_approve",
                    use_container_width=True,
                )
            with col_b:
                stop_clicked = st.button(
                    "⏹ Stop here", key=f"{key_prefix}_stop", use_container_width=True,
                    help="End the debate now. Everything submitted so far is kept; this pending turn is not sent.",
                )

    if submit_clicked:
        pending_box.empty()
        if is_human_turn:
            _advance_manual_run(resume_value={"human_input": human_text})
        else:
            _advance_manual_run(resume_value={"system": system_edit, "user": user_edit})
        st.rerun()
    if stop_clicked:
        hub.check_convergence()
        state["finished"] = True
        state["pending"] = None
        st.rerun()


if (manual_mode or has_human) and "manual_run" in st.session_state:
    _render_manual_run()
    if not st.session_state["manual_run"]["finished"]:
        st.stop()
    _render_outcome_and_downloads(st.session_state["run_results"], cfg)
    _show_nav_buttons()
    st.stop()


# ── Launch button ─────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    launch = st.button("▶ Launch", type="primary", width="stretch")

if not launch:
    # If a previous run's results are stored, show them even without relaunching
    if "run_results" in st.session_state:
        result = st.session_state["run_results"][0]
        st.divider()
        st.subheader("Last run")
        st.caption(f"Turns: {result.get('num_turns', '-')}")
        _render_outcome_and_downloads(st.session_state["run_results"], cfg)
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
    st.markdown(f"**Visibility mode:** {visibility_mode}")
    st.markdown(f"**Peer review depth:** {review_depth}")

    if agents:
        st.markdown("**System prompt preview (Agent 1):**")
        from core.agent import _build_system_prompt
        preview = _build_system_prompt(
            agent_name=agents[0].name,
            agent_role=agents[0].role,
            base_instructions=cfg.get("instructions", {}).get("base_instructions", ""),
            guideline_notes=cfg.get("instructions", {}).get("guideline_notes", ""),
            agent_overrides=cfg.get("agent_prompt_overrides", {}),
        )
        st.code(preview, language=None)

id_col = column_mapping.get("id", None)
item_id = str(topic_row[id_col]) if id_col and id_col in df.columns else "topic"
item_data = build_item_data(topic_row, column_mapping)

if manual_mode or has_human:
    peer_reviewer: PeerReviewRound | None = build_peer_reviewer(cfg, review_depth)
    _start_manual_run(
        cfg, agents, agent_names, peer_reviewer, item_id, item_data, review_depth,
        auto_ai=not manual_mode, human_agent_name=human_agent_name,
    )
    st.rerun()

# ── Build the single run ──────────────────────────────────────────────────────
peer_reviewer = build_peer_reviewer(cfg, review_depth)

item_ledger, item_coalition, item_orchestrator = build_ecu_components(cfg, agent_names, review_depth)
protocol = build_protocol(
    agents, cfg,
    peer_reviewer=peer_reviewer,
    coalition_tracker=item_coalition,
    orchestrator=item_orchestrator,
)
hub = build_hub(item_id, item_data, cfg, agent_names, item_ledger)

st.divider()
st.subheader("Live feed")

max_cycles = int(proto.get("max_cycles", 5))
progress_bar = st.progress(0.0, text="Starting…")

is_crowd = proto.get("setting", "") in ("Simultaneous", "Crowd (parallel)")

t_start = time.time()
_last_cycle_shown = -1
_last_crowd_notice_cycle = -1

gen = protocol.run_iter(hub)
while True:
    event = next(gen, None)
    if event is None:
        break

    if event.cycle != _last_cycle_shown:
        st.markdown(f"### Round {event.cycle + 1}")
        _last_cycle_shown = event.cycle

    if event.kind == "dispatch":
        if is_crowd:
            if event.cycle > 0 and event.cycle != _last_crowd_notice_cycle:
                _last_crowd_notice_cycle = event.cycle
                packet = event.packet
                if packet.visible_history:
                    round_labels = ", ".join(
                        f"{h.agent_name} (round {h.cycle + 1})" for h in packet.visible_history
                    )
                    st.caption(f"Context sent to all agents — previous contributions visible: {round_labels}")
                else:
                    st.caption("Agents contribute blind (no prior context).")
        else:
            # Sequential dispatch — a brief note on what this agent can see
            packet = event.packet
            if packet.visible_history:
                lines = [
                    f"- {h.agent_name}: {str(h.contribution)[:120]}"
                    for h in packet.visible_history
                    if h.cycle == event.cycle
                ]
                if lines:
                    st.caption(f"**{event.agent_name}** sees this round so far:")
                    st.markdown("\n".join(lines))
            else:
                st.caption(f"**{event.agent_name}** is contributing blind (φ₁: {visibility_mode})")
        progress = min((event.cycle + 1) / max_cycles, 1.0)
        progress_bar.progress(progress, text=f"Round {event.cycle + 1} / {max_cycles}")
        continue

    if event.kind == "stream_chunk":
        # The response is rendered as the API actually generates it, not shown all at once when done.
        event = _drain_stream_and_render(gen, event)
        if event.kind == "submission":
            changed_fields = [f for f, c in event.output.changed.items() if c]
            if changed_fields:
                st.caption(f"✏️ revised: {', '.join(changed_fields)}")
        else:
            entry = _feed_entry_from_event(event, hub)
            if entry:
                _render_feed_entry(entry)
    else:
        entry = _feed_entry_from_event(event, hub)
        if entry:
            _render_feed_entry(entry)

    progress = min((event.cycle + 1) / max_cycles, 1.0)
    progress_bar.progress(progress, text=f"Round {event.cycle + 1} / {max_cycles}")

progress_bar.progress(1.0, text="Done!")

elapsed = time.time() - t_start

st.divider()
_show_run_summary(hub, elapsed)

result = collect_result(hub, item_ledger, item_coalition, item_orchestrator)
st.session_state["run_results"] = [result]

_render_outcome_and_downloads(st.session_state["run_results"], cfg)
_show_nav_buttons()
