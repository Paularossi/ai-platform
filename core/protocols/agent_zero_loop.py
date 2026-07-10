"""
core/protocols/agent_zero_loop.py

Agent 0 mode's orchestration loop — replaces the fixed-roster protocol
iteration (CrowdProtocol/GossipProtocol) with a dynamic loop:

    roster = agent_zero.initialize(topic)              # one-time call
    while cycle < max_rounds and total_spawns <= max_total_spawns:
        run one debate round with the current roster    # Phase 1 + Phase 2,
                                                          # reusing existing
                                                          # Agent/PeerReviewRound/
                                                          # EcuLedger/CoalitionTracker
        decision = agent_zero.decide(context)
        apply decision (add/remove agents, subject to hard caps)
        if decision.end_debate or a hard bound was hit: stop

All contribution/peer-review/ECU mechanics are identical to manual mode —
this module only owns the outer loop and Agent 0 integration. The debate
round style is fixed to Simultaneous (Crowd-style) for now; Agent 0 does not
yet choose the protocol.

The Orchestrator (ECU weight search) is never constructed in this mode —
ECU weights stay fixed at their initial config values. Social welfare is
still computed and logged every round, and is one of the signals fed to
Agent 0, but nothing here optimises it directly.
"""

from __future__ import annotations

import difflib
import json
from typing import Any

from core.agent import Agent
from core.agent_zero import AgentZero
from core.ecu import CoalitionTracker, EcuLedger, PeerReviewRound
from core.hub import CommunicationHub
from core.protocols import build_ecu_info_str
from core.providers import get_provider


def _build_agent_notes(
    hub: CommunicationHub,
    ledger: EcuLedger | None,
) -> dict[str, str]:
    """
    Compact per-agent trajectory note for Agent 0's context — how similar an
    agent's latest contribution is to its own previous round, plus its ECU
    trend. This is bookkeeping for Agent 0 only: it does not touch the
    debating agents' own φ1/φ2 visibility settings or prompts.
    """
    notes: dict[str, str] = {}
    for name in hub.agent_names:
        subs = sorted(hub.submissions_by_agent(name), key=lambda o: o.cycle)
        if len(subs) >= 2:
            prev_text = str(subs[-2].contribution or "")
            curr_text = str(subs[-1].contribution or "")
            ratio = difflib.SequenceMatcher(None, prev_text, curr_text).ratio()
            note = f"{ratio * 100:.0f}% similar to its own previous round"
        else:
            note = "first contribution"

        if ledger:
            recs = [r for r in ledger.history if r.agent_name == name]
            if len(recs) >= 2:
                delta = recs[-1].ecu_earned - recs[-2].ecu_earned
                trend = "up" if delta > 0.01 else ("down" if delta < -0.01 else "flat")
                note += f"; ECU trend {trend} ({recs[-2].ecu_earned:.2f} -> {recs[-1].ecu_earned:.2f})"
            elif recs:
                note += f"; ECU earned {recs[-1].ecu_earned:.2f}"

        notes[name] = note
    return notes


def _run_debate_round(
    hub: CommunicationHub,
    cycle: int,
    agents_by_name: dict[str, Agent],
    dry_run: bool,
) -> dict[str, Any]:
    """Phase 1 — simultaneous blind contributions. Returns {agent_name: contribution}."""
    current_agents = [agents_by_name[n] for n in hub.agent_names if n in agents_by_name]
    packets = [(a, hub.build_context(a.name, cycle)) for a in current_agents]

    round_outputs = []
    for agent, packet in packets:
        output = agent.call(packet, dry_run=dry_run)
        hub.submit(output)
        round_outputs.append(output)
        hub.log_prompt(
            cycle, agent.name, "contribution",
            prompt=f"[SYSTEM]\n{agent._last_system_prompt}\n\n[USER]\n{agent._last_user_message}",
            response=output.raw_response or "",
        )

    return {o.agent_name: o.contribution for o in round_outputs}


def _run_peer_review_round(
    hub: CommunicationHub,
    cycle: int,
    all_contributions: dict[str, Any],
    agents_by_name: dict[str, Agent],
    peer_reviewer: PeerReviewRound,
    coalition_tracker: CoalitionTracker | None,
    ledger: EcuLedger | None,
    dry_run: bool,
) -> None:
    """
    Phase 2 — peer review. Reuses hub.compute_ecus_for_round() and
    ledger.record_social_welfare(), which already read the roster dynamically
    (hub.agent_names), so no fixed-n assumptions leak in here.
    """
    if not all_contributions:
        return

    item_context = ", ".join(f"{k}: {v}" for k, v in hub.item_data.items())
    agent_histories = hub.build_review_history(cycle)
    current_agents = [agents_by_name[n] for n in hub.agent_names if n in agents_by_name]

    for agent in current_agents:
        ecu_info = build_ecu_info_str(hub, cycle, agent.name)
        if dry_run:
            review = peer_reviewer.parse(
                reviewer_name=agent.name, cycle=cycle, raw="[dry-run]",
                all_contributions=all_contributions,
            )
        else:
            prompt = peer_reviewer.build_prompt(
                reviewer_name=agent.name,
                reviewer_contribution=all_contributions.get(agent.name, ""),
                all_contributions=all_contributions,
                cycle=cycle,
                item_context=item_context,
                ecu_info=ecu_info,
                reviewer_role=agent.role,
                agent_histories=agent_histories,
                collect_importance_votes=False,  # Orchestrator disabled in this mode
            )
            try:
                pr_kwargs = {"json_mode": True} if agent.provider == "Google" else {}
                raw = get_provider(agent.provider).complete(
                    model=agent.model,
                    system_prompt="You are a helpful assistant evaluating contributions in a deliberation experiment. Read the evaluation instructions carefully and respond with the requested JSON.",
                    user_message=prompt,
                    max_tokens=800,
                    temperature=agent.temperature,
                    **pr_kwargs,
                )
            except Exception as exc:
                raw = f"[ERROR: {exc}]"
            hub.log_prompt(cycle, agent.name, "peer_review", prompt=prompt, response=raw)
            review = peer_reviewer.parse(
                reviewer_name=agent.name, cycle=cycle, raw=raw,
                all_contributions=all_contributions,
            )
        hub.submit_peer_review(review)

    round_reviews = [r for r in hub.peer_review_log if r.cycle == cycle]
    if coalition_tracker:
        coalition_tracker.find_coalition(round_reviews)
    if ledger:
        hub.compute_ecus_for_round(cycle)
        ledger.record_social_welfare(cycle, round_reviews)


def run_agent_zero_experiment(
    cfg: dict,
    dry_run: bool = False,
    on_round: Any = None,
    on_init: Any = None,
) -> dict:
    """
    Run one full Agent 0-mode experiment on a single topic.

    Parameters
    ----------
    cfg : dict
        Experiment config. Must contain cfg["task"]["description"] (the topic),
        cfg["ecu"] (dimensions/thresholds — same shape as manual mode), and
        cfg["agent_zero"] (provider, model, max_rounds, max_agents, max_total_spawns).
    dry_run : bool
        Skip all LLM calls (placeholder outputs), including Agent 0's own calls.
    on_round : callable | None
        Optional callback(round_summary: dict) invoked after each round with
        the same compact record that gets appended to "rounds" in the return
        value — used by callers (CLI, UI) that want progress output without
        re-implementing the loop or reaching into hub/ledger/coalition directly.
    on_init : callable | None
        Optional callback(design_summary: dict) invoked once, right after the
        initialization call and config merge, before the round loop starts —
        lets a caller show "what Agent 0 set up" (roster, instructions,
        dimensions, coalition threshold, φ1/φ2/info-condition) up front.

    Returns
    -------
    dict with "mode", "agent_zero_design" (Agent 0's initialization decision),
    "rounds" (the primary per-round record), "final_brief", "ended_reason",
    "agent_zero_config", and a "debug" section holding the raw contribution/
    peer-review objects, full coalition history, ECU ledger, and every LLM
    prompt+response (including Agent 0's own calls).
    """
    from core.runner import build_ecu_components, build_hub, build_peer_reviewer

    az_cfg = cfg.get("agent_zero", {})
    max_rounds = int(az_cfg.get("max_rounds", 10))
    max_agents = int(az_cfg.get("max_agents", 6))
    max_total_spawns = int(az_cfg.get("max_total_spawns", 8))
    az_provider = az_cfg.get("provider", "Anthropic")
    az_model = az_cfg.get("model", "claude-sonnet-4-6")
    az_temperature = float(az_cfg.get("temperature", 0.0))

    topic = cfg.get("task", {}).get("description", "")

    agent_zero = AgentZero(
        provider=az_provider, model=az_model, max_agents=max_agents, dry_run=dry_run,
        temperature=az_temperature,
    )

    # Shared, mutable dict — every Agent below is constructed with this same
    # `cfg` reference, so Agent 0's per-agent instructions (added later via
    # `cfg["agent_prompt_overrides"][name] = ...`) reach every Agent's cached
    # `_overrides` without rebuilding any Agent object.
    cfg.setdefault("agent_prompt_overrides", {})

    # ── One-time initialization call ──────────────────────────────────────
    # Agent 0 designs everything about how the debate runs except the hard
    # bounds above: roster, shared instructions, quality dimensions/weights,
    # coalition threshold, and φ1/φ2/ECU-info-condition visibility. A
    # human-provided value in cfg (e.g. from a hand-authored batch config)
    # is respected as an override; anything left unset is Agent 0's call,
    # not a hardcoded default.
    init_result = agent_zero.initialize(topic)
    initial_roster = init_result["agents"][:max_agents]

    instructions_cfg = cfg.setdefault("instructions", {})
    if not instructions_cfg.get("base_instructions", "").strip():
        instructions_cfg["base_instructions"] = init_result["base_instructions"]
    if not instructions_cfg.get("guideline_notes", "").strip():
        instructions_cfg["guideline_notes"] = init_result["guideline_notes"]

    ecu_cfg = cfg.setdefault("ecu", {})
    ecu_cfg["enabled"] = True  # foundational to this mode — not a human-configurable toggle
    ecu_cfg.setdefault("include_self_assessment", False)
    if not ecu_cfg.get("dimensions"):
        ecu_cfg["dimensions"] = init_result["ecu_dimensions"]
    if "coalition_threshold" not in ecu_cfg:
        ecu_cfg["coalition_threshold"] = init_result["coalition_threshold"]
    if "info_condition" not in ecu_cfg:
        ecu_cfg["info_condition"] = init_result["info_condition"]

    protocol_cfg = cfg.setdefault("protocol", {})
    if "visibility_mode" not in protocol_cfg:
        protocol_cfg["visibility_mode"] = init_result["visibility_mode"]
    if "review_depth" not in protocol_cfg:
        protocol_cfg["review_depth"] = init_result["review_depth"]
    review_depth = protocol_cfg["review_depth"]

    agents_by_name: dict[str, Agent] = {
        spec["name"]: Agent(spec, cfg) for spec in initial_roster
    }
    total_spawns = len(agents_by_name)
    agent_names = list(agents_by_name.keys())

    peer_reviewer = build_peer_reviewer(cfg, review_depth)
    ledger, coalition, _ = build_ecu_components(cfg, agent_names, review_depth)
    # Orchestrator is deliberately discarded (`_`) — ECU weights stay fixed
    # at their initial config values in Agent 0 mode.

    hub = build_hub("item_0", {"topic": topic}, cfg, agent_names, ledger)

    initial_roster_names = list(agent_names)

    # ── Design summary — Agent 0's full initialization decision, exposed to
    # the caller (UI/CLI) before the round loop starts, and included in the
    # final result for post-hoc inspection. ──────────────────────────────
    design_summary = {
        "reasoning": init_result["reasoning"],
        "agents": [
            {"name": n, "role": agents_by_name[n].role,
             "provider": agents_by_name[n].provider, "model": agents_by_name[n].model}
            for n in initial_roster_names
        ],
        "base_instructions": instructions_cfg["base_instructions"],
        "guideline_notes": instructions_cfg["guideline_notes"],
        "ecu_dimensions": ecu_cfg["dimensions"],
        "coalition_threshold": ecu_cfg["coalition_threshold"],
        "visibility_mode": protocol_cfg["visibility_mode"],
        "review_depth": protocol_cfg["review_depth"],
        "info_condition": ecu_cfg["info_condition"],
    }
    if on_init:
        on_init(design_summary)
    own_history: list[dict] = []  # Agent 0's own {"cycle", "reasoning"} log
    round_summaries: list[dict] = []  # compact, human-readable per-round record
    final_brief: str | None = None
    ended_reason = "max_rounds_reached"
    context = ""  # last round's Agent 0 context — reused for the wrap-up brief call if needed

    for cycle in range(max_rounds):
        if not hub.agent_names:
            ended_reason = "roster_empty"
            break

        # ── Phase 1 + 2 — reuse existing contribution/peer-review/ECU logic ──
        round_contributions = _run_debate_round(hub, cycle, agents_by_name, dry_run)
        if peer_reviewer:
            _run_peer_review_round(
                hub, cycle, round_contributions, agents_by_name,
                peer_reviewer, coalition, ledger, dry_run,
            )

        # ── Agent 0 call ──────────────────────────────────────────────────
        roster_specs = [
            {
                "name": n,
                "role": agents_by_name[n].role,
                "provider": agents_by_name[n].provider,
                "model": agents_by_name[n].model,
            }
            for n in hub.agent_names if n in agents_by_name
        ]
        coalition_state = coalition.history[-1]["coalition"] if coalition and coalition.history else []
        agent_notes = _build_agent_notes(hub, ledger)
        context = agent_zero.build_context(
            topic=topic,
            cycle=cycle,
            roster=roster_specs,
            round_contributions=round_contributions,
            ecu_balances=hub.ecu_balances,
            sw_history=ledger.social_welfare_history if ledger else [],
            coalition_state=coalition_state,
            own_history=own_history,
            agent_notes=agent_notes,
            total_spawns=total_spawns,
            max_total_spawns=max_total_spawns,
        )
        decision = agent_zero.decide(context, hub.agent_names)
        own_history.append({"cycle": cycle, "reasoning": decision["reasoning"]})

        # Log Agent 0's own call alongside contribution/peer_review entries so
        # it's inspectable in prompt_log rather than invisible. Logging the
        # raw pre-validation response (not just the validated decision) makes
        # it possible to tell whether the model's `reasoning` drifted from
        # what it actually put in the structured fields.
        raw_response = decision.pop("_raw_response", None)
        hub.log_prompt(
            cycle, "Agent0", "agent_zero_decision",
            prompt=context,
            response=json.dumps({"raw": raw_response, "validated": decision}, ensure_ascii=False),
        )

        # ── Execute decision, subject to hard bounds ─────────────────────
        actually_removed: list[str] = []
        actually_added: list[str] = []

        for name in decision["remove_agents"]:
            if len(hub.agent_names) <= 1:
                break
            hub.remove_agent(name, cycle=cycle)
            actually_removed.append(name)

        for spec in decision["add_agents"]:
            if len(hub.agent_names) >= max_agents or total_spawns >= max_total_spawns:
                break
            name = spec["name"]
            if name in agents_by_name:
                continue
            agents_by_name[name] = Agent(spec, cfg)
            hub.add_agent(name, cycle=cycle)
            if ledger:
                ledger.add_agent(name)
            total_spawns += 1
            actually_added.append(name)

        # Dynamic per-agent instructions — mutate the shared dict in place so
        # every Agent's cached `_overrides` reference sees the update without
        # rebuilding the Agent. Only agents still on the roster are targeted
        # (already filtered in _validate_decision).
        for name, instruction in decision["agent_instructions"].items():
            cfg["agent_prompt_overrides"][name] = instruction

        # Per-agent mean dimension scores for this round (includes "consensus",
        # which drives coalition membership) plus ECU earned this round —
        # without this, the coalition list has no visible justification.
        cycle_records = [r for r in (ledger.history if ledger else []) if r.cycle == cycle]
        dimension_scores = {
            rec.agent_name: {k: round(v, 3) for k, v in rec.aggregated_scores.items()}
            for rec in cycle_records
        }
        ecu_earned_this_round = {rec.agent_name: round(rec.ecu_earned, 3) for rec in cycle_records}

        # ── Compact, human-readable per-round record ─────────────────────
        # Contributions are stored in full here (this is the primary readable
        # record) — the truncation used to live only in "debug", but that
        # buried the actual argument text a reader most wants to see.
        round_summaries.append({
            "cycle": cycle,
            "roster": roster_specs,
            "contributions": dict(round_contributions),
            "agent_trajectory_notes": agent_notes,
            "dimension_scores": dimension_scores,
            "ecu_earned_this_round": ecu_earned_this_round,
            "ecu_balances": dict(hub.ecu_balances),
            "social_welfare": (
                ledger.social_welfare_history[-1]["social_welfare"]
                if ledger and ledger.social_welfare_history else None
            ),
            "coalition": coalition_state,
            "agent_zero_reasoning": decision["reasoning"],
            "agents_added": actually_added,
            "agents_removed": actually_removed,
            "agent_instructions_issued": decision["agent_instructions"],
            "end_debate": decision["end_debate"],
        })

        if on_round:
            on_round(round_summaries[-1])

        if decision["end_debate"]:
            final_brief = decision["final_brief"]
            ended_reason = "agent_zero_ended"
            break

        if total_spawns >= max_total_spawns and len(hub.agent_names) >= max_agents:
            # Both hard caps reached — no more roster edits are possible even
            # if Agent 0 keeps asking; end gracefully rather than spin.
            pass  # not a stop condition by itself — only max_rounds/end_debate stop the loop

    # A brief is produced no matter how the loop stopped — if Agent 0 never
    # issued an end_debate decision (e.g. max_rounds was hit first), ask it
    # for a wrap-up brief explicitly. `ended_reason` still reflects why the
    # loop actually stopped ("max_rounds_reached" vs "agent_zero_ended").
    if not final_brief or not final_brief.strip():
        last = round_summaries[-1] if round_summaries else None
        if last:
            coalition_str = ", ".join(last["coalition"]) if len(last["coalition"]) >= 2 else "none formed"
            fallback_summary = (
                f"[Auto-generated summary. Agent 0 did not supply a final brief.] "
                f"The debate ran for {last['cycle'] + 1} round(s) and stopped "
                f"({ended_reason.replace('_', ' ')}). Final roster: "
                f"{', '.join(a['name'] for a in last['roster'])}. "
                f"Final social welfare: {last['social_welfare']}. "
                f"Coalition: {coalition_str}. "
                f"Agent 0's last reasoning: {last['agent_zero_reasoning']}"
            )
        else:
            fallback_summary = f"[Auto-generated summary] Debate ended ({ended_reason}) with no completed rounds."
        final_brief = agent_zero.write_final_brief(context, fallback_summary)

    # Result shape is deliberately different from collect_result()/manual mode:
    # "rounds" is the primary, chronological, human-readable record — read this
    # first. Everything else (raw contribution/peer-review objects, the full
    # coalition agreement matrix per round, every LLM prompt+response) is
    # grouped under "debug" for when you need to trace a specific call.
    return {
        "mode": "agent_zero",
        "item_id": hub.item_id,
        "num_turns": hub.num_submissions,
        "ended_reason": ended_reason,
        "final_brief": final_brief,
        "agent_zero_config": {
            "provider": az_provider, "model": az_model, "temperature": az_temperature,
            "max_rounds": max_rounds, "max_agents": max_agents,
            "max_total_spawns": max_total_spawns,
        },
        "initial_roster": initial_roster_names,
        "agent_zero_design": design_summary,
        "rounds": round_summaries,
        "final_ecu_balances": hub.ecu_balances,
        "final_ecu_weights": ledger.ecu_weights if ledger else {},
        "debug": {
            # Untruncated contribution text only — no raw_response/timestamp/
            # changed-flags, since the exact same raw text already lives in
            # prompt_log below. Storing it twice was the single biggest
            # source of bloat in these logs.
            "full_contributions": [
                {"cycle": o.cycle, "agent_name": o.agent_name, "contribution": o.contribution}
                for o in hub.log
            ],
            # Parsed scores/justifications only — raw_response dropped for
            # the same reason (duplicated in prompt_log).
            "peer_review_scores": [
                {"cycle": p.cycle, "reviewer": p.reviewer_name,
                 "scores": p.scores, "justifications": p.justifications}
                for p in hub.peer_review_log
            ],
            "coalition_history": coalition.to_dict() if coalition else {},
            "ecu_ledger": ledger.to_dict() if ledger else {},
            "prompt_log": hub.prompt_log,  # the one place with full raw prompts + responses
        },
    }
