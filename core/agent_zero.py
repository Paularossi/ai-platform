"""
core/agent_zero.py

Agent 0 — the autonomous orchestrator for "Agent 0 mode".

Unlike manual mode (fixed roster, human-configured protocol), Agent 0 mode
only requires a human-provided deliberation topic. A single super-agent:

  1. Reads the topic and proposes the starting roster (one-time call).
  2. After every debate round (post peer-review, post ECU/SW computation),
     is called once with the round's state and returns a structured decision:
     agents to add, agents to remove, whether to end the debate, and — if
     ending — the final policy brief.

All decisions are produced via the provider's structured-output mode
(`complete_structured`) and validated before being applied. A decision that
fails validation degrades to a safe no-op rather than raising, so a single
bad model response can never crash a run or destructively resize the roster.

Hard safety bounds (max rounds, max roster size, max total spawns) are
enforced by the calling loop (core/protocols/agent_zero_loop.py), not here —
Agent 0's own judgement is never the last line of defence against runaway
cost or an infinite loop.
"""

from __future__ import annotations

from typing import Any

from core.providers import get_provider

# ---------------------------------------------------------------------------
# Standing pre-prompt — Agent 0's fixed mandate
# ---------------------------------------------------------------------------

AGENT_ZERO_SYSTEM_PROMPT = """You are Agent 0, the autonomous orchestrator of a multi-agent deliberation \
platform. A human has provided only a topic; you are responsible for everything else: \
choosing which agents deliberate, adjusting the roster as the debate evolves, deciding \
when to give an agent a new instruction, and deciding when the debate has run its course.

Your available actions, each round:
1. INITIAL ROSTER (one-time, before round 1) — propose the set of agents that will \
deliberate on the topic. Give each a name, a role description (their mandate or \
perspective, not a personality), and an LLM provider + model.
2. ADD AGENTS — introduce a new agent to the roster starting the next round.
3. REMOVE AGENTS — drop an agent from the roster; they stop contributing and being \
scored from the next round, but their prior contributions remain part of the record.
4. STEER AGENTS — give any agent a short, specific instruction that is added to their \
prompt from the next round onward.
5. END THE DEBATE — stop the debate and write a final policy brief that represents \
what the deliberation actually produced — the consensus reached, or the substantive \
disagreement, if no consensus formed.

You decide when and whether to use each of these based on what you observe. Nothing about \
the debate's length, roster size, or how often you intervene is prescribed in advance — \
form your own judgement from the signals available to you each round: the round number, \
current roster (with roles), this round's contributions, ECU balances, the social welfare \
value and its trajectory so far, the current coalition state, your own reasoning from \
previous rounds, and a short per-agent trajectory note (how similar their contribution is \
to their own previous round, and their ECU trend).

For every agent (initial roster or newly added), you must choose exactly one of these \
two provider/model pairs: OpenAI/gpt-4o or Anthropic/claude-sonnet-4-6. Do not propose \
any other provider or model.

The roster has a hard capacity each round, shown to you as "roster: X/Y". If the roster \
is already full, an addition will only take effect if you also remove an agent in the \
same decision — there is no room to add without freeing a slot first.

Everything you decide must be reflected in the structured fields, not just described in \
`reasoning`. If your reasoning discusses adding, removing, or instructing an agent, or \
ending the debate, the corresponding field (add_agents, remove_agents, agent_instructions, \
end_debate) must carry that same action — reasoning is a record of your thinking, not a \
substitute for acting on it.

Keep your `reasoning` field short — 2-4 sentences capturing the signal(s) behind this \
round's decision. If you decide to end the debate, reserve most of your response for a \
thorough `final_brief` (roughly 200-500 words) — a short or missing brief means the \
decision to end will be rejected and the debate will continue regardless of your \
reasoning, so do not shortchange it."""


DEFAULT_PROVIDER = "OpenAI"
DEFAULT_MODEL = "gpt-4o"

# The only provider/model pairs Agent 0 is allowed to assign to an agent.
# Keeps the roster restricted to models we've validated for this mode and
# avoids provider-name casing mismatches (get_provider() is case-sensitive).
ALLOWED_PROVIDER_MODELS: dict[str, str] = {
    "OpenAI": "gpt-4o",
    "Anthropic": "claude-sonnet-4-6",
}


# ---------------------------------------------------------------------------
# JSON schemas for structured output
# ---------------------------------------------------------------------------

_AGENT_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "role": {"type": "string"},
        "provider": {"type": "string", "enum": list(ALLOWED_PROVIDER_MODELS.keys())},
        "model": {"type": "string", "enum": list(ALLOWED_PROVIDER_MODELS.values())},
    },
    "required": ["name", "role", "provider", "model"],
    "additionalProperties": False,
}

INIT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "agents": {"type": "array", "items": _AGENT_SPEC_SCHEMA},
    },
    "required": ["reasoning", "agents"],
    "additionalProperties": False,
}

_AGENT_INSTRUCTION_SCHEMA = {
    "type": "object",
    "properties": {
        "agent_name": {"type": "string"},
        "instruction": {"type": "string"},
    },
    "required": ["agent_name", "instruction"],
    "additionalProperties": False,
}

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "final_brief": {"type": "string"},
    },
    "required": ["final_brief"],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "add_agents": {"type": "array", "items": _AGENT_SPEC_SCHEMA},
        "remove_agents": {"type": "array", "items": {"type": "string"}},
        "agent_instructions": {"type": "array", "items": _AGENT_INSTRUCTION_SCHEMA},
        "end_debate": {"type": "boolean"},
        "final_brief": {"type": "string"},
    },
    "required": ["reasoning", "add_agents", "remove_agents", "agent_instructions",
                 "end_debate", "final_brief"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Safe defaults
# ---------------------------------------------------------------------------

def _default_roster() -> list[dict]:
    """Fallback starting roster used if Agent 0's init call fails validation."""
    return [
        {"name": "Advocate", "role": "Argue in favour of the strongest available position.",
         "provider": DEFAULT_PROVIDER, "model": DEFAULT_MODEL},
        {"name": "Skeptic", "role": "Stress-test claims and surface risks or downsides.",
         "provider": DEFAULT_PROVIDER, "model": DEFAULT_MODEL},
        {"name": "Mediator", "role": "Seek common ground and synthesise competing views.",
         "provider": DEFAULT_PROVIDER, "model": DEFAULT_MODEL},
    ]


def _noop_decision(reason: str = "validation failed") -> dict:
    return {
        "reasoning": f"[safe default: {reason}]",
        "add_agents": [],
        "remove_agents": [],
        "agent_instructions": {},
        "end_debate": False,
        "final_brief": "",
    }


def _normalize_provider_model(raw_provider: Any, raw_model: Any) -> tuple[str, str]:
    """
    Snap whatever the model returned onto one of ALLOWED_PROVIDER_MODELS,
    case-insensitively. Falls back to the default pair if the provider name
    doesn't match anything we recognise — get_provider() is case-sensitive
    and only accepts the exact registry names, so anything else would crash
    the run rather than degrading gracefully.
    """
    provider_str = str(raw_provider or "").strip().lower()
    for canonical in ALLOWED_PROVIDER_MODELS:
        if canonical.lower() == provider_str:
            return canonical, ALLOWED_PROVIDER_MODELS[canonical]
    return DEFAULT_PROVIDER, DEFAULT_MODEL


def _validate_agent_spec(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name", "")).strip()
    role = str(raw.get("role", "")).strip()
    if not name or not role:
        return None
    provider, model = _normalize_provider_model(raw.get("provider"), raw.get("model"))
    return {"name": name, "role": role, "provider": provider, "model": model,
            "custom_role": role, "temperature": 0.0}


def _validate_roster(raw: Any, max_agents: int) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        spec = _validate_agent_spec(item)
        if spec and spec["name"] not in seen:
            out.append(spec)
            seen.add(spec["name"])
        if len(out) >= max_agents:
            break
    return out


def _validate_decision(raw: Any, current_roster: list[str], max_agents: int) -> dict:
    if not isinstance(raw, dict):
        return _noop_decision("non-dict response")

    reasoning = str(raw.get("reasoning", ""))

    remove_agents = [
        n for n in raw.get("remove_agents", [])
        if isinstance(n, str) and n in current_roster
    ]
    # Never allow removing the entire roster.
    if len(remove_agents) >= len(current_roster):
        remove_agents = remove_agents[:max(0, len(current_roster) - 1)]

    add_raw = raw.get("add_agents", [])
    add_agents: list[dict] = []
    if isinstance(add_raw, list):
        existing = set(current_roster)
        for item in add_raw:
            spec = _validate_agent_spec(item)
            if spec and spec["name"] not in existing:
                add_agents.append(spec)
                existing.add(spec["name"])
            if len(add_agents) >= max_agents:
                break

    # Instructions target agents still on the roster after removals above,
    # and never a self-instruction loop for an agent being removed this round.
    instr_raw = raw.get("agent_instructions", [])
    agent_instructions: dict[str, str] = {}
    if isinstance(instr_raw, list):
        valid_targets = set(current_roster) - set(remove_agents)
        for item in instr_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("agent_name", "")).strip()
            instruction = str(item.get("instruction", "")).strip()
            if name in valid_targets and instruction:
                agent_instructions[name] = instruction

    end_debate = bool(raw.get("end_debate", False))
    final_brief = str(raw.get("final_brief", "") or "")
    if end_debate and not final_brief.strip():
        # An "end" decision with no brief is not actionable — treat as no-op
        # on the ending part but keep any valid roster edits. This is loud on
        # purpose: it usually means the response ran out of max_tokens before
        # writing the brief, which would otherwise silently loop forever.
        print("[AgentZero] end_debate=true but final_brief is empty — "
              "rejecting the end decision and continuing (likely truncated response).")
        end_debate = False

    return {
        "reasoning": reasoning,
        "add_agents": add_agents,
        "remove_agents": remove_agents,
        "agent_instructions": agent_instructions,
        "end_debate": end_debate,
        "final_brief": final_brief,
    }


# ---------------------------------------------------------------------------
# AgentZero
# ---------------------------------------------------------------------------

class AgentZero:
    """
    Parameters
    ----------
    provider, model : str
        LLM used for Agent 0's own decisions (independent of the debating agents).
    max_agents : int
        Roster size cap — enforced here defensively, and again by the loop.
    dry_run : bool
        Skip the API call and return deterministic placeholder decisions.
    """

    def __init__(
        self,
        provider: str = "Anthropic",
        model: str = "claude-opus-4-8",
        max_agents: int = 6,
        dry_run: bool = False,
    ):
        self.provider = provider
        self.model = model
        self.max_agents = max_agents
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    def initialize(self, topic: str) -> list[dict]:
        """One-time call: propose the starting roster from the topic alone."""
        if self.dry_run:
            return _default_roster()

        prompt = (
            f"Deliberation topic: {topic}\n\n"
            f"Propose the starting roster (at most {self.max_agents} agents). "
            "Return your reasoning and the agent list."
        )
        try:
            result = get_provider(self.provider).complete_structured(
                model=self.model,
                system_prompt=AGENT_ZERO_SYSTEM_PROMPT,
                user_message=prompt,
                schema=INIT_SCHEMA,
                max_tokens=1200,
            )
            roster = _validate_roster(result.get("agents"), self.max_agents)
        except Exception as exc:
            print(f"[AgentZero] initialize() failed: {exc} — falling back to default roster")
            roster = []

        return roster or _default_roster()

    # ------------------------------------------------------------------
    def build_context(
        self,
        topic: str,
        cycle: int,
        roster: list[dict],
        round_contributions: dict[str, Any],
        ecu_balances: dict[str, float],
        sw_history: list[dict],
        coalition_state: list[str],
        own_history: list[dict] | None = None,
        agent_notes: dict[str, str] | None = None,
        max_own_history: int = 5,
        total_spawns: int | None = None,
        max_total_spawns: int | None = None,
    ) -> str:
        """
        Assemble the per-round context packet as a readable text block.

        own_history : list[dict] | None
            Agent 0's own {"cycle", "reasoning"} entries from previous rounds
            (most recent `max_own_history` only, to bound token growth).
        agent_notes : dict[str, str] | None
            One compact line per agent summarising round-over-round change
            (similarity to their own previous contribution, ECU trend). This
            is Agent 0-specific bookkeeping, independent of φ1/φ2 — the
            debating agents' own visibility settings are unaffected by it.
        total_spawns, max_total_spawns : int | None
            Cumulative agents ever created vs. the hard cap — shown so Agent 0
            knows whether an addition is even possible before proposing one.
        """
        lines: list[str] = []
        lines.append(f"Topic: {topic}")
        lines.append(f"Round just completed: {cycle + 1}")
        lines.append("")

        cap_str = f"{len(roster)}/{self.max_agents}"
        lines.append(f"Current roster ({cap_str}):")
        for a in roster:
            lines.append(f"  - {a['name']} ({a.get('provider', '?')}/{a.get('model', '?')}): "
                          f"{a.get('role', a.get('custom_role', ''))}")
        if len(roster) >= self.max_agents:
            lines.append(f"  Roster is at capacity ({cap_str}) — adding requires removing one first.")
        if total_spawns is not None and max_total_spawns is not None:
            lines.append(f"Total agents spawned so far: {total_spawns}/{max_total_spawns}"
                          + (" — no further additions possible." if total_spawns >= max_total_spawns else ""))
        lines.append("")

        lines.append("This round's contributions:")
        for name, contrib in round_contributions.items():
            lines.append(f"  [{name}]: {contrib}")
        lines.append("")

        if agent_notes:
            lines.append("Per-agent trajectory notes:")
            for name, note in agent_notes.items():
                lines.append(f"  [{name}]: {note}")
            lines.append("")

        if ecu_balances:
            bal_str = ", ".join(f"{k}={v:.3f}" for k, v in ecu_balances.items())
            lines.append(f"ECU balances (cumulative): {bal_str}")

        if sw_history:
            sw_str = ", ".join(f"r{e['cycle']+1}={e['social_welfare']:.4f}" for e in sw_history)
            lines.append(f"Social welfare trajectory: {sw_str}")
            lines.append(f"Current social welfare: {sw_history[-1]['social_welfare']:.4f}")
        lines.append("")

        if coalition_state and len(coalition_state) >= 2:
            lines.append(f"Current coalition: {', '.join(coalition_state)}")
        else:
            lines.append("Current coalition: none")
        lines.append("")

        if own_history:
            lines.append("Your own reasoning from previous rounds:")
            for entry in own_history[-max_own_history:]:
                lines.append(f"  Round {entry['cycle'] + 1}: {entry['reasoning']}")
            lines.append("")

        lines.append(
            "Decide: any agents to add, any agents to remove, whether to end the debate now, "
            "and — only if ending — the final policy brief."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def decide(self, context: str, current_roster: list[str]) -> dict:
        """Per-round call: returns a validated decision, or a safe no-op."""
        if self.dry_run:
            return _noop_decision("dry-run")

        try:
            result = get_provider(self.provider).complete_structured(
                model=self.model,
                system_prompt=AGENT_ZERO_SYSTEM_PROMPT,
                user_message=context,
                schema=DECISION_SCHEMA,
                # Generous headroom: reasoning + up to a few agent_instructions
                # + a 200-500 word final_brief can easily exceed 1200 tokens.
                # Running out mid-response silently drops the brief, which
                # forces end_debate back to False regardless of intent.
                max_tokens=4096,
            )
            validated = _validate_decision(result, current_roster, self.max_agents)
            # Keep the pre-validation response for debugging drift between
            # `reasoning` and the structured fields (e.g. the model discusses
            # adding an agent but the add_agents array stays empty). Not part
            # of the public decision shape used by the loop's own logic.
            validated["_raw_response"] = result
            return validated
        except Exception as exc:
            print(f"[AgentZero] decide() failed: {exc} — defaulting to no-op")
            return _noop_decision(str(exc))

    # ------------------------------------------------------------------
    def write_final_brief(self, context: str, fallback_summary: str) -> str:
        """
        Produce a final policy brief regardless of how the debate ended —
        called by the loop whenever it stops without Agent 0 having already
        supplied one (e.g. max_rounds reached without an end_debate decision).

        fallback_summary : str
            A deterministic, always-available summary (roster, final SW,
            coalition, last reasoning) used verbatim if the model call fails
            or returns an empty brief — this method can never return "".
        """
        if self.dry_run:
            return fallback_summary

        try:
            result = get_provider(self.provider).complete_structured(
                model=self.model,
                system_prompt=AGENT_ZERO_SYSTEM_PROMPT,
                user_message=(
                    context
                    + "\n\nThe debate is ending now (round limit reached without an earlier "
                      "end decision). Write the final policy brief only, summarising the "
                      "consensus reached (or the substantive disagreement, if no consensus "
                      "formed), grounded in the actual contributions made."
                ),
                schema=BRIEF_SCHEMA,
                max_tokens=2000,
            )
            brief = str(result.get("final_brief", "") or "").strip()
            return brief or fallback_summary
        except Exception as exc:
            print(f"[AgentZero] write_final_brief() failed: {exc} — using deterministic fallback summary")
            return fallback_summary
