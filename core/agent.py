"""
core/agent.py

Stateless agent wrapper.

The agent receives a ContextPacket from the hub, builds a prompt,
calls the appropriate LLM provider, parses the response, and returns
an AgentOutput.
"""

from __future__ import annotations

import re
from typing import Any

from core.hub import ContextPacket
from core.state import AgentOutput
from core.providers import get_provider
# Prompt builders
# ---------------------------------------------------------------------------

def _build_system_prompt(
    agent_name: str,
    agent_role: str,
    base_instructions: str,
    guideline_notes: str,
    agent_overrides: dict[str, str],
    ecu_info_condition: str = "opaque",
    ecu_dimensions: list[dict] | None = None,
) -> str:
    parts: list[str] = []

    role_str = agent_role.rstrip(".")
    parts.append(f"You are {agent_name}. Your role: {role_str}.")
    parts.append("")

    override = agent_overrides.get(agent_name, "").strip()
    if override:
        parts.append(override)
        parts.append("")

    if base_instructions.strip():
        parts.append(base_instructions.strip())
        parts.append("")

    if guideline_notes.strip():
        parts.append("--- Definitions and guidelines ---")
        parts.append(guideline_notes.strip())
        parts.append("")

    if ecu_info_condition in ("semi-transparent", "transparent") and ecu_dimensions:
        dim_labels = ", ".join(d.get("label", d["name"]) for d in ecu_dimensions)
        parts.append("--- Evaluation and incentives ---")
        parts.append(
            f"After each round your contribution is peer-reviewed by the other participants "
            f"on these quality dimensions: {dim_labels}. "
            f"Each dimension is scored 0–1 and you earn ECU (Experimental Currency Units) "
            f"based on those scores."
        )
        if ecu_info_condition == "transparent":
            parts.append(
                "You can see all participants' scores, ECU balances, and the current "
                "dimension weights in each round."
            )
        else:
            parts.append(
                "You can see your own ECU balance and approximate dimension weights each round."
            )
        parts.append("")

    # Output format is entirely defined by the user's base_instructions since this is a free-form deliberation.

    return "\n".join(parts)


def _build_user_message(packet: ContextPacket) -> str:
    parts: list[str] = []

    # ── Item data (from dataset) ──────────────────────────────────────────────
    for key, value in packet.item_data.items():
        parts.append(f"{key}: {value}")
    parts.append("")

    # ── Previous round context (from hub, cycle > 0) ──────────────────────────
    if packet.visible_history:
        # Group history by agent, preserving chronological order
        by_agent: dict[str, list] = {}
        for out in packet.visible_history:
            by_agent.setdefault(out.agent_name, []).append(out)
        # Sort each agent's entries by cycle
        for entries in by_agent.values():
            entries.sort(key=lambda o: o.cycle)

        is_full_history = any(len(entries) > 1 for entries in by_agent.values())

        parts.append("=== Contribution history ===" if is_full_history else "=== Previous round ===")
        parts.append("")

        parts.append("Contributions:")
        for agent_name, entries in by_agent.items():
            own = agent_name == packet.agent_name
            label = f"{agent_name} (your contribution{'s' if is_full_history else ''})" if own else agent_name
            if len(entries) == 1:
                contrib = str(entries[0].contribution) if entries[0].contribution else "(none)"
                parts.append(f"  [{label}]: {contrib}")
            else:
                parts.append(f"  [{label}]:")
                for out in entries:
                    contrib = str(out.contribution) if out.contribution else "(none)"
                    parts.append(f"    Round {out.cycle + 1}: {contrib}")
        parts.append("")

        # Show peer review scores from the most recent round only
        most_recent = [entries[-1] for entries in by_agent.values()]
        scores_shown = False
        for out in most_recent:
            if out.ecu_scores:
                if not scores_shown:
                    parts.append("Peer review scores (average received, last round):")
                    scores_shown = True
                score_str = ", ".join(f"{d}: {s:.2f}" for d, s in out.ecu_scores.items())
                ecu_str = f"  [{out.ecu_earned:.2f} ecus]" if out.ecu_earned is not None else ""
                parts.append(f"  {out.agent_name}: {score_str}{ecu_str}")
        if scores_shown:
            parts.append("")

        # Show ECU balance info based on information condition
        # packet.ecu_balances is populated for both T and S conditions
        if packet.ecu_balances:
            own_balance = packet.ecu_balances.get(packet.agent_name)
            if own_balance is not None and len(packet.ecu_balances) == 1:
                # Semi-transparent: only own balance was passed
                parts.append(f"Your current ECU balance: {own_balance:.2f}")
            else:
                # Transparent: all balances + weight vector
                bal_str = ", ".join(f"{k}: {v:.2f}" for k, v in packet.ecu_balances.items())
                parts.append(f"Current ECU balances: {bal_str}")
                if packet.ecu_weights:
                    w_str = ", ".join(f"{k}={v:.2f}" for k, v in packet.ecu_weights.items())
                    parts.append(f"Current quality weights: {w_str}")
            parts.append("")

        parts.append("=== Your turn ===")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------


def unescape_literal_whitespace(text: str) -> str:
    """
    Un-escape "\\n" so the text splits into real lines/paragraphs, both for parsing and for 
    display (e.g. live streaming in the UI, where otherwise "\\n" shows up as literal text on the page).

    Only touches the escape sequences a model would plausibly emit this way;
    doesn't touch other backslashes, so a stray "C:\\notes" is left alone.
    """
    if "\\n" not in text and "\\r" not in text and "\\t" not in text:
        return text
    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .replace("\\t", "\t")
    )


# Matches a "LABEL = " style field line — e.g. "REASONING = ..." or "RESPONSE = ..." 
# A plainsingle newline before such a line (as opposed to a blank line) doesn't
# otherwise start a new paragraph, so without this the field boundary gets
# silently merged into the same paragraph as whatever came before it.
_FIELD_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,30}\s*=")


def _join_paragraphs(lines: list[str]) -> str:
    """
    Collapse a list of lines back into text: consecutive lines within a paragraph are joined 
    with a space, but paragraph breaks are kept as a blank line.
    """
    paragraphs: list[list[str]] = [[]]
    for line in lines:
        if line == "":
            if paragraphs[-1]:
                paragraphs.append([])
        else:
            paragraphs[-1].append(line)
    paragraphs = [p for p in paragraphs if p]
    return "\n\n".join(" ".join(p) for p in paragraphs).strip()


def _parse_deliberation(raw_text: str) -> str:
    """
    Parse a free-text deliberation response into the agent's contribution.

    Also handles the case where a model (e.g. Gemini in JSON mode) wraps
    the contribution in a JSON envelope like {"response": "..."}.
    """
    # Unwrap JSON envelope if present — e.g. {"response": "actual text"}
    stripped_raw = raw_text.strip()
    if stripped_raw.startswith("{") and '"response"' in stripped_raw:
        try:
            import json as _json
            obj = _json.loads(stripped_raw)
            if isinstance(obj, dict) and "response" in obj:
                raw_text = str(obj["response"])
        except Exception:
            pass  # not valid JSON — continue with raw_text as-is

    raw_text = unescape_literal_whitespace(raw_text)

    contribution_lines: list[str] = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            # A blank line marks a paragraph break so record it as a "" marker
            if contribution_lines and contribution_lines[-1] != "":
                contribution_lines.append("")
            continue

        # Skip Setext-style underline markers (e.g. "===" or "---" directly under a title line)
        if re.fullmatch(r"[=\-]{3,}", stripped):
            continue
        # Strip leading Markdown heading markers (#, ##, ...)
        stripped = re.sub(r"^#{1,6}\s*", "", stripped)
        if _FIELD_LABEL_RE.match(stripped) and contribution_lines and contribution_lines[-1] != "":
            contribution_lines.append("")
        contribution_lines.append(stripped)

    return _join_paragraphs(contribution_lines) or raw_text.strip()


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class Agent:
    """
    Stateless LLM agent.

    Parameters
    ----------
    config : dict
        Agent config from the UI: name, provider, model, role.
    experiment_config : dict
        Full experiment config (instructions, overrides).
    """

    def __init__(self, config: dict[str, Any], experiment_config: dict[str, Any]):
        self.name: str = config.get("name", "Agent")
        self.provider: str = config.get("provider", "OpenAI")
        self.model: str = config.get("model", "gpt-4o")
        self.temperature: float = float(config.get("temperature", 0.0))
        # A human-controlled agent uses send(human_input=...) to send its contribution. 
        # Protocols and peer review check this to skip API calls for it.
        self.is_human: bool = self.provider == "Human"

        raw_role = config.get("role", "Participant")
        custom_role_text = config.get("custom_role", "").strip()
        self.role: str = custom_role_text if raw_role == "Custom" and custom_role_text else raw_role

        self._instructions = experiment_config.get("instructions", {})
        self._overrides = experiment_config.get("agent_prompt_overrides", {})
        self._ecu_info_condition: str = experiment_config.get("ecu", {}).get("info_condition", "opaque")
        self._ecu_dimensions: list[dict] = experiment_config.get("ecu", {}).get("dimensions", [])

        # Cached from the last build_prompt() call — used by protocols for prompt logging
        self._last_system_prompt: str = ""
        self._last_user_message: str = ""

    def build_prompt(self, packet: ContextPacket) -> tuple[str, str]:
        """
        Build the (system_prompt, user_message) pair for this agent's next
        turn, without calling the LLM. Exposed separately from send() so a
        caller (e.g. a step-through UI) can show the exact text about to be 
        sent, let the user edit it, and only then dispatch it.
        """
        system_prompt = _build_system_prompt(
            agent_name=self.name,
            agent_role=self.role,
            base_instructions=self._instructions.get("base_instructions", ""),
            guideline_notes=self._instructions.get("guideline_notes", ""),
            agent_overrides=self._overrides,
            ecu_info_condition=self._ecu_info_condition,
            ecu_dimensions=self._ecu_dimensions,
        )
        user_message = _build_user_message(packet)
        return system_prompt, user_message

    def send(
        self,
        packet: ContextPacket,
        system_prompt: str,
        user_message: str,
        dry_run: bool = False,
        human_input: str | None = None,
    ) -> AgentOutput:
        """
        Dispatch an already-built (system_prompt, user_message) pair and
        return the resulting AgentOutput.

        Parameters
        ----------
        human_input : str | None
            If set, skip the LLM call entirely and use this text as the
            contribution instead (Control / human-moderated turns). The
            returned AgentOutput has the same shape as an LLM-produced one,
            so it is indistinguishable to other agents and to the hub.
        dry_run : bool
            If True (and human_input is None), skip the API call and return
            a placeholder contribution. Used by the test suite.
        """
        self._last_system_prompt = system_prompt
        self._last_user_message = user_message

        if human_input is not None:
            return AgentOutput(
                agent_name=self.name,
                cycle=packet.cycle,
                contribution=human_input,
                raw_response=None,
            )

        if dry_run:
            dummy = "[dry-run: no contribution]"
            return AgentOutput(
                agent_name=self.name,
                cycle=packet.cycle,
                contribution=dummy,
                raw_response="[dry-run]",
            )

        total_tokens = None
        try:
            provider = get_provider(self.provider)
            raw_text = provider.complete(
                model=self.model,
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=self.temperature,
            )
            total_tokens = provider.get_last_usage().get("total_tokens")
        except Exception as exc:
            raw_text = f"[ERROR: {exc}]"

        contribution = _parse_deliberation(raw_text)

        return AgentOutput(
            agent_name=self.name,
            cycle=packet.cycle,
            contribution=contribution,
            raw_response=raw_text,
            total_tokens=total_tokens,
        )

    def stream(self, packet: ContextPacket, system_prompt: str, user_message: str):
        """
        Stream an already-built pair chunk by chunk, for live UI rendering.

        After the generator is exhausted, call finalize_stream(packet) to
        get the resulting AgentOutput built from the accumulated text —
        mirrors what send() does for a real (non-dry-run, non-human) call.
        """
        self._last_system_prompt = system_prompt
        self._last_user_message = user_message
        self._stream_chunks: list[str] = []
        self._stream_total_tokens: int | None = None
        try:
            provider = get_provider(self.provider)
            for chunk in provider.stream(
                model=self.model,
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=self.temperature,
            ):
                self._stream_chunks.append(chunk)
                yield chunk
            self._stream_total_tokens = provider.get_last_usage().get("total_tokens")
        except Exception as exc:
            err = f"[ERROR: {exc}]"
            self._stream_chunks.append(err)
            yield err

    def finalize_stream(self, packet: ContextPacket) -> AgentOutput:
        """Build the AgentOutput from the text accumulated by stream()."""
        raw_text = "".join(getattr(self, "_stream_chunks", []))
        contribution = _parse_deliberation(raw_text)
        return AgentOutput(
            agent_name=self.name,
            cycle=packet.cycle,
            contribution=contribution,
            raw_response=raw_text,
            total_tokens=getattr(self, "_stream_total_tokens", None),
        )