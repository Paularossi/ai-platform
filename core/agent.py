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
    questions: list[dict],
) -> str:
    parts: list[str] = []

    parts.append(f"You are {agent_name}. Your role: {agent_role}.")
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

    if questions:
        # Text/score-only questions without option codes
        parts.append("--- Questions ---")
        parts.append("Answer each question on its own line:")
        parts.append("  field_name: your answer")
        parts.append("")
        for i, q in enumerate(questions, 1):
            parts.append(f"Q{i}. [{q['field_name']}] {q.get('instruction', '')}")
        parts.append("")

    # Deliberation mode — no structured questions.
    # Output format is entirely defined by the user's base_instructions.

    return "\n".join(parts)


def _build_user_message(packet: ContextPacket, questions: list[dict]) -> str:
    parts: list[str] = []

    # ── Item data (from dataset) ──────────────────────────────────────────────
    for key, value in packet.item_data.items():
        parts.append(f"{key}: {value}")
    parts.append("")

    # ── Previous round context (from hub, cycle > 0) ──────────────────────────
    if packet.visible_history:
        parts.append("=== Previous round ===")
        parts.append("")

        # Group history by agent for clean display
        by_agent: dict[str, Any] = {}
        for out in packet.visible_history:
            by_agent[out.agent_name] = out

        parts.append("Contributions:")
        for agent_name, out in by_agent.items():
            if agent_name == packet.agent_name:
                label = f"{agent_name} (your previous contribution)"
            else:
                label = agent_name
            contrib = str(out.contribution) if out.contribution else "(none)"
            parts.append(f"  [{label}]: {contrib}")
        parts.append("")

        # Show peer review scores if available
        scores_shown = False
        for out in packet.visible_history:
            if out.ecu_scores:
                if not scores_shown:
                    parts.append("Peer review scores (average received):")
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


def _parse_deliberation(
    raw_text: str,
    questions: list[dict],
) -> tuple[str, dict, float | None, list[str], list[str]]:
    """
    Parse free-text deliberation response.
    Extracts the main contribution, confidence (if reported), pros and cons.
    """
    pros: list[str] = []
    cons: list[str] = []
    confidence: float | None = None
    contribution_lines: list[str] = []

    lines = raw_text.splitlines()
    in_pros = False
    in_cons = False
    in_contribution = True

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.lower().startswith("pros:") or stripped.lower() == "pros":
            in_pros, in_cons, in_contribution = True, False, False
            continue
        if stripped.lower().startswith("cons:") or stripped.lower() == "cons":
            in_cons, in_pros, in_contribution = True, False, False
            continue

        # Confidence line: "confidence: 0.85"
        conf_match = re.match(r"^confidence\s*:\s*([0-9.]+)", stripped, re.IGNORECASE)
        if conf_match:
            in_contribution = False
            try:
                v = float(conf_match.group(1))
                confidence = v / 100.0 if v > 1.0 else v
            except ValueError:
                pass
            continue

        if in_pros:
            text = stripped[1:].strip() if stripped.startswith("-") else stripped
            pros.append(text)
        elif in_cons:
            text = stripped[1:].strip() if stripped.startswith("-") else stripped
            cons.append(text)
        elif in_contribution:
            contribution_lines.append(stripped)

    # For text-only questions, also parse simple field: answer lines
    if questions:
        field_answers: dict[str, str] = {}
        field_types = {q["field_name"]: q["field_type"] for q in questions}
        for line in lines:
            m = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$", line)
            if m and m.group(1) in field_types:
                field_answers[m.group(1)] = m.group(2).strip()
        if field_answers:
            return field_answers, {}, confidence, pros, cons

    contribution = " ".join(contribution_lines).strip() or raw_text.strip()
    return contribution, {}, confidence, pros, cons


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
        Full experiment config (instructions, questions, overrides).
    """

    def __init__(self, config: dict[str, Any], experiment_config: dict[str, Any]):
        self.name: str = config.get("name", "Agent")
        self.provider: str = config.get("provider", "OpenAI")
        self.model: str = config.get("model", "gpt-4o")

        # Resolve effective role: if the dropdown says "Custom", use the
        # custom_role text the user typed; otherwise use the dropdown value.
        raw_role = config.get("role", "Participant")
        custom_role_text = config.get("custom_role", "").strip()
        self.role: str = custom_role_text if raw_role == "Custom" and custom_role_text else raw_role

        self._instructions = experiment_config.get("instructions", {})
        self._questions = experiment_config.get("questions", [])
        self._overrides = experiment_config.get("agent_prompt_overrides", {})

    def call(self, packet: ContextPacket, dry_run: bool = False) -> AgentOutput:
        """
        Receive a ContextPacket from the hub, call the LLM, return AgentOutput.

        Parameters
        ----------
        dry_run : bool
            If True, skip the API call and return a placeholder contribution.
            Used by the test suite to exercise the full pipeline without
            incurring API costs.
        """
        system_prompt = _build_system_prompt(
            agent_name=self.name,
            agent_role=self.role,
            base_instructions=self._instructions.get("base_instructions", ""),
            guideline_notes=self._instructions.get("guideline_notes", ""),
            agent_overrides=self._overrides,
            questions=self._questions,
        )

        user_message = _build_user_message(packet, self._questions)

        if dry_run:
            dummy = "[dry-run: no contribution]"
            return AgentOutput(
                agent_name=self.name,
                cycle=packet.cycle,
                contribution=dummy,
                raw_response="[dry-run]",
            )

        try:
            provider = get_provider(self.provider)
            raw_text = provider.complete(
                model=self.model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except Exception as exc:
            raw_text = f"[ERROR: {exc}]"

        contribution, _, _, _, _ = _parse_deliberation(raw_text, self._questions)

        return AgentOutput(
            agent_name=self.name,
            cycle=packet.cycle,
            contribution=contribution,
            raw_response=raw_text,
        )