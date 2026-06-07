"""
core/agent.py

Stateless agent wrapper.

The agent receives a ContextPacket from the hub, builds a prompt,
calls the OpenAI API, parses the structured response, and returns
an AgentOutput.

Two task modes are supported, determined by whether the experiment
config contains structured questions:

  Classification mode  — questions with option codes are present.
                         Output: contribution = dict[field→verdict],
                         prob_distribution populated, confidence derived.

  Deliberation mode    — no structured questions (or text-only).
                         Output: contribution = str (free-text statement),
                         prob_distribution empty, confidence self-reported.

The agent has no memory between calls — all context comes from the hub.
"""

from __future__ import annotations

import base64
import re
from typing import Any
from pathlib import Path

from core.hub import ContextPacket
from core.state import AgentOutput


# ---------------------------------------------------------------------------
# Image helpers (unchanged — multimodal input still supported)
# ---------------------------------------------------------------------------

_MIME_MAP = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "png": "image/png", "webp": "image/webp", "gif": "image/gif",
}
_IMAGE_KEYS = ("image_path", "image_url", "image")


def _image_path_to_data_url(image_path: str) -> str | None:
    try:
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime = _MIME_MAP.get(ext, "image/png")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _resolve_image_url(item_data: dict[str, Any]) -> str | None:
    for key in _IMAGE_KEYS:
        raw_val = item_data.get(key)
        if not raw_val:
            continue
        val = str(raw_val).strip()
        if not val:
            continue
        if val.lower().startswith(("http://", "https://", "data:image/")):
            return val
        if Path(val).exists():
            return _image_path_to_data_url(val)
    return None


# ---------------------------------------------------------------------------
# Task mode detection
# ---------------------------------------------------------------------------

def _is_classification_task(questions: list[dict]) -> bool:
    """
    True if the task has structured questions with option codes.
    False for deliberation tasks (no questions, or text/score-only questions).
    """
    if not questions:
        return False
    choice_types = {"single_label", "multi_label"}
    return any(
        q.get("field_type", "single_label") in choice_types
        for q in questions
    )


# ---------------------------------------------------------------------------
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

    if _is_classification_task(questions):
        # ── Classification mode ───────────────────────────────────────────
        parts.append("--- Questions ---")

        choice_types = {"single_label", "multi_label"}
        has_choice = any(q.get("field_type", "single_label") in choice_types for q in questions)
        example_field = questions[0]["field_name"]

        fmt: list[str] = [
            "For EACH question, respond using this exact block format:",
            "",
            f"{example_field}:",
            "  verdict: <your answer>",
        ]
        if has_choice:
            fmt += [
                "  probabilities: CODE1=0.XX, CODE2=0.XX, ...   (must sum to 1.0)",
                "  confidence: 0.XX",
            ]
        fmt += [
            "",
            "After ALL questions, add:",
            "pros:",
            "  - <reason supporting your answers>",
            "cons:",
            "  - <reason against or alternative reading>",
            "",
            "Rules:",
            "- Use the EXACT field name from each question as the block header. "
            "Expected headers: "
            + "  ".join(f"{q['field_name']}:" for q in questions),
            "- For single/multi-label fields: verdict is the option code (e.g. YES, NO)",
            "- For multi-label fields: list all selected codes separated by commas",
            "- For boolean fields: verdict is true or false",
            "- For text fields: verdict is your free-text answer",
            "- For score fields: verdict is a number",
        ]
        if has_choice:
            fmt += [
                "- Probabilities must cover ALL option codes and sum to 1.0",
                "- Omit probabilities and confidence for text, score, and boolean fields",
            ]
        fmt.append("- Do not add any text outside these blocks")

        parts.append("\n".join(fmt))
        parts.append("")

        for i, q in enumerate(questions, 1):
            ftype = q.get("field_type", "single_label")
            parts.append(f"Q{i}. [{q['field_name']}] {q.get('instruction', '')}")
            for opt in q.get("options", []):
                parts.append(f"   • {opt['code']}: {opt['description']}")
            if ftype == "boolean":
                parts.append("   (verdict: true or false)")
            elif ftype == "text":
                parts.append("   (verdict: free-text answer)")
            elif ftype == "score":
                parts.append("   (verdict: numeric score)")
        parts.append("")

    elif questions:
        # ── Text/score questions without option codes ─────────────────────
        # Still structured but no probability distribution needed.
        parts.append("--- Questions ---")
        parts.append("Answer each question on its own line:")
        parts.append("  field_name: your answer")
        parts.append("")
        for i, q in enumerate(questions, 1):
            parts.append(f"Q{i}. [{q['field_name']}] {q.get('instruction', '')}")
        parts.append("")
        parts.append("After your answers, add:")
        parts.append("pros:")
        parts.append("  - <reason supporting your answers>")
        parts.append("cons:")
        parts.append("  - <reason against or alternative reading>")
        parts.append("")

    else:
        # ── Deliberation mode — no structured questions ───────────────────
        # Output format is entirely defined by the user's base_instructions.
        # We add nothing here — the user decides what structure they want.
        pass

    return "\n".join(parts)


def _build_user_message(packet: ContextPacket, questions: list[dict]) -> str:
    parts: list[str] = []

    # ── Item data (from dataset) ──────────────────────────────────────────────
    for key, value in packet.item_data.items():
        if key.lower() in ("image", "image_url", "image_path"):
            continue
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
            # Determine what to show based on which agent this is
            own_balance = packet.ecu_balances.get(packet.agent_name)
            if own_balance is not None and len(packet.ecu_balances) == 1:
                # Semi-transparent: only own balance was passed
                parts.append(f"Your current ECU balance: {own_balance:.2f}")
            else:
                # Transparent: all balances
                bal_str = ", ".join(f"{k}: {v:.2f}" for k, v in packet.ecu_balances.items())
                parts.append(f"Current ECU balances: {bal_str}")
            parts.append("")

        parts.append("=== Your turn ===")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def _parse_contribution(
    raw_text: str,
    questions: list[dict],
) -> tuple[Any, dict, float | None, list[str], list[str]]:
    """
    Parse raw LLM text into (contribution, prob_distribution, confidence, pros, cons).

    For classification tasks:
        contribution    = dict[field_name → verdict]
        prob_distribution = dict[field_name → {code: float}]
        confidence      = mean of per-field confidence values (float | None)

    For deliberation tasks:
        contribution    = str (the full statement, stripped of pros/cons/confidence)
        prob_distribution = {}
        confidence      = float | None (if agent reported one)
    """
    if _is_classification_task(questions):
        return _parse_classification(raw_text, questions)
    else:
        return _parse_deliberation(raw_text, questions)


def _parse_classification(
    raw_text: str,
    questions: list[dict],
) -> tuple[dict, dict, float | None, list[str], list[str]]:
    """Parse structured classification block format."""
    labels: dict[str, Any] = {}
    prob_distribution: dict[str, dict[str, float]] = {}
    per_field_confidence: dict[str, float] = {}
    pros: list[str] = []
    cons: list[str] = []

    field_types = {q["field_name"]: q["field_type"] for q in questions}
    lines = raw_text.splitlines()
    current_field: str | None = None
    in_pros = False
    in_cons = False
    _placeholder_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Alternate header formats
        field_name_match = re.match(r"^field_name\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*$", stripped, re.IGNORECASE)
        if field_name_match:
            fname = field_name_match.group(1).strip()
            if fname in field_types:
                current_field = fname
                in_pros, in_cons = False, False
                continue

        q_header_match = re.match(r"^q\d+\.?\s*\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*$", stripped, re.IGNORECASE)
        if q_header_match:
            fname = q_header_match.group(1).strip()
            if fname in field_types:
                current_field = fname
                in_pros, in_cons = False, False
                continue

        if stripped.lower().startswith("pros:") or stripped.lower() == "pros":
            in_pros, in_cons, current_field = True, False, None
            continue
        if stripped.lower().startswith("cons:") or stripped.lower() == "cons":
            in_cons, in_pros, current_field = True, False, None
            continue

        if in_pros:
            text = stripped[1:].strip() if stripped.startswith("-") else stripped
            pros.append(text)
            continue
        if in_cons:
            text = stripped[1:].strip() if stripped.startswith("-") else stripped
            cons.append(text)
            continue

        header_match = re.match(r"^\*?([a-zA-Z_][a-zA-Z0-9_]*)\*?\s*:(.*)$", stripped)
        if header_match:
            fname = header_match.group(1).strip()
            rest = header_match.group(2).strip()

            if fname in field_types:
                current_field = fname
                in_pros, in_cons = False, False
                if rest:
                    value_str = re.split(r"\s*[|\-]\s*", rest)[0].strip()
                    _store_label(fname, value_str, field_types, labels)
                continue

            if fname.upper() == "FIELD_NAME":
                if _placeholder_count < len(questions):
                    current_field = questions[_placeholder_count]["field_name"]
                    _placeholder_count += 1
                    in_pros, in_cons = False, False
                    if rest:
                        value_str = re.split(r"\s*[|\-]\s*", rest)[0].strip()
                        _store_label(current_field, value_str, field_types, labels)
                continue

            fname_l = fname.lower()
            if current_field and fname_l == "verdict":
                _store_label(current_field, rest.strip(), field_types, labels)
                continue
            if current_field and fname_l == "probabilities":
                probs = _parse_probabilities(rest)
                if probs:
                    prob_distribution[current_field] = probs
                continue
            if current_field and fname_l == "confidence":
                try:
                    v = float(rest.strip())
                    per_field_confidence[current_field] = v / 100.0 if v > 1.0 else v
                except ValueError:
                    pass
                continue

        indent_match = re.match(r"^\s+(verdict|probabilities|confidence)\s*:\s*(.+)$", line, re.IGNORECASE)
        if indent_match and current_field:
            key = indent_match.group(1).lower()
            val = indent_match.group(2).strip()
            if key == "verdict":
                _store_label(current_field, val, field_types, labels)
            elif key == "probabilities":
                probs = _parse_probabilities(val)
                if probs:
                    prob_distribution[current_field] = probs
            elif key == "confidence":
                try:
                    v = float(val)
                    per_field_confidence[current_field] = v / 100.0 if v > 1.0 else v
                except ValueError:
                    pass
            continue

    # Derive missing per-field confidence from prob_distribution
    for fname, label_val in labels.items():
        if fname not in per_field_confidence and fname in prob_distribution:
            verdict_code = label_val if isinstance(label_val, str) else (label_val[0] if label_val else None)
            if verdict_code and verdict_code in prob_distribution[fname]:
                per_field_confidence[fname] = prob_distribution[fname][verdict_code]

    # Overall confidence = mean of per-field values
    confidence: float | None = None
    if per_field_confidence:
        confidence = round(sum(per_field_confidence.values()) / len(per_field_confidence), 4)

    return labels, prob_distribution, confidence, pros, cons


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
    if questions and not _is_classification_task(questions):
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


def _store_label(fname: str, value_str: str, field_types: dict, labels: dict) -> None:
    ftype = field_types.get(fname, "single_label")
    if ftype == "multi_label":
        labels[fname] = [v.strip() for v in value_str.split(",") if v.strip()]
    elif ftype == "boolean":
        labels[fname] = value_str.lower() in ("true", "yes", "1")
    elif ftype == "score":
        try:
            labels[fname] = float(value_str)
        except ValueError:
            labels[fname] = value_str
    else:
        labels[fname] = value_str


def _parse_probabilities(raw: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if "=" in part:
            code, _, val = part.partition("=")
            try:
                result[code.strip()] = float(val.strip())
            except ValueError:
                pass
    return result


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
        is_classification = _is_classification_task(self._questions)

        if dry_run:
            if is_classification:
                dummy_contribution = {
                    q["field_name"]: (q["options"][0]["code"] if q.get("options") else "UNKNOWN")
                    for q in self._questions
                }
                dummy_probs = {
                    q["field_name"]: {
                        opt["code"]: round(1.0 / len(q["options"]), 2)
                        for opt in q.get("options", [])
                    }
                    for q in self._questions if q.get("options")
                }
                n_fields = max(len(self._questions), 1)
                dummy_conf = round(1.0 / max(len(self._questions[0].get("options", [1])), 1), 2) if self._questions else 0.5
            else:
                dummy_contribution = "[dry-run: no contribution]"
                dummy_probs = {}
                dummy_conf = 0.5

            return AgentOutput(
                agent_name=self.name,
                cycle=packet.cycle,
                contribution=dummy_contribution,
                prob_distribution=dummy_probs,
                confidence=dummy_conf,
                pros=["[dry-run]"],
                cons=["[dry-run]"],
                raw_response="[dry-run]",
            )

        try:
            from openai import OpenAI
            client = OpenAI()

            image_url = _resolve_image_url(packet.item_data)
            if image_url:
                user_content: Any = [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            else:
                fallback_refs = [
                    str(packet.item_data.get(k)).strip()
                    for k in _IMAGE_KEYS if packet.item_data.get(k)
                ]
                if fallback_refs:
                    user_content = (
                        user_message
                        + "\n\n[Image reference (could not attach as vision input)]: "
                        + ", ".join(fallback_refs)
                    )
                else:
                    user_content = user_message

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=1500,
            )
            raw_text = response.choices[0].message.content or ""

        except Exception as exc:
            raw_text = f"[ERROR: {exc}]"

        contribution, prob_dist, confidence, pros, cons = _parse_contribution(
            raw_text, self._questions
        )

        return AgentOutput(
            agent_name=self.name,
            cycle=packet.cycle,
            contribution=contribution,
            prob_distribution=prob_dist,
            confidence=confidence,
            pros=pros,
            cons=cons,
            raw_response=raw_text,
        )