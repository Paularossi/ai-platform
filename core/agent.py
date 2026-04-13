"""
core/agent.py

Stateless agent wrapper.

The agent receives a ContextPacket from the hub, builds a prompt,
calls the OpenAI API (gpt-4o for now), parses the structured response,
and returns an AgentOutput.

The agent has no memory between calls - all context comes from the hub
via the ContextPacket.
"""

from __future__ import annotations

import base64
import re
from typing import Any
from pathlib import Path

from core.hub import ContextPacket
from core.state import AgentOutput


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

_MIME_MAP = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "png": "image/png", "webp": "image/webp", "gif": "image/gif",
}

_IMAGE_KEYS = ("image_path", "image_url", "image")


def _image_path_to_data_url(image_path: str) -> str | None:
    """Load an image from disk and return a base64 data URL, or None on failure."""
    try:
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime = _MIME_MAP.get(ext, "image/png")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _resolve_image_url(item_data: dict[str, Any]) -> str | None:
    """
    Resolve image input from item data into a URL consumable by OpenAI image_url.

    Accepted sources:
    - local path in image_path / image
    - remote URL in image_url / image
    - data URL in image_url / image
    """
    for key in _IMAGE_KEYS:
        raw_val = item_data.get(key)
        if not raw_val:
            continue

        val = str(raw_val).strip()
        if not val:
            continue

        lowered = val.lower()
        if lowered.startswith(("http://", "https://", "data:image/")):
            return val

        # If this looks like a local file path, try encoding to a data URL (supports both explicit image_path and cases where "image" contains a path).
        if Path(val).exists():
            data_url = _image_path_to_data_url(val)
            if data_url:
                return data_url

    return None
    
    
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

    parts.append(f"You are {agent_name}, an AI annotation agent. Your role: {agent_role}.")
    parts.append("")

    # Per-agent override prepended before base instructions
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
        parts.append("--- Classification questions ---")
        parts.append(
            "For EACH question, provide your answer in this exact block format:\n"
            "\n"
            "FIELD_NAME:\n"
            "  verdict: CODE\n"
            "  probabilities: CODE1=0.XX, CODE2=0.XX, ...   (all options, must sum to 1.0)\n"
            "  confidence: 0.XX                              (probability of your verdict)\n"
            "\n"
            "After ALL questions, add a pros/cons section:\n"
            "\n"
            "pros:\n"
            "  - <reason supporting the classification>\n"
            "  - <reason supporting the classification>\n"
            "cons:\n"
            "  - <reason against or alternative reading>\n"
            "  - <reason against or alternative reading>\n"
            "\n"
            "Rules:\n"
            "- Probabilities must cover ALL listed option codes and sum to 1.0\n"
            "- For multi-label fields, verdict lists all selected codes: CODE1, CODE2\n"
            "- Do not add any text outside these blocks"
        )
        parts.append("")
        for i, q in enumerate(questions, 1):
            parts.append(f"Q{i}. [{q['field_name']}] {q.get('instruction', '')}")
            for opt in q.get("options", []):
                parts.append(f"   • {opt['code']}: {opt['description']}")
        parts.append("")

    return "\n".join(parts)


def _build_user_message(packet: ContextPacket, questions: list[dict]) -> str:
    parts: list[str] = []

    # Item data (skip image keys - handled as vision input separately)
    parts.append("--- Item to annotate ---")
    for key, value in packet.item_data.items():
        if key.lower() in ("image", "image_url", "image_path"):
            continue
        parts.append(f"{key}: {value}")
    parts.append("")

    # Current label state - always shown
    if packet.current_labels:
        parts.append("--- Current label state (set so far) ---")
        for fname, val in packet.current_labels.items():
            parts.append(f"  {fname}: {val}")
        parts.append("")

    # History from hub, filtered by visibility_mode
    if packet.visible_history:
        if len(packet.visible_history) == 1:
            prev = packet.visible_history[0]
            parts.append(f"--- Previous submission (by {prev.agent_name}, cycle {prev.cycle + 1}) ---")
            for fname, val in prev.labels.items():
                changed = prev.changed.get(fname, False)
                prob_str = ""
                if fname in prev.probabilities:
                    prob_str = "  probs: " + ", ".join(
                        f"{k}={v:.2f}" for k, v in prev.probabilities[fname].items()
                    )
                parts.append(f"  {fname}: {val}{' [revised]' if changed else ''}{prob_str}")
            if prev.pros:
                parts.append("  pros: " + "; ".join(prev.pros))
            if prev.cons:
                parts.append("  cons: " + "; ".join(prev.cons))
        else:
            parts.append("--- Submission history (from hub) ---")
            for out in packet.visible_history:
                parts.append(f"[Cycle {out.cycle + 1} | {out.agent_name}]")
                for fname, val in out.labels.items():
                    changed = out.changed.get(fname, False)
                    prob_str = ""
                    if fname in out.probabilities:
                        prob_str = "  probs: " + ", ".join(
                            f"{k}={v:.2f}" for k, v in out.probabilities[fname].items()
                        )
                    parts.append(f"  {fname}: {val}{' [revised]' if changed else ''}{prob_str}")
                if out.pros:
                    parts.append("  pros: " + "; ".join(out.pros))
                if out.cons:
                    parts.append("  cons: " + "; ".join(out.cons))
        parts.append("")

    # Task prompt
    if packet.cycle > 0 or packet.visible_history:
        parts.append(
            f"This is cycle {packet.cycle + 1}. "
            "Review the current labels and revise any you disagree with, "
            "or confirm them if you agree. Answer ALL questions."
        )
    else:
        parts.append("Please answer all questions about the item above.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def _parse_output(
    raw_text: str,
    questions: list[dict],
) -> tuple[dict, dict, dict, list[str], list[str]]:
    """
    Parse raw LLM text into (labels, probabilities, confidence, pros, cons).

    Expected block format per field:
        FIELD_NAME:
          verdict: CODE
          probabilities: CODE1=0.70, CODE2=0.20, CODE3=0.10
          confidence: 0.70

    Followed by:
        pros:
          - reason one
          - reason two
        cons:
          - reason one

    Also handles the legacy asterisk format for backwards compatibility:
        *field_name*: CODE | reasoning
    """
    labels: dict[str, Any] = {}
    probabilities: dict[str, dict[str, float]] = {}
    confidence: dict[str, float] = {}
    pros: list[str] = []
    cons: list[str] = []

    field_types = {q["field_name"]: q["field_type"] for q in questions}
    field_options = {
        q["field_name"]: [o["code"] for o in q.get("options", [])]
        for q in questions
    }

    # ── Block format parser ───────────────────────────────────────────────────
    # Split into per-field blocks by looking for lines that start with a known
    # field name followed by a colon (the block header).
    lines = raw_text.splitlines()
    current_field: str | None = None
    in_pros = False
    in_cons = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # try two different header formats
        # FIELD_NAME: target_age
        field_name_match = re.match(r"^field_name\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*$", stripped, re.IGNORECASE)
        if field_name_match:
            fname = field_name_match.group(1).strip()
            if fname in field_types:
                current_field = fname
                in_pros, in_cons = False, False
                continue

        # Q1. [target_age]
        q_header_match = re.match(r"^q\d+\.?\s*\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]\s*$", stripped, re.IGNORECASE)
        if q_header_match:
            fname = q_header_match.group(1).strip()
            if fname in field_types:
                current_field = fname
                in_pros, in_cons = False, False
                continue

        # Detect pros/cons section headers
        if stripped.lower().startswith("pros:") or stripped.lower() == "pros":
            in_pros, in_cons, current_field = True, False, None
            continue
        if stripped.lower().startswith("cons:") or stripped.lower() == "cons":
            in_cons, in_pros, current_field = True, False, None
            continue

        # Collect pros/cons bullet points
        if in_pros and stripped.startswith("-"):
            pros.append(stripped[1:].strip())
            continue
        if in_cons and stripped.startswith("-"):
            cons.append(stripped[1:].strip())
            continue

        # if there are no bullet markers for the pros/cons lines.
        if in_pros:
            pros.append(stripped)
            continue
        if in_cons:
            cons.append(stripped)
            continue

        # Detect a field block header: "field_name:" at the start of a line
        header_match = re.match(r"^\*?([a-zA-Z_][a-zA-Z0-9_]*)\*?\s*:(.*)$", stripped)
        if header_match:
            fname = header_match.group(1).strip()
            rest = header_match.group(2).strip()

            if fname in field_types:
                current_field = fname
                in_pros, in_cons = False, False

                # Could be block header (rest is empty) or inline legacy format
                if rest:
                    # field_name: CODE | reasoning  OR  field_name: CODE - reasoning
                    value_str = re.split(r"\s*[|\-]\s*", rest)[0].strip()
                    _store_label(fname, value_str, field_types, labels)
                continue

            # Sub-keys inside a field block (case-insensitive)
            fname_l = fname.lower()
            if current_field and fname_l == "verdict":
                value_str = rest.strip()
                _store_label(current_field, value_str, field_types, labels)
                continue

            if current_field and fname_l == "probabilities":
                probs = _parse_probabilities(rest)
                if probs:
                    probabilities[current_field] = probs
                continue

            if current_field and fname_l == "confidence":
                try:
                    conf_val = float(rest.strip())
                    if conf_val > 1.0:
                        conf_val /= 100.0
                    confidence[current_field] = conf_val
                except ValueError:
                    pass
                continue

        # Indented sub-keys (with leading spaces)
        indent_match = re.match(r"^\s+(verdict|probabilities|confidence)\s*:\s*(.+)$", line, re.IGNORECASE)
        if indent_match and current_field:
            key = indent_match.group(1).lower()
            val = indent_match.group(2).strip()
            if key == "verdict":
                _store_label(current_field, val, field_types, labels)
            elif key == "probabilities":
                probs = _parse_probabilities(val)
                if probs:
                    probabilities[current_field] = probs
            elif key == "confidence":
                try:
                    conf_val = float(val)
                    if conf_val > 1.0:
                        conf_val /= 100.0
                    confidence[current_field] = conf_val
                except ValueError:
                    pass
            continue

    # ── Derive confidence from probabilities where missing ───────────────────
    for fname, label_val in labels.items():
        if fname not in confidence and fname in probabilities:
            verdict_code = label_val if isinstance(label_val, str) else (label_val[0] if label_val else None)
            if verdict_code and verdict_code in probabilities[fname]:
                confidence[fname] = probabilities[fname][verdict_code]

    return labels, probabilities, confidence, pros, cons


def _store_label(fname: str, value_str: str, field_types: dict, labels: dict) -> None:
    """Parse a verdict string and store into labels dict."""
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
    """Parse 'CODE1=0.70, CODE2=0.20' into {CODE1: 0.70, CODE2: 0.20}."""
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
        self.role: str = config.get("role", "Annotator")

        self._instructions = experiment_config.get("instructions", {})
        self._questions = experiment_config.get("questions", [])
        self._overrides = experiment_config.get("agent_prompt_overrides", {})

    def call(self, packet: ContextPacket, dry_run: bool = False) -> AgentOutput:
        """
        Receive a ContextPacket from the hub, call the LLM, return AgentOutput.

        Parameters
        ----------
        packet : ContextPacket
            Built by the hub for this agent's turn.
        dry_run : bool
            Skip the LLM call and return a placeholder. Useful for UI testing.
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
            dummy_labels = {
                q["field_name"]: (
                    q["options"][0]["code"] if q.get("options") else "UNKNOWN"
                )
                for q in self._questions
            }
            dummy_probs = {
                q["field_name"]: {
                    opt["code"]: round(1.0 / len(q["options"]), 2)
                    for opt in q.get("options", [])
                }
                for q in self._questions if q.get("options")
            }
            dummy_conf = {
                q["field_name"]: round(1.0 / max(len(q.get("options", [1])), 1), 2)
                for q in self._questions
            }
            return AgentOutput(
                agent_name=self.name,
                cycle=packet.cycle,
                labels=dummy_labels,
                probabilities=dummy_probs,
                confidence=dummy_conf,
                pros=["[dry-run placeholder]"],
                cons=["[dry-run placeholder]"],
                raw_response="[dry-run]",
            )

        try:
            from openai import OpenAI  # lazy import - not required at module level

            client = OpenAI()  # reads OPENAI_API_KEY from environment
            
            # Build user content - multimodal whenever image data can be resolved.
            image_url = _resolve_image_url(packet.item_data)
            if image_url:
                user_content: Any = [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            else:
                # Keep text-only fallback, but retain explicit image refs if provided
                # so the model still receives image context when attachment fails.
                fallback_refs = [
                    str(packet.item_data.get(k)).strip()
                    for k in _IMAGE_KEYS
                    if packet.item_data.get(k)
                ]
                if fallback_refs:
                    user_content = (
                        user_message
                        + "\n\n[Image reference provided but could not be attached as vision input]: "
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
                max_tokens=1500,  # increased to fit full probability distributions
            )
            raw_text = response.choices[0].message.content or ""

        except Exception as exc:
            raw_text = f"[ERROR: {exc}]"

        labels, probs, conf, pros, cons = _parse_output(raw_text, self._questions)

        return AgentOutput(
            agent_name=self.name,
            cycle=packet.cycle,
            labels=labels,
            probabilities=probs,
            confidence=conf,
            pros=pros,
            cons=cons,
            raw_response=raw_text,
        )