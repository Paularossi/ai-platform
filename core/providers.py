"""
core/providers.py

Thin LLM provider wrappers.

Each provider class exposes a single method:

    complete(model, system_prompt, user_message, max_tokens) -> str

Module-level singletons are cached after first construction so the
underlying client objects are created once and reused across all calls
(same model as the old get_openai_client() singleton in agent.py).

Adding a new provider:
  1. Subclass LLMProvider and implement complete().
  2. Add it to PROVIDERS and PROVIDER_MODELS.
  3. Add its env-var key name to API_KEY_ENV_VARS.
  4. The run page will automatically detect it from the agent config and
     prompt the user for the key.
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Supported providers and their model menus
# ---------------------------------------------------------------------------

PROVIDERS: list[str] = ["OpenAI", "Anthropic", "Google"]

PROVIDER_MODELS: dict[str, list[str]] = {
    "OpenAI": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "o1",
        "o1-mini",
    ],
    "Anthropic": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ],
    "Google": [
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ],
}

# Environment variable name that holds the API key for each provider
API_KEY_ENV_VARS: dict[str, str] = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google": "GOOGLE_API_KEY",
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LLMProvider:
    """Abstract base. Subclasses must implement complete()."""

    def complete(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError

    def complete_structured(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        schema: dict,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> dict:
        """
        Call the model in structured/JSON-schema output mode and return the
        parsed dict. Used for control-flow decisions (e.g. Agent 0) where the
        response must conform to a schema rather than free text.

        Raises on any failure — callers must validate/degrade to a safe
        default themselves.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def complete(self, model, system_prompt, user_message,
                 max_tokens=1500, temperature=0.0) -> str:
        response = self._get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def complete_structured(self, model, system_prompt, user_message, schema,
                             max_tokens=1500, temperature=0.0) -> dict:
        response = self._get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": schema, "strict": True},
            },
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, model, system_prompt, user_message,
                 max_tokens=1500, temperature=0.0) -> str:
        response = self._get_client().messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text if response.content else ""

    def complete_structured(self, model, system_prompt, user_message, schema,
                             max_tokens=1500, temperature=0.0) -> dict:
        # Anthropic has no native JSON-schema response mode — force a single
        # tool call whose input_schema is the desired schema.
        kwargs = dict(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=max_tokens,
            tools=[{
                "name": "submit_decision",
                "description": "Submit the structured decision.",
                "input_schema": schema,
            }],
            tool_choice={"type": "tool", "name": "submit_decision"},
        )
        try:
            response = self._get_client().messages.create(temperature=temperature, **kwargs)
        except Exception as exc:
            # Some Anthropic models reject an explicit `temperature` on this
            # specific call path ("temperature is deprecated for this
            # model") — retry without it rather than dropping temperature
            # control for every Anthropic model regardless of whether it
            # actually has this restriction.
            if "temperature" in str(exc).lower() and "deprecated" in str(exc).lower():
                response = self._get_client().messages.create(**kwargs)
            else:
                raise

        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return block.input
        raise ValueError("Anthropic response contained no tool_use block")


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

class GoogleProvider(LLMProvider):
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client()
        return self._client

    def complete(self, model: str, system_prompt: str, user_message: str,
                 max_tokens: int = 1500, temperature: float = 0.0,
                 json_mode: bool = False) -> str:
        """
        Call a Gemini model.

        Parameters
        ----------
        json_mode : bool
            When True, set response_mime_type="application/json" so the model
            returns a JSON object matching the prompt's requested format.
            Use this for peer review calls. Leave False for free-text
            contribution calls so the model returns plain prose.
        """
        from google.genai import types

        config_kwargs: dict = {
            "system_instruction": system_prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
            # Disable safety categories that incorrectly block structured
            # evaluation prompts (peer review scoring). These prompts contain
            # no harmful content — the blocks are false positives from
            # patterns like "scoring", "judging", "0 = bad, 1 = good".
            from google.genai.types import HarmCategory, HarmBlockThreshold
            config_kwargs["safety_settings"] = [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                 "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                 "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                 "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                 "threshold": HarmBlockThreshold.BLOCK_NONE},
            ]

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = self._get_client().models.generate_content(
                model=model,
                contents=user_message,
                config=config,
            )
        except Exception as exc:
            print(f"[GoogleProvider] API error: {exc}")
            return f"[ERROR: {exc}]"

        # Extract text robustly — response.text can be None/empty when the
        # response is blocked. Go via candidates for a reliable path.
        # content can be None (SAFETY block) → accessing .parts raises TypeError.
        try:
            text = response.candidates[0].content.parts[0].text
            if text:
                return text
        except (IndexError, AttributeError, TypeError):
            pass

        # Fallback: surface the finish_reason as a legible error marker
        try:
            reason = response.candidates[0].finish_reason.name
        except (IndexError, AttributeError, TypeError):
            reason = "UNKNOWN"

        if reason == "STOP":
            return ""
        return f"[BLOCKED: finish_reason={reason}]"

    def complete_structured(self, model, system_prompt, user_message, schema,
                             max_tokens=1500, temperature=0.0) -> dict:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = self._get_client().models.generate_content(
            model=model, contents=user_message, config=config,
        )
        text = response.candidates[0].content.parts[0].text
        return json.loads(text)


# ---------------------------------------------------------------------------
# Registry — one singleton per provider
# ---------------------------------------------------------------------------

_registry: dict[str, LLMProvider] = {}


def reset_provider(name: str) -> None:
    """Evict the cached provider instance so the next call rebuilds with the current env key."""
    _registry.pop(name, None)


def get_provider(name: str) -> LLMProvider:
    """Return the cached provider instance for the given name."""
    if name not in _registry:
        if name == "OpenAI":
            _registry[name] = OpenAIProvider()
        elif name == "Anthropic":
            _registry[name] = AnthropicProvider()
        elif name == "Google":
            _registry[name] = GoogleProvider()
        else:
            raise ValueError(
                f"Unknown provider '{name}'. "
                f"Supported: {PROVIDERS}"
            )
    return _registry[name]