"""Provider adapters with consistent validation, token usage and request diagnostics."""
from __future__ import annotations

import json
import re
from threading import local

from core.llm.call_log import audit_call
from core.llm.usage import normalize_usage, _plain

PROVIDER_MODELS = {
    "OpenAI": ["gpt-6-luna", "gpt-4.1-mini", "gpt-6.1-sol"],
    "Anthropic": ["claude-sonnet-5-5", "claude-haiku-4-5"],
    "Google": ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
}
PROVIDERS = list(PROVIDER_MODELS)
API_KEY_ENV_VARS = {"OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY", "Google": "GOOGLE_API_KEY"}


def uses_default_temperature(provider, model):
    if provider == "OpenAI":
        return bool(re.match(r"(?:gpt-[5-9]|o[1-9])", model))
    if provider == "Anthropic":
        return bool(re.match(r"claude-(?:sonnet|opus|haiku)-[5-9]", model)
                    or model.startswith(("claude-opus-4-7", "claude-opus-4-8")))
    return provider == "Google" and model.startswith("gemini-3")


def openai_options(model, max_tokens, temperature):
    options = {"max_completion_tokens": max_tokens}
    if uses_default_temperature("OpenAI", model):
        options["reasoning_effort"] = "none" if model.startswith("gpt-6-luna") else "low"
    else:
        options["temperature"] = temperature
    return options


def effective_output_limit(provider, model, requested):
    """One source for adapter limits and live-check cost previews."""
    floor = 0
    if provider == "Anthropic":
        floor = 8192 if model.startswith("claude-sonnet-5-5") else (
            4096 if model.startswith("claude-haiku-4-5") else 0)
    elif provider == "Google" and model.startswith("gemini-3"):
        floor = 8192
    return max(requested, floor)


def anthropic_options(model, max_tokens, temperature):
    # Thinking and answer text share this limit. Keep adaptive thinking enabled.
    options = {"max_tokens": effective_output_limit("Anthropic", model, max_tokens)}
    if not uses_default_temperature("Anthropic", model):
        options["temperature"] = temperature
    return options


class LLMProvider:
    provider = ""

    def __init__(self):
        self._client = None
        self._state = local()

    def reset_usage(self):
        self._state.usage = normalize_usage(self.provider, None)
        self._state.response = {}
        self._state.last_call = None
        self._state.web_search = False

    def get_last_usage(self):
        return dict(getattr(self._state, "usage", normalize_usage(self.provider, None)))

    def get_last_response(self):
        return dict(getattr(self._state, "response", {}))

    @property
    def last_call(self):
        return getattr(self._state, "last_call", None)

    @last_call.setter
    def last_call(self, value):
        self._state.last_call = value

    def _parameters(self, options):
        self._state.response["parameters"] = options

    def _response(self, response, text, reason, usage, block_types=None):
        self._state.usage = normalize_usage(self.provider, usage)
        self._state.response.update(response=text, stop_reason=getattr(reason, "value", reason),
                                    response_id=getattr(response, "id", None) or getattr(response, "response_id", None),
                                    block_types=block_types or [])
        request_id = getattr(response, "_request_id", None)
        if request_id:
            self._state.response["request_id"] = request_id

    def _require_text(self, text, reason, accepted):
        if reason not in accepted:
            raise ValueError(f"Incomplete or blocked response: stop_reason={reason}. "
                             "The output limit includes any thinking tokens; inspect API call diagnostics.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Model returned no answer text: stop_reason={reason}. Inspect API call diagnostics.")

    @audit_call
    def complete(self, model, system_prompt, user_message, max_tokens=1500, temperature=0.0, json_mode=False, web_search=False):
        self._state.web_search = bool(web_search)
        return self._request(model, system_prompt, user_message, max_tokens, temperature, None, json_mode)

    @audit_call
    def complete_structured(self, model, system_prompt, user_message, schema, max_tokens=1500, temperature=0.0):
        text = self._request(model, system_prompt, user_message, max_tokens, temperature, schema, True)
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Structured response must be a JSON object")
        return parsed


class OpenAIProvider(LLMProvider):
    provider = "OpenAI"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _request(self, model, system_prompt, user_message, max_tokens, temperature, schema, json_mode):
        options = openai_options(model, max_tokens, temperature)
        if self._state.web_search:
            options["max_output_tokens"] = options.pop("max_completion_tokens")
            if "reasoning_effort" in options:
                options["reasoning"] = {"effort": options.pop("reasoning_effort")}
            options.update(tools=[{"type": "web_search"}], max_tool_calls=2,
                           include=["web_search_call.action.sources"])
            self._parameters(options)
            response = self._get_client().responses.create(
                model=model, instructions=system_prompt, input=user_message, **options)
            raw = _plain(response)
            self._response(response, response.output_text, response.status, getattr(response, "usage", None))
            self._state.response["web_search"] = {"output": raw.get("output", []),
                "search_calls": sum(item.get("type") == "web_search_call" and
                    item.get("action", {}).get("type") == "search" for item in raw.get("output", []))}
            self._require_text(response.output_text, response.status, ("completed",))
            from core.llm.search import openai_text
            return openai_text(raw.get("output", [])) or response.output_text
        if schema is not None:
            options["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "decision", "schema": schema, "strict": True}}
        self._parameters(options)
        response = self._get_client().chat.completions.create(
            model=model, messages=[{"role": "system", "content": system_prompt},
                                   {"role": "user", "content": user_message}], **options)
        choice = response.choices[0]
        text = choice.message.content or ""
        self._response(response, text, choice.finish_reason, getattr(response, "usage", None))
        if getattr(choice.message, "refusal", None):
            raise ValueError(f"Model refused: {choice.message.refusal}")
        self._require_text(text, choice.finish_reason, ("stop",))
        return text


class AnthropicProvider(LLMProvider):
    provider = "Anthropic"

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _request(self, model, system_prompt, user_message, max_tokens, temperature, schema, json_mode):
        options = anthropic_options(model, max_tokens, temperature)
        if self._state.web_search:
            options["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}]
        native = (uses_default_temperature("Anthropic", model)
                  or model.startswith(("claude-haiku-4-5", "claude-sonnet-4-5", "claude-sonnet-4-6",
                                       "claude-opus-4-5", "claude-opus-4-6")))
        if schema is not None:
            if native:
                options["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
            else:
                # older models in saved configurations still support forced tools
                options["tools"] = [{"name": "submit_decision", "description": "Submit the decision.", "input_schema": schema}]
                options["tool_choice"] = {"type": "tool", "name": "submit_decision"}
        self._parameters(options)
        response = self._get_client().messages.create(model=model, system=system_prompt,
            messages=[{"role": "user", "content": user_message}], **options)
        text = "".join(b.text for b in response.content if b.type == "text")
        if schema is not None and not native:
            blocks = [b.input for b in response.content if b.type == "tool_use"]
            if blocks:
                text = json.dumps(blocks[0])
        self._response(response, text, response.stop_reason, getattr(response, "usage", None),
                       [b.type for b in response.content])
        if self._state.web_search:
            blocks = _plain(response.content)
            self._state.response["web_search"] = {"content": blocks,
                "search_calls": ((_plain(getattr(response, "usage", None)) or {}).get("server_tool_use") or {}).get("web_search_requests")}
            from core.llm.search import anthropic_text
            text = anthropic_text(blocks)
        accepted = ("tool_use",) if schema is not None and not native else ("end_turn",)
        self._require_text(text, response.stop_reason, accepted)
        return text


class GoogleProvider(LLMProvider):
    provider = "Google"

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client()
        return self._client

    def _request(self, model, system_prompt, user_message, max_tokens, temperature, schema, json_mode):
        from google.genai import types
        # Gemini 3 shares the output cap between reasoning and answer text.
        # A 1500-token cap can end a contribution before any answer is emitted.
        options = {"max_output_tokens": effective_output_limit("Google", model, max_tokens)}
        if model.startswith("gemini-3"):
            options["thinking_config"] = {"thinking_level":
                "minimal" if model.startswith("gemini-3.5-flash-lite") else "low"}
        if not uses_default_temperature("Google", model):
            options["temperature"] = temperature
        if self._state.web_search:
            options["tools"] = [{"google_search": {}}]
        if json_mode:
            options["response_mime_type"] = "application/json"
            if schema is None:
                # Preserve the existing peer-review safety configuration.
                options["safety_settings"] = [
                    {"category": category, "threshold": "BLOCK_NONE"}
                    for category in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                                     "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
                ]
        if schema is not None:
            options["response_json_schema"] = schema
        self._parameters(options)
        response = self._get_client().models.generate_content(model=model, contents=user_message,
            config=types.GenerateContentConfig(system_instruction=system_prompt, **options))
        candidates = response.candidates or []
        candidate = candidates[0] if candidates else None
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        reason = getattr(candidate, "finish_reason", "NO_CANDIDATES")
        self._response(response, text, reason, getattr(response, "usage_metadata", None))
        if self._state.web_search:
            metadata = _plain(getattr(candidate, "grounding_metadata", None)) or {}
            self._state.response["web_search"] = {"grounding_metadata": metadata,
                "query_count": len(metadata["web_search_queries"]) if metadata.get("web_search_queries") is not None else None}
            from core.llm.search import google_text
            text = google_text(text, metadata)
        self._require_text(text, reason, ("STOP",))
        return text


_registry = {}


def reset_provider(name):
    _registry.pop(name, None)


def get_provider(name):
    if name not in _registry:
        classes = {"OpenAI": OpenAIProvider, "Anthropic": AnthropicProvider, "Google": GoogleProvider}
        if name not in classes:
            raise ValueError(f"Unknown provider {name!r}; expected one of {PROVIDERS}")
        _registry[name] = classes[name]()
    return _registry[name]
