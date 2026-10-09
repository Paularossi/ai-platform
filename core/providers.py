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

import re


def uses_default_temperature(provider: str, model: str) -> bool:
    """Models whose sampling controls are restricted or best left at defaults."""
    if provider == "OpenAI":
        return bool(re.match(r"(?:gpt-[5-9]|o[1-9])", model))
    if provider == "Anthropic":
        return bool(re.match(r"claude-(?:sonnet|opus|haiku)-[5-9]", model)
                    or model.startswith(("claude-opus-4-7", "claude-opus-4-8")))
    return provider == "Google" and model.startswith("gemini-3")


def openai_options(model: str, max_tokens: int, temperature: float) -> dict:
    options = {"max_completion_tokens": max_tokens}
    if uses_default_temperature("OpenAI", model):
        options["reasoning_effort"] = "none" if model.startswith("gpt-6-luna") else "low"
    else:
        options["temperature"] = temperature
    return options


def anthropic_options(model: str, max_tokens: int, temperature: float) -> dict:
    options = {"max_tokens": max_tokens}
    if not uses_default_temperature("Anthropic", model):
        options["temperature"] = temperature
    return options


# ---------------------------------------------------------------------------
# Supported providers and their model menus
# ---------------------------------------------------------------------------

PROVIDERS: list[str] = ["OpenAI", "Anthropic", "Google"]

PROVIDER_MODELS: dict[str, list[str]] = {
    "OpenAI": [
        "gpt-6-luna", # default, $0.10 input / $0.50 output
        "gpt-4.1-mini", # $0.40 input / $1.60 output
        "gpt-6.1-sol", # $2.00 input / $10.00 output
        "gpt-4o", # $2.50 input / $10.00 output
    ],
    "Anthropic": [ # opus and fable don't seem relevant
        "claude-sonnet-5-5", # default, $2.00 input / $10.00 output
        "claude-haiku-4-5", # $1.00 input / $5.00 output
    ],
    "Google": [ # api calls seem to be a bit slower
        "gemini-3.8-flash", # default, $0.75 input / $3.75 output
        "gemini-3.5-flash-lite", # $0.30 input / $2.50 output
        #"gemini-3.1-pro-preview",
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
    """Abstract base. Subclasses must implement complete() and stream()."""

    def __init__(self):
        # Usage from the most recent complete()/stream() call — {input_tokens,
        # output_tokens, total_tokens}, all None until a call has completed.
        # Streaming SDKs surface usage differently per provider (see each
        # subclass's stream()), so this is populated at different points.
        self.last_usage: dict[str, int | None] = {
            "input_tokens": None, "output_tokens": None, "total_tokens": None,
        }

    def get_last_usage(self) -> dict[str, int | None]:
        return dict(self.last_usage)

    def reset_usage(self) -> None:
        self.last_usage = dict.fromkeys(("input_tokens", "output_tokens", "total_tokens"))

    def complete(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError

    def stream(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ):
        """Yield the response as it's generated, chunk by chunk (str pieces)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def complete(self, model, system_prompt, user_message,
                 max_tokens=1500, temperature=0.0) -> str:
        self.reset_usage()
        response = self._get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            **openai_options(model, max_tokens, temperature),
        )
        if response.usage:
            self.last_usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return response.choices[0].message.content or ""

    def stream(self, model, system_prompt, user_message,
               max_tokens=1500, temperature=0.0):
        self.reset_usage()
        response = self._get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            **openai_options(model, max_tokens, temperature),
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in response:
            if chunk.usage:
                self.last_usage = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, model, system_prompt, user_message,
                 max_tokens=1500, temperature=0.0) -> str:
        self.reset_usage()
        response = self._get_client().messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            **anthropic_options(model, max_tokens, temperature),
        )
        if response.usage:
            self.last_usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
        return "".join(block.text for block in response.content if block.type == "text")

    def stream(self, model, system_prompt, user_message,
               max_tokens=1500, temperature=0.0):
        self.reset_usage()
        with self._get_client().messages.stream(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            **anthropic_options(model, max_tokens, temperature),
        ) as stream:
            for text in stream.text_stream:
                yield text
            # Usage isn't available until the stream is fully drained —
            # get_final_message() blocks until then and carries it.
            final = stream.get_final_message()
            if final.usage:
                self.last_usage = {
                    "input_tokens": final.usage.input_tokens,
                    "output_tokens": final.usage.output_tokens,
                    "total_tokens": final.usage.input_tokens + final.usage.output_tokens,
                }


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

class GoogleProvider(LLMProvider):
    def __init__(self):
        super().__init__()
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

        self.reset_usage()

        config_kwargs: dict = {
            "system_instruction": system_prompt,
            "max_output_tokens": max_tokens,
        }
        if not uses_default_temperature("Google", model):
            config_kwargs["temperature"] = temperature
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
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.last_usage = {
                "input_tokens": usage.prompt_token_count,
                "output_tokens": usage.candidates_token_count,
                "total_tokens": usage.total_token_count,
            }

        try:
            text = "".join(part.text for part in response.candidates[0].content.parts
                           if part.text and not getattr(part, "thought", False))
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

    def stream(self, model: str, system_prompt: str, user_message: str,
               max_tokens: int = 1500, temperature: float = 0.0):
        """
        Stream a Gemini response chunk by chunk. Only used for free-text
        contribution calls (never json_mode — streaming raw JSON pieces to
        the UI would just show broken fragments), so no json_mode param.
        """
        from google.genai import types

        self.reset_usage()
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=None if uses_default_temperature("Google", model) else temperature,
            max_output_tokens=max_tokens,
        )
        try:
            for chunk in self._get_client().models.generate_content_stream(
                model=model,
                contents=user_message,
                config=config,
            ):
                # usage_metadata is present on every chunk but only reaches
                # its final totals on the last one — later chunks just
                # overwrite last_usage until it settles there.
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    self.last_usage = {
                        "input_tokens": usage.prompt_token_count,
                        "output_tokens": usage.candidates_token_count,
                        "total_tokens": usage.total_token_count,
                    }
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            yield f"[ERROR: {exc}]"


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
