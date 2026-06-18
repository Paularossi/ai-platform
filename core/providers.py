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


# ---------------------------------------------------------------------------
# Supported providers and their model menus
# ---------------------------------------------------------------------------

PROVIDERS: list[str] = ["OpenAI", "Anthropic"]

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
}

# Environment variable name that holds the API key for each provider
API_KEY_ENV_VARS: dict[str, str] = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
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


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    _client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            OpenAIProvider._client = OpenAI()
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


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    _client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            AnthropicProvider._client = anthropic.Anthropic()
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


# ---------------------------------------------------------------------------
# Registry — one singleton per provider
# ---------------------------------------------------------------------------

_registry: dict[str, LLMProvider] = {}


def get_provider(name: str) -> LLMProvider:
    """Return the cached provider instance for the given name."""
    if name not in _registry:
        if name == "OpenAI":
            _registry[name] = OpenAIProvider()
        elif name == "Anthropic":
            _registry[name] = AnthropicProvider()
        else:
            raise ValueError(
                f"Unknown provider '{name}'. "
                f"Supported: {PROVIDERS}"
            )
    return _registry[name]