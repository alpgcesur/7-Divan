"""Model factory using langchain's init_chat_model for provider-agnostic LLM creation."""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from divan.config import DivanSettings


def _set_api_keys(settings: DivanSettings) -> None:
    """Push API keys from settings into environment variables for langchain providers."""
    if settings.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.google_api_key:
        os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)


def _is_openai_reasoning_model(model_name: str) -> bool:
    """Check if an OpenAI model is a reasoning model that uses reasoning tokens.

    Reasoning models (o-series, gpt-5 series) use internal chain-of-thought
    that consumes tokens. They need special handling: max_completion_tokens
    instead of max_tokens, and reasoning effort control.
    """
    name = model_name.lower()
    return any(name.startswith(prefix) for prefix in ("o1", "o3", "o4", "gpt-5"))


def _create_openai_reasoning_model(model_name: str, max_tokens: int) -> BaseChatModel:
    """Create an OpenAI reasoning model with proper configuration.

    Uses ChatOpenAI directly (not init_chat_model) because reasoning models
    need the `reasoning` dict parameter and `max_completion_tokens`, which
    init_chat_model doesn't pass through correctly.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        max_completion_tokens=max_tokens,
        reasoning={"effort": "low", "summary": "auto"},
    )


def create_model(model_spec: str, settings: DivanSettings, max_tokens: int) -> BaseChatModel:
    """Create a chat model from a 'provider:model_name' spec.

    Uses langchain's init_chat_model which handles provider routing automatically.
    Supported providers: anthropic, openai, google_genai

    For OpenAI reasoning models (o-series, gpt-5), uses ChatOpenAI directly
    with reasoning effort control and max_completion_tokens to prevent
    reasoning tokens from consuming the entire output budget.
    """
    _set_api_keys(settings)

    if ":" in model_spec:
        provider, model_name = model_spec.split(":", 1)
    else:
        model_name = model_spec
        provider = None

    if provider == "openai" and _is_openai_reasoning_model(model_name):
        return _create_openai_reasoning_model(model_name, max_tokens)

    return init_chat_model(model_name, model_provider=provider, max_tokens=max_tokens)


def create_advisor_model(settings: DivanSettings) -> BaseChatModel:
    """Create the model used for advisor deliberations."""
    return create_model(settings.advisor_model, settings, settings.max_tokens)


def create_synthesis_model(settings: DivanSettings) -> BaseChatModel:
    """Create the model used for Bas Vezir synthesis."""
    return create_model(settings.synthesis_model, settings, settings.synthesis_max_tokens)
