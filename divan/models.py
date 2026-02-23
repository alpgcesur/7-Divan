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


def create_model(model_spec: str, settings: DivanSettings, max_tokens: int) -> BaseChatModel:
    """Create a chat model from a 'provider:model_name' spec.

    Uses langchain's init_chat_model which handles provider routing automatically.
    Supported providers: anthropic, openai, google_genai
    """
    _set_api_keys(settings)

    if ":" in model_spec:
        provider, model_name = model_spec.split(":", 1)
    else:
        model_name = model_spec
        provider = None

    return init_chat_model(model_name, model_provider=provider, max_tokens=max_tokens)


def create_advisor_model(settings: DivanSettings) -> BaseChatModel:
    """Create the model used for advisor deliberations."""
    return create_model(settings.advisor_model, settings, settings.max_tokens)


def create_synthesis_model(settings: DivanSettings) -> BaseChatModel:
    """Create the model used for Bas Vezir synthesis."""
    return create_model(settings.synthesis_model, settings, settings.synthesis_max_tokens)
