"""Configuration for Divan using pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DivanSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DIVAN_")

    # API keys (at least one required)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""

    # Model configuration (format: "provider:model_name")
    advisor_model: str = "openai:gpt-5-mini-2025-08-07"
    synthesis_model: str = "openai:gpt-5.1-2025-11-13"

    # Token limits
    max_tokens: int = 1500
    synthesis_max_tokens: int = 2000

    # Paths
    personas_dir: str = str(Path(__file__).parent.parent / "personas")
    templates_dir: str = str(Path(__file__).parent.parent / "templates")


def get_settings(**overrides: str) -> DivanSettings:
    """Load settings with optional overrides."""
    return DivanSettings(**{k: v for k, v in overrides.items() if v is not None})
