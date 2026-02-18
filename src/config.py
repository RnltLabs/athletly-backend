"""Centralized configuration for AgenticSports using Pydantic Settings v2.

All environment variables are loaded from .env and validated at startup.
Use ``get_settings()`` to obtain the cached singleton instance.

Example::

    from src.config import get_settings

    settings = get_settings()
    print(settings.gemini_api_key)
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings populated from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",  # no prefix -- use raw env var names
        extra="ignore",
    )

    # -- LLM API keys --------------------------------------------------------
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    agenticsports_model: str = "gemini/gemini-2.5-flash"
    agent_temperature: float = 0.7

    # -- Supabase -------------------------------------------------------------
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""

    # -- MCP / Search ---------------------------------------------------------
    brave_search_api_key: str = ""

    # -- Agent config ---------------------------------------------------------
    max_tool_rounds: int = 25
    max_consecutive_errors: int = 3
    compression_threshold: int = 40
    compression_keep_rounds: int = 4

    # -- User / multi-tenancy -------------------------------------------------
    agenticsports_user_id: str = ""  # Set for Supabase mode; leave empty for file-based

    # -- Data (legacy file-based, kept for gradual migration) -----------------
    data_dir: str = "data"

    @property
    def use_supabase(self) -> bool:
        """Auto-detect: use Supabase when URL + key + user_id are configured."""
        return bool(
            self.supabase_url
            and (self.supabase_service_role_key or self.supabase_anon_key)
            and self.agenticsports_user_id
        )


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
