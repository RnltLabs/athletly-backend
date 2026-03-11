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
    daily_token_budget: int = 2_000_000  # Max tokens per user per day

    # -- User / multi-tenancy -------------------------------------------------
    agenticsports_user_id: str = ""  # Set for Supabase mode; leave empty for file-based

    # -- FastAPI / Server -----------------------------------------------------
    supabase_jwt_secret: str = ""  # HS256 secret for Supabase JWT verification
    redis_url: str = "redis://localhost:6379"
    debug: bool = False  # Set DEBUG=true in .env for local development
    cors_origins: str = "*"  # comma-separated origins or "*"
    webhook_secret: str = ""  # HMAC secret for activity webhooks

    # -- LLM fallback chain ---------------------------------------------------
    litellm_fallback_models: str = "gemini/gemini-2.5-flash,anthropic/claude-haiku-4-5-20251001"

    # -- Heartbeat / proactive ------------------------------------------------
    heartbeat_interval_seconds: int = 1800  # 30 minutes

    # -- Garmin sync ----------------------------------------------------------
    garmin_sync_cooldown_seconds: int = 900  # 15 min between syncs

    # -- Tool output budget (tokens) ------------------------------------------
    tool_output_budget_default: int = 2000

    # -- Product Recommendations / Affiliate -----------------------------------
    amazon_affiliate_tag: str = ""
    amazon_pa_api_access_key: str = ""
    amazon_pa_api_secret_key: str = ""
    awin_affiliate_id: str = ""
    awin_adidas_merchant_id: str = ""

    # -- Data (legacy file-based, kept for gradual migration) -----------------
    data_dir: str = "data"

    @property
    def fallback_models(self) -> list[str]:
        """Parse comma-separated fallback model string into a list."""
        return [m.strip() for m in self.litellm_fallback_models.split(",") if m.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def use_supabase(self) -> bool:
        """Auto-detect: use Supabase when URL + key are configured.

        In multi-tenant API mode the user_id comes from the JWT, not from
        an env var, so we no longer require agenticsports_user_id here.
        """
        return bool(
            self.supabase_url
            and (self.supabase_service_role_key or self.supabase_anon_key)
        )


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
