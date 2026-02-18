"""Supabase client singleton for AgenticSports.

Usage::

    from src.db.client import get_supabase

    client = get_supabase()
    result = client.table("profiles").select("*").execute()
"""

from functools import lru_cache

from supabase import Client, create_client

from src.config import get_settings


@lru_cache
def get_supabase() -> Client:
    """Return a cached singleton Supabase client.

    Uses the SERVICE_ROLE_KEY when available (bypasses RLS for backend
    operations), falling back to the ANON_KEY for client-level access.
    """
    settings = get_settings()
    key = settings.supabase_service_role_key or settings.supabase_anon_key
    if not key:
        raise RuntimeError(
            "No Supabase key configured. "
            "Set SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY in your .env file."
        )
    return create_client(settings.supabase_url, key)
