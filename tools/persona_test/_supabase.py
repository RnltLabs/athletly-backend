"""Supabase helpers shared by seed.py, reset.py, and chat.py.

Wraps the auth-admin and JWT-signin flows for persona test users so the
CLI scripts can stay focused on their respective concerns.

Safety:
- Every helper uses the service-role key and refuses to operate on users
  whose ``user_metadata.is_test`` is not ``True``. This prevents a stray
  reset from nuking a real production user even if the email collides.
- Passwords are generated once per persona slug and persisted to
  ``.env.persona_test`` (gitignored) so subsequent runs reuse the same
  credentials.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from supabase import Client, create_client

from src.config import get_settings

logger = logging.getLogger(__name__)

# Storage paths for ephemeral persona-test state. All gitignored.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PERSONA_TEST = _REPO_ROOT / ".env.persona_test"
_JWT_CACHE_DIR = _REPO_ROOT / ".persona_test_jwt"
_SESSION_CACHE_DIR = _REPO_ROOT / ".persona_test_session"

# 5 minutes safety buffer before considering a JWT expired.
_JWT_EXPIRY_BUFFER_S = 300


@dataclass(frozen=True)
class PersonaAuth:
    """Resolved auth state for a persona test user."""

    user_id: str
    email: str
    password: str
    access_token: str
    expires_at: int  # unix epoch seconds


# ---------------------------------------------------------------------------
# Service-role client
# ---------------------------------------------------------------------------


def get_service_client() -> Client:
    """Return a Supabase client bound to the service-role key.

    Exits with code 2 if the service-role key is not configured. Persona
    seeding cannot proceed without it.
    """
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        sys.stderr.write(
            "ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must both be set.\n"
            "Add them to your .env file. The persona test framework requires "
            "service-role access to create test users via the auth admin API.\n",
        )
        sys.exit(2)
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def get_anon_client() -> Client:
    """Return a Supabase client bound to the anon key (used for sign-in)."""
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_anon_key:
        sys.stderr.write(
            "ERROR: SUPABASE_URL and SUPABASE_ANON_KEY must both be set.\n"
            "The chat CLI signs in via the anon endpoint to obtain a real JWT "
            "that flows through the same /chat auth path as a production user.\n",
        )
        sys.exit(2)
    return create_client(settings.supabase_url, settings.supabase_anon_key)


# ---------------------------------------------------------------------------
# Persona test password store (.env.persona_test)
# ---------------------------------------------------------------------------


def get_or_create_password(slug: str) -> str:
    """Look up or mint a persistent password for the given persona slug.

    Persisted to ``.env.persona_test`` as ``PERSONA_TEST_<SLUG>_PASSWORD``.
    """
    var_name = f"PERSONA_TEST_{slug.upper()}_PASSWORD"
    existing = _read_env_var(_ENV_PERSONA_TEST, var_name)
    if existing:
        return existing
    new_pw = "persona-" + secrets.token_urlsafe(24)
    _append_env_var(_ENV_PERSONA_TEST, var_name, new_pw)
    return new_pw


def _read_env_var(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


def _append_env_var(path: Path, key: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ""
    if not path.exists():
        header = (
            "# Persona test credentials. AUTO-GENERATED. Do not commit.\n"
            "# Used by tools/persona_test/{seed,reset,chat}.py.\n"
        )
    with path.open("a", encoding="utf-8") as fh:
        if header:
            fh.write(header)
        fh.write(f"{key}={value}\n")


# ---------------------------------------------------------------------------
# Auth user lookup / create / verify
# ---------------------------------------------------------------------------


def find_user_by_email(client: Client, email: str) -> dict | None:
    """Return the auth user dict matching *email*, or None if absent.

    Uses the admin list_users endpoint and filters in Python because the
    SDK does not expose a direct lookup-by-email helper.
    """
    # list_users paginates; persona test users are few, so the first page
    # is sufficient unless the project has thousands of test users.
    try:
        page = client.auth.admin.list_users()
    except Exception as exc:  # noqa: BLE001 -- surface the underlying error
        sys.stderr.write(f"ERROR: auth.admin.list_users failed: {exc}\n")
        sys.exit(2)

    for user in page or []:
        if getattr(user, "email", None) == email:
            return _user_to_dict(user)
    return None


def ensure_test_user(client: Client, *, slug: str, email: str, password: str) -> dict:
    """Find or create the persona test user. Returns the user dict.

    Always ensures ``user_metadata.is_test=True`` and ``persona=<slug>`` so
    other test users in the same project can be filtered safely.
    """
    user = find_user_by_email(client, email)
    metadata = {"is_test": True, "persona": slug}
    if user is None:
        try:
            created = client.auth.admin.create_user(
                {
                    "email": email,
                    "password": password,
                    "email_confirm": True,
                    "user_metadata": metadata,
                },
            )
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"ERROR: auth.admin.create_user failed: {exc}\n")
            sys.exit(2)
        return _user_to_dict(getattr(created, "user", created))

    # Existing user: refresh metadata so the is_test marker is enforced.
    try:
        client.auth.admin.update_user_by_id(
            user["id"],
            {"user_metadata": {**(user.get("user_metadata") or {}), **metadata}},
        )
    except Exception:
        logger.exception("Failed to refresh user_metadata for %s", email)
    user.setdefault("user_metadata", {}).update(metadata)
    return user


def assert_is_test_user(user: dict) -> None:
    """Refuse to operate on a user that lacks the is_test marker.

    This is the last line of defence against a stray reset clobbering a
    real production user that happens to share an email with a persona.
    """
    meta = (user.get("user_metadata") or {})
    if not meta.get("is_test"):
        sys.stderr.write(
            f"REFUSED: user {user.get('email')!r} is not marked is_test=True. "
            "Persona test scripts will not touch this user.\n",
        )
        sys.exit(3)


def delete_test_user(client: Client, user_id: str) -> None:
    """Hard-delete the auth user (cascades to all user-scoped tables)."""
    try:
        client.auth.admin.delete_user(user_id)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"WARNING: auth.admin.delete_user failed: {exc}\n")


# ---------------------------------------------------------------------------
# JWT acquisition (sign-in with password)
# ---------------------------------------------------------------------------


def sign_in_persona(slug: str) -> PersonaAuth:
    """Return a fresh PersonaAuth, signing in via the anon endpoint.

    The returned JWT flows through the same verify_jwt path as a real
    production user, so the test exercises the full auth stack.
    """
    from tools.persona_test._persona import load_persona

    persona = load_persona(slug)
    password = get_or_create_password(slug)

    cached = _read_cached_jwt(slug)
    if cached and cached.expires_at - _JWT_EXPIRY_BUFFER_S > int(time.time()):
        return cached

    anon = get_anon_client()
    try:
        result = anon.auth.sign_in_with_password({
            "email": persona.email,
            "password": password,
        })
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"ERROR: sign_in_with_password failed for {persona.email}: {exc}\n"
            "Run `python tools/persona_test/seed.py {slug}` first to ensure "
            "the auth user exists with the right password.\n",
        )
        sys.exit(2)

    session = getattr(result, "session", None)
    user = getattr(result, "user", None)
    if not session or not user:
        sys.stderr.write("ERROR: sign-in returned no session/user.\n")
        sys.exit(2)

    auth = PersonaAuth(
        user_id=user.id,
        email=persona.email,
        password=password,
        access_token=session.access_token,
        expires_at=int(session.expires_at or 0),
    )
    _write_cached_jwt(slug, auth)
    return auth


def _read_cached_jwt(slug: str) -> PersonaAuth | None:
    path = _JWT_CACHE_DIR / f"{slug}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PersonaAuth(**data)
    except Exception:
        return None


def _write_cached_jwt(slug: str, auth: PersonaAuth) -> None:
    _JWT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _JWT_CACHE_DIR / f"{slug}.json"
    path.write_text(
        json.dumps(
            {
                "user_id": auth.user_id,
                "email": auth.email,
                "password": auth.password,
                "access_token": auth.access_token,
                "expires_at": auth.expires_at,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def clear_cached_jwt(slug: str) -> None:
    """Remove the cached JWT for a persona slug (after reset)."""
    path = _JWT_CACHE_DIR / f"{slug}.json"
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Chat session id cache
# ---------------------------------------------------------------------------


def read_session_id(slug: str) -> str | None:
    """Return the persisted chat session id for the persona, or None."""
    path = _SESSION_CACHE_DIR / f"{slug}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("session_id")
    except Exception:
        return None


def write_session_id(slug: str, session_id: str) -> None:
    """Persist the chat session id so the next chat call resumes it."""
    _SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _SESSION_CACHE_DIR / f"{slug}.json"
    path.write_text(
        json.dumps({"session_id": session_id}, indent=2),
        encoding="utf-8",
    )


def clear_session_id(slug: str) -> None:
    """Remove the persisted chat session id (used by reset)."""
    path = _SESSION_CACHE_DIR / f"{slug}.json"
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _user_to_dict(user) -> dict:
    """Normalize a gotrue User object into a plain dict."""
    if isinstance(user, dict):
        return user
    out: dict = {}
    for key in (
        "id",
        "email",
        "user_metadata",
        "app_metadata",
        "created_at",
        "updated_at",
    ):
        value = getattr(user, key, None)
        if value is not None:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Test helpers exposed for unit tests
# ---------------------------------------------------------------------------


def _set_paths_for_testing(*, env_path: Path, jwt_dir: Path, session_dir: Path) -> None:
    """Override on-disk cache paths in tests. Internal use only."""
    global _ENV_PERSONA_TEST, _JWT_CACHE_DIR, _SESSION_CACHE_DIR
    _ENV_PERSONA_TEST = env_path
    _JWT_CACHE_DIR = jwt_dir
    _SESSION_CACHE_DIR = session_dir
