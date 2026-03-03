"""Tests for src/api/auth.py — JWT verification and get_current_user dependency.

Covers:
- valid JWT returns the correct payload fields
- expired JWT raises 401
- missing Authorization header raises 401
- invalid signature raises 401
- token with no `sub` claim raises 401
"""

import time
from unittest.mock import patch

import jwt as pyjwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.api.auth import get_current_user, verify_jwt
from src.config import Settings

JWT_SECRET = "test-auth-secret-xyz-minimum-32-bytes-long-for-hs256"
USER_ID = "auth-test-user-001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(secret: str = JWT_SECRET) -> Settings:
    return Settings(
        supabase_jwt_secret=secret,
        redis_url="redis://localhost:6379",
    )


def _sign(payload: dict, secret: str = JWT_SECRET) -> str:
    return pyjwt.encode(payload, secret, algorithm="HS256")


def _base_payload(offset: int = 3600) -> dict:
    now = int(time.time())
    return {
        "sub": USER_ID,
        "email": "user@test.com",
        "role": "authenticated",
        "aud": "authenticated",
        "iat": now,
        "exp": now + offset,
    }


# ---------------------------------------------------------------------------
# verify_jwt
# ---------------------------------------------------------------------------


class TestVerifyJwt:
    def test_valid_jwt_returns_payload(self) -> None:
        """verify_jwt decodes a valid token and returns sub, email, role."""
        token = _sign(_base_payload())

        with patch("src.api.auth.get_settings", return_value=_make_settings()):
            result = verify_jwt(token)

        assert result["sub"] == USER_ID
        assert result["email"] == "user@test.com"
        assert result["role"] == "authenticated"

    def test_expired_jwt_raises_401(self) -> None:
        """verify_jwt raises HTTPException(401) for tokens past their exp."""
        expired_payload = _base_payload(offset=-7200)  # exp 2 hours ago
        token = _sign(expired_payload)

        with patch("src.api.auth.get_settings", return_value=_make_settings()):
            with pytest.raises(HTTPException) as exc_info:
                verify_jwt(token)

        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_invalid_signature_raises_401(self) -> None:
        """verify_jwt raises 401 when the token is signed with a different secret."""
        token = _sign(_base_payload(), secret="wrong-secret-entirely-different-from-the-configured-one")

        with patch("src.api.auth.get_settings", return_value=_make_settings()):
            with pytest.raises(HTTPException) as exc_info:
                verify_jwt(token)

        assert exc_info.value.status_code == 401
        assert "invalid" in exc_info.value.detail.lower()

    def test_missing_sub_claim_raises_401(self) -> None:
        """verify_jwt raises 401 when the decoded payload has no `sub` field."""
        payload_no_sub = {
            "email": "nosub@test.com",
            "role": "authenticated",
            "aud": "authenticated",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
        token = _sign(payload_no_sub)

        with patch("src.api.auth.get_settings", return_value=_make_settings()):
            with pytest.raises(HTTPException) as exc_info:
                verify_jwt(token)

        assert exc_info.value.status_code == 401
        assert "missing" in exc_info.value.detail.lower() or "claims" in exc_info.value.detail.lower()

    def test_unconfigured_secret_raises_401(self) -> None:
        """verify_jwt raises 401 when supabase_jwt_secret is empty."""
        token = _sign(_base_payload())
        empty_settings = _make_settings(secret="")

        with patch("src.api.auth.get_settings", return_value=empty_settings):
            with pytest.raises(HTTPException) as exc_info:
                verify_jwt(token)

        assert exc_info.value.status_code == 401
        assert "not configured" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------


class TestGetCurrentUser:
    def test_missing_token_raises_401(self) -> None:
        """get_current_user raises 401 when credentials is None (no header)."""
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials=None)

        assert exc_info.value.status_code == 401
        assert "missing" in exc_info.value.detail.lower()

    def test_valid_credentials_returns_user_dict(self) -> None:
        """get_current_user returns a dict with sub, email, role."""
        token = _sign(_base_payload())
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        with patch("src.api.auth.get_settings", return_value=_make_settings()):
            result = get_current_user(credentials=credentials)

        assert result["sub"] == USER_ID
        assert result["email"] == "user@test.com"
        assert result["role"] == "authenticated"

    def test_role_defaults_to_authenticated(self) -> None:
        """get_current_user defaults role to 'authenticated' when not in payload."""
        payload = _base_payload()
        payload.pop("role")
        token = _sign(payload)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        with patch("src.api.auth.get_settings", return_value=_make_settings()):
            result = get_current_user(credentials=credentials)

        assert result["role"] == "authenticated"
