"""JWT authentication for the Athletly FastAPI backend.

Supabase may issue JWTs signed with either HS256 (legacy) or ES256 (new JWKS).
Every protected endpoint depends on `get_current_user`, which validates
the token and returns the decoded payload.
"""

import logging
from functools import lru_cache
from typing import Annotated

import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import get_settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


@lru_cache
def _get_jwks_client() -> PyJWKClient:
    """Cached JWKS client for Supabase ES256 token verification."""
    url = get_settings().supabase_url
    return PyJWKClient(f"{url}/auth/v1/.well-known/jwks.json")


def verify_jwt(token: str) -> dict:
    """Decode and validate a Supabase-issued JWT.

    Supports both ES256 (JWKS) and HS256 (legacy secret).
    Raises HTTPException(401) for any validation failure.
    """
    settings = get_settings()

    # Peek at the token header to determine algorithm
    try:
        header = jwt.get_unverified_header(token)
    except jwt.DecodeError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    try:
        if header.get("alg") == "ES256":
            # New Supabase JWKS-based verification
            signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
            payload: dict = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256"],
                audience="authenticated",
            )
        else:
            # Legacy HS256 verification
            secret = settings.supabase_jwt_secret
            if not secret:
                logger.error("supabase_jwt_secret is not configured")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication not configured",
                )
            payload = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience="authenticated",
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError as exc:
        logger.debug("JWT validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    # Guarantee minimum required claims are present
    if not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claims",
        )

    return payload


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> dict:
    """FastAPI dependency — extracts the Bearer token and returns the user dict.

    The returned dict always contains at minimum:
        - sub   (user_id, str)
        - email (str | None)
        - role  (str)
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_jwt(credentials.credentials)

    return {
        "sub": payload["sub"],
        "email": payload.get("email"),
        "role": payload.get("role", "authenticated"),
    }


def get_user_id(
    current_user: Annotated[dict, Depends(get_current_user)],
) -> str:
    """Extract user_id (UUID) from JWT — convenience dependency."""
    return current_user["sub"]
