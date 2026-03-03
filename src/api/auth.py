"""JWT authentication for the Athletly FastAPI backend.

Supabase issues HS256 JWTs signed with the project's JWT secret.
Every protected endpoint depends on `get_current_user`, which validates
the token and returns the decoded payload.
"""

import logging
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import get_settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


def verify_jwt(token: str) -> dict:
    """Decode and validate a Supabase-issued JWT.

    Raises HTTPException(401) for any validation failure so callers
    never have to handle raw PyJWT exceptions.
    """
    secret = get_settings().supabase_jwt_secret
    if not secret:
        logger.error("supabase_jwt_secret is not configured")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication not configured",
        )

    try:
        payload: dict = jwt.decode(
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
