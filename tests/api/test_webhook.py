"""Tests for POST /webhook/activity endpoint.

Covers:
- Valid HMAC signature + fresh timestamp → 200
- Invalid HMAC signature → 401
- Timestamp older than 5 minutes → 400
- Missing X-Webhook-Signature header → 401
- Missing X-Webhook-Timestamp header → 400

Supabase (_store_activity) is mocked so no real DB calls are made.
The webhook secret is injected via patched settings.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.routers.webhook import router

# Build a minimal FastAPI app with only the webhook router
from fastapi import FastAPI

_app = FastAPI()
_app.include_router(router, prefix="/webhook")

_TEST_SECRET = "super-secret-webhook-key"

_VALID_PAYLOAD: dict[str, Any] = {
    "user_id": "test-user-123",
    "activity_type": "run",
    "data": {"distance_m": 10000, "duration_s": 3600},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sign(body: bytes, secret: str = _TEST_SECRET) -> str:
    """Compute HMAC-SHA256 hex digest for the given body."""
    return hmac.new(
        secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256,
    ).hexdigest()


def _fresh_timestamp() -> str:
    """Return current Unix time as a string (within the 5-min window)."""
    return str(int(time.time()))


def _old_timestamp() -> str:
    """Return a Unix timestamp 10 minutes in the past (outside the window)."""
    return str(int(time.time()) - 601)


def _make_mock_settings(secret: str = _TEST_SECRET):
    mock = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    mock.webhook_secret = secret
    return mock


def _post_webhook(
    client: TestClient,
    payload: dict | None = None,
    signature: str | None = None,
    timestamp: str | None = None,
    body_bytes: bytes | None = None,
) -> Any:
    """Helper to POST to /webhook/activity with the given headers."""
    if payload is None:
        payload = _VALID_PAYLOAD
    if body_bytes is None:
        body_bytes = json.dumps(payload).encode()
    if timestamp is None:
        timestamp = _fresh_timestamp()
    if signature is None:
        signature = _sign(body_bytes)

    headers: dict[str, str] = {}
    if signature is not None:
        headers["x-webhook-signature"] = signature
    if timestamp is not None:
        headers["x-webhook-timestamp"] = timestamp

    return client.post(
        "/webhook/activity",
        content=body_bytes,
        headers={"Content-Type": "application/json", **headers},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient with Supabase storage mocked out."""
    mock_settings = _make_mock_settings()

    with (
        patch("src.api.routers.webhook.get_settings", return_value=mock_settings),
        patch(
            "src.api.routers.webhook._store_activity",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        with TestClient(_app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidWebhook:
    def test_valid_webhook_returns_200(self, client: TestClient):
        """Correct signature + fresh timestamp → 200 with status=ok."""
        resp = _post_webhook(client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_valid_webhook_stored_true(self, client: TestClient):
        """stored field reflects the return value of _store_activity (True)."""
        resp = _post_webhook(client)
        assert resp.status_code == 200
        assert resp.json()["stored"] is True


# ---------------------------------------------------------------------------
# Signature validation
# ---------------------------------------------------------------------------


class TestSignatureValidation:
    def test_invalid_signature_returns_401(self, client: TestClient):
        """Wrong HMAC digest → 401."""
        body_bytes = json.dumps(_VALID_PAYLOAD).encode()
        resp = _post_webhook(
            client,
            body_bytes=body_bytes,
            signature="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef0000000000000000000000",
        )
        assert resp.status_code == 401

    def test_missing_signature_header_returns_401(self, client: TestClient):
        """No X-Webhook-Signature header → 401."""
        body_bytes = json.dumps(_VALID_PAYLOAD).encode()
        timestamp = _fresh_timestamp()

        resp = client.post(
            "/webhook/activity",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "x-webhook-timestamp": timestamp,
                # deliberately omit x-webhook-signature
            },
        )
        assert resp.status_code == 401

    def test_tampered_body_rejected(self, client: TestClient):
        """Signing one body but sending a different body → 401."""
        original = json.dumps(_VALID_PAYLOAD).encode()
        sig = _sign(original)

        tampered_payload = dict(_VALID_PAYLOAD, activity_type="hack")
        tampered_body = json.dumps(tampered_payload).encode()

        resp = _post_webhook(client, body_bytes=tampered_body, signature=sig)
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------


class TestTimestampValidation:
    def test_old_timestamp_rejected(self, client: TestClient):
        """Timestamp older than 5 minutes → 400."""
        old_ts = _old_timestamp()
        body_bytes = json.dumps(_VALID_PAYLOAD).encode()
        sig = _sign(body_bytes)

        resp = _post_webhook(client, body_bytes=body_bytes, signature=sig, timestamp=old_ts)
        assert resp.status_code == 400

    def test_missing_timestamp_header_returns_400(self, client: TestClient):
        """No X-Webhook-Timestamp header → 400."""
        body_bytes = json.dumps(_VALID_PAYLOAD).encode()
        sig = _sign(body_bytes)

        resp = client.post(
            "/webhook/activity",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "x-webhook-signature": sig,
                # deliberately omit x-webhook-timestamp
            },
        )
        assert resp.status_code == 400

    def test_non_numeric_timestamp_rejected(self, client: TestClient):
        """A non-numeric timestamp string → 400."""
        body_bytes = json.dumps(_VALID_PAYLOAD).encode()
        sig = _sign(body_bytes)

        resp = _post_webhook(client, body_bytes=body_bytes, signature=sig, timestamp="not-a-number")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------------


class TestPayloadValidation:
    def test_empty_user_id_rejected(self, client: TestClient):
        """user_id that is blank/whitespace → 422 (Pydantic validation)."""
        bad_payload = dict(_VALID_PAYLOAD, user_id="   ")
        body_bytes = json.dumps(bad_payload).encode()
        sig = _sign(body_bytes)

        resp = _post_webhook(client, payload=bad_payload, body_bytes=body_bytes, signature=sig)
        assert resp.status_code == 422

    def test_empty_activity_type_rejected(self, client: TestClient):
        """activity_type that is blank → 422."""
        bad_payload = dict(_VALID_PAYLOAD, activity_type=" ")
        body_bytes = json.dumps(bad_payload).encode()
        sig = _sign(body_bytes)

        resp = _post_webhook(client, payload=bad_payload, body_bytes=body_bytes, signature=sig)
        assert resp.status_code == 422
