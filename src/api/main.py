"""FastAPI application factory for the Athletly backend.

Entry point for uvicorn:
    uvicorn src.api.main:app --reload

`create_app()` wires together:
    - CORS middleware
    - slowapi rate limiting
    - Chat router (mounted at /chat)
    - Webhook router (mounted at /webhook)
    - HeartbeatService (started/stopped with app lifespan)
    - Lifespan startup/shutdown logging
    - Health endpoint
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from src.api.rate_limiter import limiter
from src.config import get_settings

logger = logging.getLogger(__name__)

VERSION = "0.1.0"


# -- Lifespan ----------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    from src.services.heartbeat import HeartbeatService

    settings = get_settings()
    logger.info(
        "Athletly backend starting — model=%s redis=%s",
        settings.agenticsports_model,
        settings.redis_url,
    )

    heartbeat = HeartbeatService(interval_seconds=settings.heartbeat_interval_seconds)
    await heartbeat.start()

    try:
        yield
    finally:
        await heartbeat.stop()
        logger.info("Athletly backend shutting down")


# -- Exception handlers ------------------------------------------------------


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limit_exceeded", "detail": str(exc.detail)},
    )


# -- Factory -----------------------------------------------------------------


def create_app() -> FastAPI:
    settings = get_settings()

    _docs_url = "/docs" if settings.debug else None
    _redoc_url = "/redoc" if settings.debug else None

    app = FastAPI(
        title="Athletly",
        version=VERSION,
        lifespan=_lifespan,
        docs_url=_docs_url,
        redoc_url=_redoc_url,
    )

    # -- Middleware -----------------------------------------------------------

    origins = settings.cors_origin_list
    # cors_origin_list returns ["*"] when cors_origins == "*"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # slowapi state must be attached before any route handlers are called
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]

    # -- Routers --------------------------------------------------------------

    # Chat router: imported lazily to avoid circular deps during early boot.
    # In debug mode a missing module is silently skipped (gradual rollout).
    # In production any ImportError propagates immediately so misconfigured
    # deployments fail loudly rather than serving silent 404s.
    if settings.debug:
        try:
            from src.api.routers.chat import router as chat_router
            app.include_router(chat_router, prefix="/chat")
        except ImportError:
            logger.warning(
                "Chat router not yet implemented — /chat will return 404 until created"
            )
    else:
        from src.api.routers.chat import router as chat_router
        app.include_router(chat_router, prefix="/chat")

    # Webhook router — activity events from external providers.
    from src.api.routers.webhook import router as webhook_router
    app.include_router(webhook_router, prefix="/webhook", tags=["webhooks"])

    # Garmin sync router — connect, sync, disconnect Garmin accounts.
    from src.api.routers.garmin import router as garmin_router
    app.include_router(garmin_router, prefix="/garmin", tags=["garmin"])

    # Onboarding router — parse-voice (public) + setup (auth required).
    from src.api.routers.onboarding import router as onboarding_router
    app.include_router(onboarding_router, prefix="/api/onboarding", tags=["onboarding"])

    # -- Health endpoint ------------------------------------------------------

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        return {"status": "ok", "version": VERSION}

    return app


# -- Module-level app for uvicorn --------------------------------------------

app = create_app()
