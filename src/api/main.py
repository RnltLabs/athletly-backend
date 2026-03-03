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

    app = FastAPI(
        title="Athletly",
        version=VERSION,
        lifespan=_lifespan,
        # Disable default /docs and /redoc in production by checking env if needed
        docs_url="/docs",
        redoc_url="/redoc",
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

    # Chat router will live at /chat (imported lazily to avoid circular deps
    # during early boot when tools/agent are still initialising).
    try:
        from src.api.routers.chat import router as chat_router
        app.include_router(chat_router, prefix="/chat")
    except ImportError:
        logger.warning(
            "Chat router not yet implemented — /chat will return 404 until created"
        )

    # Webhook router — activity events from external providers.
    from src.api.routers.webhook import router as webhook_router
    app.include_router(webhook_router, prefix="/webhook", tags=["webhooks"])

    # -- Health endpoint ------------------------------------------------------

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        return {"status": "ok", "version": VERSION}

    return app


# -- Module-level app for uvicorn --------------------------------------------

app = create_app()
