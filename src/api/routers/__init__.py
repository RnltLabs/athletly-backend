"""FastAPI routers for the Athletly API."""

from src.api.routers.chat import router as chat_router
from src.api.routers.onboarding import router as onboarding_router

__all__ = ["chat_router", "onboarding_router"]
