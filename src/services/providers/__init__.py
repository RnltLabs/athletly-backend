"""Provider abstraction package — see base.py for the contract."""

from src.services.providers.base import Provider, ProviderCapabilities
from src.services.providers.garmin import GarminProvider
from src.services.providers.registry import get_provider, list_providers
from src.services.providers.strava import StravaProvider

__all__ = [
    "Provider",
    "ProviderCapabilities",
    "GarminProvider",
    "StravaProvider",
    "get_provider",
    "list_providers",
]
