"""Provider registry — single source of truth for available data providers.

Use ``get_provider("garmin")`` to obtain a provider instance, or
``list_providers()`` to iterate over all registered providers (e.g. for
``athctl provider list``).

Each provider is a singleton — its state lives in the database, not on the
instance, so reusing one object across calls is safe.
"""

from __future__ import annotations

from src.services.providers.base import Provider
from src.services.providers.garmin import GarminProvider
from src.services.providers.strava import StravaProvider

_REGISTRY: dict[str, Provider] = {
    GarminProvider.name: GarminProvider(),
    StravaProvider.name: StravaProvider(),
}


def get_provider(name: str) -> Provider:
    """Return the provider singleton for *name* or raise ``ValueError``."""
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown provider {name!r}; known: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_providers() -> list[Provider]:
    """Return all registered provider singletons."""
    return list(_REGISTRY.values())
