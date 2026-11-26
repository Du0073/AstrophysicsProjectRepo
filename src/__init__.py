"""SDSS Galaxy Morphology & Scaling Relations Atlas package."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ProjectConfig",
    "SDSSCatalogFetcher",
    "GalaxyDatabase",
    "SkyServerRGBDownloader",
    "MultiBandFITSProcessor",
]


def __getattr__(name: str) -> Any:
    """Lazily expose top-level classes to keep ``python -m src.<module>`` warning-free."""
    if name == "ProjectConfig":
        from .config import ProjectConfig

        return ProjectConfig
    if name == "SDSSCatalogFetcher":
        from .acquisition import SDSSCatalogFetcher

        return SDSSCatalogFetcher
    if name == "GalaxyDatabase":
        from .database import GalaxyDatabase

        return GalaxyDatabase
    if name == "SkyServerRGBDownloader":
        from .imaging import SkyServerRGBDownloader

        return SkyServerRGBDownloader
    if name == "MultiBandFITSProcessor":
        from .imaging import MultiBandFITSProcessor

        return MultiBandFITSProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
