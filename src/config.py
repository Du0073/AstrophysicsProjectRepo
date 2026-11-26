"""Configuration, cosmology, SDSS DR17 selection cuts, and Phase 2 imaging settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy.cosmology import FlatLambdaCDM


def _default_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProjectConfig:
    """Central configuration for the SDSS Galaxy Morphology & Scaling Relations Atlas.

    All physical quantities derived under this configuration use a standard flat
    LambdaCDM cosmology (H0 = 70 km/s/Mpc, Om0 = 0.3) and SDSS DR17 Main Galaxy
    Sample volume-limited cuts.
    """

    project_root: Path = field(default_factory=_default_project_root)
    db_relative_path: Path = Path("data/galaxy_atlas.db")
    raw_catalog_relative_path: Path = Path("data/raw/sdss_catalog_raw.csv")
    figures_relative_dir: Path = Path("outputs/figures")

    # Phase 2 imaging directories (all under git-ignored data/raw and data/processed)
    rgb_cutouts_relative_dir: Path = Path("data/processed/rgb_cutouts")
    fits_raw_relative_dir: Path = Path("data/raw/fits")
    fits_cutouts_relative_dir: Path = Path("data/processed/fits_cutouts")
    lupton_rgb_relative_dir: Path = Path("data/processed/lupton_rgb")

    # Cosmology parameters (FlatLambdaCDM)
    h0: float = 70.0
    om0: float = 0.3

    # SDSS DR17 selection parameters
    data_release: int = 17
    default_top_n: int = 2500
    z_min: float = 0.02
    z_max: float = 0.10
    rmag_max: float = 17.77
    photo_type: int = 3
    spec_class: str = "GALAXY"
    z_warning: int = 0
    frac_dev_early_threshold: float = 0.5
    sdss_timeout_s: int = 120

    # Phase 2 imaging parameters
    cutout_size_px: int = 128
    pixel_scale_arcsec: float = 0.396
    rgb_max_workers: int = 6
    fits_showcase_count: int = 12
    lupton_q: float = 8.0
    lupton_stretch: float = 0.5
    image_timeout_s: int = 45

    @property
    def db_path(self) -> Path:
        """Absolute path to the normalized SQLite database."""
        return self.project_root / self.db_relative_path

    @property
    def raw_catalog_path(self) -> Path:
        """Absolute path to the raw SDSS CSV export."""
        return self.project_root / self.raw_catalog_relative_path

    @property
    def figures_dir(self) -> Path:
        """Absolute path to the diagnostic figures directory."""
        return self.project_root / self.figures_relative_dir

    @property
    def rgb_cutouts_dir(self) -> Path:
        """Absolute path to downloaded SkyServer RGB JPEG cutouts."""
        return self.project_root / self.rgb_cutouts_relative_dir

    @property
    def fits_raw_dir(self) -> Path:
        """Absolute path to raw downloaded SDSS multi-band FITS frames."""
        return self.project_root / self.fits_raw_relative_dir

    @property
    def fits_cutouts_dir(self) -> Path:
        """Absolute path to WCS-cropped and NaN-masked FITS cutouts."""
        return self.project_root / self.fits_cutouts_relative_dir

    @property
    def lupton_rgb_dir(self) -> Path:
        """Absolute path to Lupton arcsinh-stretched RGB composites from FITS."""
        return self.project_root / self.lupton_rgb_relative_dir

    @property
    def cosmology(self) -> FlatLambdaCDM:
        """Astropy FlatLambdaCDM cosmology instance."""
        return FlatLambdaCDM(H0=self.h0, Om0=self.om0)

    def ensure_directories(self) -> None:
        """Create required output and data directories if they do not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_cutouts_dir.mkdir(parents=True, exist_ok=True)
        self.fits_raw_dir.mkdir(parents=True, exist_ok=True)
        self.fits_cutouts_dir.mkdir(parents=True, exist_ok=True)
        self.lupton_rgb_dir.mkdir(parents=True, exist_ok=True)

    def cosmology_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the cosmology configuration."""
        return {
            "model": "FlatLambdaCDM",
            "H0_km_s_Mpc": self.h0,
            "Om0": self.om0,
        }

    def quality_cuts_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of SDSS DR17 selection cuts."""
        return {
            "data_release": self.data_release,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "modelMag_r_max": self.rmag_max,
            "photo_type": self.photo_type,
            "spec_class": self.spec_class,
            "zWarning": self.z_warning,
            "fracDeV_r_early_threshold": self.frac_dev_early_threshold,
        }

    def imaging_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of Phase 2 imaging configuration."""
        return {
            "cutout_size_px": self.cutout_size_px,
            "pixel_scale_arcsec": self.pixel_scale_arcsec,
            "fits_showcase_count": self.fits_showcase_count,
            "lupton_q": self.lupton_q,
            "lupton_stretch": self.lupton_stretch,
        }
