"""Unit tests for Phase 2 SkyServer RGB cutouts, FITS Cutout2D, NaN masking, and Lupton RGB."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

from src.acquisition import SDSSCatalogFetcher
from src.config import ProjectConfig
from src.database import GalaxyDatabase
from src.imaging import (
    MultiBandFITSProcessor,
    SkyServerRGBDownloader,
    extract_wcs_cutout,
    mask_and_clean_cutout,
    synthesize_lupton_rgb,
    validate_image_bytes,
)


def _make_synthetic_wcs_frame(
    shape: tuple[int, int] = (200, 200),
    ra_center: float = 180.0,
    dec_center: float = 0.5,
    pixel_scale_deg: float = 0.396 / 3600.0,
) -> tuple[np.ndarray, fits.Header]:
    """Build a synthetic calibrated 2D FITS frame with a Gaussian galaxy + NaN artifacts."""
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2.0, shape[0] / 2.0]
    w.wcs.crval = [ra_center, dec_center]
    w.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = w.to_header()

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r2 = (xx - shape[1] / 2.0) ** 2 + (yy - shape[0] / 2.0) ** 2
    sky = 0.25
    galaxy = 5.0 * np.exp(-r2 / (2.0 * 8.0**2))
    data = (sky + galaxy).astype(np.float32)

    # Inject NaN and Inf bad pixels to verify masking
    data[95:98, 95:98] = np.nan
    data[10, 10] = np.inf
    return data, header


def _seed_test_database(db_path: Path) -> GalaxyDatabase:
    """Populate a temporary SQLite DB with two galaxies for imaging tests."""
    raw_df = pd.DataFrame(
        [
            {
                "objID": 1001,
                "ra": 180.0,
                "dec": 0.5,
                "z": 0.04,
                "modelMag_u": 17.8,
                "modelMag_g": 16.0,
                "modelMag_r": 15.1,
                "modelMag_i": 14.7,
                "modelMag_z": 14.4,
                "petroRad_r": 9.0,
                "petroR50_r": 3.0,
                "petroR90_r": 9.0,
                "fracDeV_r": 0.90,
                "lgm_tot_p50": 10.8,
                "sfr_tot_p50": -1.1,
                "p_el_debiased": 0.92,
                "p_cs_debiased": 0.05,
            },
            {
                "objID": 1002,
                "ra": 181.0,
                "dec": 0.8,
                "z": 0.05,
                "modelMag_u": 17.2,
                "modelMag_g": 15.9,
                "modelMag_r": 15.3,
                "modelMag_i": 15.0,
                "modelMag_z": 14.8,
                "petroRad_r": 11.0,
                "petroR50_r": 4.2,
                "petroR90_r": 9.2,
                "fracDeV_r": 0.15,
                "lgm_tot_p50": 10.1,
                "sfr_tot_p50": 0.3,
                "p_el_debiased": 0.06,
                "p_cs_debiased": 0.91,
            },
        ]
    )
    clean_df = SDSSCatalogFetcher(ProjectConfig()).clean_and_derive(raw_df)
    db = GalaxyDatabase(db_path).connect()
    db.upsert_galaxies(clean_df)
    return db


def test_cutout_url_and_image_validation() -> None:
    cfg = ProjectConfig()
    downloader = SkyServerRGBDownloader(cfg)
    url = downloader.build_cutout_url(ra=180.123456, dec=-0.5, scale=0.396, width=128, height=128)

    assert "skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg" in url
    assert "ra=180.123456" in url
    assert "width=128" in url
    assert "height=128" in url

    legacy_url = downloader.build_legacy_sdss_url(ra=180.123456, dec=-0.5, scale=0.396, size=128)
    assert "legacysurvey.org/viewer/cutout.jpg" in legacy_url
    assert "layer=sdss" in legacy_url
    assert "size=128" in legacy_url

    # Valid JPEG bytes pass validation
    buf = io.BytesIO()
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(buf, format="JPEG")
    assert validate_image_bytes(buf.getvalue()) == (128, 128)

    # HTML error page fails validation
    with pytest.raises(Exception):
        validate_image_bytes(b"<html><body>403 Forbidden</body></html>")


def test_fits_cutout2d_nan_masking_and_lupton_rgb() -> None:
    data, header = _make_synthetic_wcs_frame()

    cutout, out_header = extract_wcs_cutout(
        data=data,
        header=header,
        ra=180.0,
        dec=0.5,
        size_px=64,
    )
    assert cutout.shape == (64, 64)
    assert "CRVAL1" in out_header
    assert np.isnan(cutout).any()  # injected NaN region falls inside cutout

    cleaned, diag = mask_and_clean_cutout(cutout, subtract_sky=True)
    assert cleaned.shape == (64, 64)
    assert np.isfinite(cleaned).all()
    assert diag["nan_fraction"] > 0.0
    assert abs(diag["sky_median"] - 0.25) < 0.05

    # Synthesize Lupton RGB from g, r, i bands
    rgb = synthesize_lupton_rgb(
        g_data=cleaned * 0.8,
        r_data=cleaned * 1.0,
        i_data=cleaned * 1.2,
        q=8.0,
        stretch=0.5,
    )
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8
    assert int(rgb.max()) > 0


def test_database_image_upsert_missing_filter_and_showcase_selection(tmp_path: Path) -> None:
    db_file = tmp_path / "imaging_test.db"
    db = _seed_test_database(db_file)
    try:
        # Both galaxies initially missing rgb_cutout
        missing_before = db.get_galaxies_missing_images("rgb_cutout")
        assert list(missing_before["objID"]) == [1001, 1002]

        # Register RGB + FITS + Lupton records for objID 1001 (and test idempotent upsert)
        records = [
            {
                "objID": 1001,
                "image_type": "rgb_cutout",
                "band": "gri",
                "file_path": "data/processed/rgb_cutouts/1001.jpg",
                "pixel_scale_arcsec": 0.396,
                "width_px": 128,
                "height_px": 128,
            },
            {
                "objID": 1001,
                "image_type": "fits_cutout",
                "band": "r",
                "file_path": "data/processed/fits_cutouts/1001_r.fits",
                "pixel_scale_arcsec": 0.396,
                "width_px": 128,
                "height_px": 128,
            },
            {
                "objID": 1001,
                "image_type": "lupton_rgb",
                "band": "gri",
                "file_path": "data/processed/lupton_rgb/1001_lupton_rgb.png",
                "pixel_scale_arcsec": 0.396,
                "width_px": 128,
                "height_px": 128,
            },
        ]
        assert db.upsert_images(records) == 3
        assert db.upsert_images(records) == 3  # idempotent conflict update
        assert db.get_table_counts()["images"] == 3

        # Only objID 1002 remains missing rgb_cutout
        missing_after = db.get_galaxies_missing_images("rgb_cutout")
        assert list(missing_after["objID"]) == [1002]

        # v_galaxy_dashboard exposes rgb_image_path for 1001
        view_df = db.query_df("SELECT objID, rgb_image_path FROM v_galaxy_dashboard ORDER BY objID")
        assert view_df.iloc[0]["rgb_image_path"] == "data/processed/rgb_cutouts/1001.jpg"
        assert pd.isna(view_df.iloc[1]["rgb_image_path"])

        # Summary groups cleanly by image_type & band
        summary = db.get_images_summary()
        assert len(summary) == 3

        # Showcase selector picks balanced early + late
        proc = MultiBandFITSProcessor(ProjectConfig())
        showcase = proc.select_showcase_galaxies(db, count=2)
        assert set(showcase["morph_proxy"]) == {"early", "late"}
    finally:
        db.close()
