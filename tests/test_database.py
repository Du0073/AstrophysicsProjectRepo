"""Unit tests for Phase 1 SDSS acquisition, cosmology derivation, and SQLite database."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.acquisition import SDSSCatalogFetcher
from src.analysis import (
    load_galaxy_zoo_agreement,
    load_redshift_binned_stats,
    load_scaling_relations_sample,
)
from src.config import ProjectConfig
from src.database import GalaxyDatabase


def _make_raw_sdss_dataframe() -> pd.DataFrame:
    """Create a synthetic raw SDSS DR17 DataFrame including valid and sentinel rows."""
    return pd.DataFrame(
        [
            {
                # Valid early-type galaxy with Galaxy Zoo match
                "objID": 1237648702972362851,
                "ra": 180.1234,
                "dec": 0.4567,
                "z": 0.05,
                "modelMag_u": 18.2,
                "modelMag_g": 16.5,
                "modelMag_r": 15.6,
                "modelMag_i": 15.2,
                "modelMag_z": 14.9,
                "petroRad_r": 8.0,
                "petroR50_r": 3.0,
                "petroR90_r": 9.3,
                "fracDeV_r": 0.85,
                "lgm_tot_p50": 10.9,
                "sfr_tot_p50": -1.2,
                "p_el_debiased": 0.88,
                "p_cs_debiased": 0.08,
            },
            {
                # Valid late-type galaxy with NULL Galaxy Zoo votes
                "objID": 1237648702972362852,
                "ra": 180.5678,
                "dec": 0.7890,
                "z": 0.08,
                "modelMag_u": 17.5,
                "modelMag_g": 16.4,
                "modelMag_r": 15.9,
                "modelMag_i": 15.7,
                "modelMag_z": 15.5,
                "petroRad_r": 10.0,
                "petroR50_r": 4.0,
                "petroR90_r": 8.4,
                "fracDeV_r": 0.20,
                "lgm_tot_p50": 10.1,
                "sfr_tot_p50": 0.4,
                "p_el_debiased": np.nan,
                "p_cs_debiased": np.nan,
            },
            {
                # Sentinel -9999 mass — must be filtered out
                "objID": 1237648702972362853,
                "ra": 181.0,
                "dec": 1.0,
                "z": 0.06,
                "modelMag_u": 18.0,
                "modelMag_g": 16.8,
                "modelMag_r": 16.0,
                "modelMag_i": 15.7,
                "modelMag_z": 15.5,
                "petroRad_r": 7.0,
                "petroR50_r": 2.5,
                "petroR90_r": 6.5,
                "fracDeV_r": 0.60,
                "lgm_tot_p50": -9999.0,
                "sfr_tot_p50": -0.5,
                "p_el_debiased": 0.5,
                "p_cs_debiased": 0.4,
            },
            {
                # Non-positive petroR50_r — must be filtered out
                "objID": 1237648702972362854,
                "ra": 181.5,
                "dec": 1.5,
                "z": 0.04,
                "modelMag_u": 18.1,
                "modelMag_g": 16.9,
                "modelMag_r": 16.2,
                "modelMag_i": 15.9,
                "modelMag_z": 15.7,
                "petroRad_r": 5.0,
                "petroR50_r": 0.0,
                "petroR90_r": 6.0,
                "fracDeV_r": 0.40,
                "lgm_tot_p50": 9.8,
                "sfr_tot_p50": 0.1,
                "p_el_debiased": 0.1,
                "p_cs_debiased": 0.85,
            },
        ]
    )


def test_sql_query_contains_dr17_cuts_and_columns() -> None:
    cfg = ProjectConfig()
    fetcher = SDSSCatalogFetcher(cfg)
    sql = fetcher.build_sql_query(top_n=100)

    assert "SELECT TOP 100" in sql
    assert "s.class = 'GALAXY'" in sql
    assert "p.type = 3" in sql
    assert "s.zWarning = 0" in sql
    assert "s.z BETWEEN 0.02 AND 0.1" in sql
    assert "p.modelMag_r < 17.77" in sql
    assert "p.petroRad_r" in sql
    assert "p.petroR50_r" in sql
    assert "p.petroR90_r" in sql
    assert "gse.lgm_tot_p50" in sql
    assert "gse.sfr_tot_p50" in sql
    assert "LEFT JOIN zooSpec" in sql


def test_clean_and_derive_physics_and_sentinel_filtering() -> None:
    cfg = ProjectConfig()
    fetcher = SDSSCatalogFetcher(cfg)
    df_clean = fetcher.clean_and_derive(_make_raw_sdss_dataframe())

    assert len(df_clean) == 2
    assert list(df_clean["objID"]) == [1237648702972362851, 1237648702972362852]
    assert "log_sfr" in df_clean.columns
    assert "log_stellar_mass" in df_clean.columns
    assert list(df_clean["morph_proxy"]) == ["early", "late"]

    row_early = df_clean.iloc[0]
    cosmo = cfg.cosmology
    expected_abs_mag_r = 15.6 - cosmo.distmod(0.05).value
    expected_re_kpc = 3.0 * (cosmo.kpc_proper_per_arcmin(0.05).value / 60.0)

    assert np.isclose(row_early["concentration"], 9.3 / 3.0)
    assert np.isclose(row_early["color_gr"], 16.5 - 15.6)
    assert np.isclose(row_early["color_ug"], 18.2 - 16.5)
    assert np.isclose(row_early["abs_mag_r"], expected_abs_mag_r)
    assert np.isclose(row_early["re_kpc"], expected_re_kpc)
    assert np.isclose(row_early["log_sfr"], -1.2)

    # Nullable Galaxy Zoo votes remain NaN on the second galaxy
    row_late = df_clean.iloc[1]
    assert pd.isna(row_late["p_el_debiased"])
    assert pd.isna(row_late["p_cs_debiased"])


def test_database_schema_idempotency_upsert_and_queries(tmp_path: Path) -> None:
    db_file = tmp_path / "test_atlas.db"
    fetcher = SDSSCatalogFetcher(ProjectConfig())
    df_clean = fetcher.clean_and_derive(_make_raw_sdss_dataframe())

    with GalaxyDatabase(db_file) as db:
        # Calling initialize_schema repeatedly must be idempotent
        db.initialize_schema()

        # Verify tables, indexes, and view exist
        objects = db.query_df(
            "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'index', 'view')"
        )
        names_by_type = {
            t: set(objects.loc[objects["type"] == t, "name"])
            for t in ("table", "index", "view")
        }
        assert {"galaxies", "images", "predictions", "metadata"}.issubset(names_by_type["table"])
        assert {
            "idx_galaxies_redshift",
            "idx_galaxies_morph_proxy",
            "idx_images_objid",
            "idx_predictions_objid_model",
        }.issubset(names_by_type["index"])
        assert "v_galaxy_dashboard" in names_by_type["view"]

        # Upsert twice to verify idempotency / deduplication
        inserted = db.upsert_galaxies(df_clean)
        assert inserted == 2
        updated_df = df_clean.copy()
        updated_df.loc[0, "log_sfr"] = -1.5
        db.upsert_galaxies(updated_df)
        assert db.get_table_counts()["galaxies"] == 2

        # Verify physical columns are strictly SQLite REAL (no embedded units)
        type_df = db.query_df(
            """
            SELECT
                typeof(redshift) AS t_z,
                typeof(re_kpc) AS t_re,
                typeof(concentration) AS t_c,
                typeof(log_stellar_mass) AS t_m,
                typeof(log_sfr) AS t_sfr
            FROM galaxies
            LIMIT 1
            """
        )
        assert list(type_df.iloc[0]) == ["real", "real", "real", "real", "real"]

        # Verify metadata logging
        db.set_metadata({"cosmology": {"H0": 70.0, "Om0": 0.3}, "row_count": 2})
        meta = db.get_metadata()
        assert json.loads(meta["cosmology"]) == {"H0": 70.0, "Om0": 0.3}
        assert meta["row_count"] == "2"

        # Verify foreign key enforcement on images and predictions
        with pytest.raises(sqlite3.IntegrityError):
            with db.conn:
                db.conn.execute(
                    """
                    INSERT INTO images (objID, image_type, band, file_path, fetched_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (999999, "rgb_cutout", "gri", "data/img.jpg", "2026-01-01T00:00:00Z"),
                )

        # Insert valid image + prediction and check v_galaxy_dashboard view
        with db.conn:
            db.conn.execute(
                """
                INSERT INTO images (
                    objID, image_type, band, file_path, pixel_scale_arcsec, width_px, height_px, fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    1237648702972362851,
                    "rgb_cutout",
                    "gri",
                    "data/processed/1237648702972362851.jpg",
                    0.396,
                    128,
                    128,
                    "2026-01-01T00:00:00Z",
                ),
            )
            db.conn.execute(
                """
                INSERT INTO predictions (
                    objID, model_name, model_version, predicted_class, confidence, predicted_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    1237648702972362851,
                    "hubble_cnn",
                    "v1",
                    "Elliptical",
                    0.94,
                    "2026-01-01T00:00:00Z",
                ),
            )

        view_df = db.query_df("SELECT * FROM v_galaxy_dashboard ORDER BY objID")
        assert len(view_df) == 2
        assert view_df.iloc[0]["rgb_image_path"] == "data/processed/1237648702972362851.jpg"
        assert view_df.iloc[0]["latest_predicted_class"] == "Elliptical"
        assert pd.isna(view_df.iloc[1]["rgb_image_path"])

        # Verify GROUP BY morph_proxy summary and analytical helpers
        summary = db.get_morphology_summary()
        assert list(summary["morph_proxy"]) == ["early", "late"]
        assert list(summary["galaxy_count"]) == [1, 1]
        assert np.isclose(summary.loc[summary["morph_proxy"] == "early", "avg_log_sfr"].iloc[0], -1.5)

        filtered = load_scaling_relations_sample(db, z_min=0.06, z_max=0.09, morph_proxy="late")
        assert len(filtered) == 1
        assert int(filtered.iloc[0]["objID"]) == 1237648702972362852

        z_bins = load_redshift_binned_stats(db, bin_width=0.02)
        assert len(z_bins) == 2

        zoo_df = load_galaxy_zoo_agreement(db)
        assert len(zoo_df) == 1
        assert zoo_df.iloc[0]["morph_proxy"] == "early"
        assert int(zoo_df.iloc[0]["zoo_matched_count"]) == 1
