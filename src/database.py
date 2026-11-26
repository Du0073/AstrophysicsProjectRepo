"""Normalized SQLite database manager for the SDSS Galaxy Atlas."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

GALAXY_COLUMNS: tuple[str, ...] = (
    "objID",
    "ra",
    "dec",
    "redshift",
    "model_mag_u",
    "model_mag_g",
    "model_mag_r",
    "model_mag_i",
    "model_mag_z",
    "petro_rad_r",
    "petro_r50_r",
    "petro_r90_r",
    "concentration",
    "frac_dev_r",
    "log_stellar_mass",
    "log_sfr",
    "p_el_debiased",
    "p_cs_debiased",
    "color_gr",
    "color_ug",
    "abs_mag_r",
    "re_kpc",
    "morph_proxy",
    "retrieved_at",
)

IMAGE_COLUMNS: tuple[str, ...] = (
    "objID",
    "image_type",
    "band",
    "file_path",
    "pixel_scale_arcsec",
    "width_px",
    "height_px",
    "fetched_at",
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "objID",
    "model_name",
    "model_version",
    "predicted_class",
    "confidence",
    "class_probabilities_json",
    "predicted_at",
)

SCHEMA_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS galaxies (
        objID INTEGER PRIMARY KEY,
        ra REAL NOT NULL,
        dec REAL NOT NULL,
        redshift REAL NOT NULL,
        model_mag_u REAL NOT NULL,
        model_mag_g REAL NOT NULL,
        model_mag_r REAL NOT NULL,
        model_mag_i REAL NOT NULL,
        model_mag_z REAL NOT NULL,
        petro_rad_r REAL NOT NULL,
        petro_r50_r REAL NOT NULL,
        petro_r90_r REAL NOT NULL,
        concentration REAL NOT NULL,
        frac_dev_r REAL NOT NULL,
        log_stellar_mass REAL NOT NULL,
        log_sfr REAL NOT NULL,
        p_el_debiased REAL,
        p_cs_debiased REAL,
        color_gr REAL NOT NULL,
        color_ug REAL NOT NULL,
        abs_mag_r REAL NOT NULL,
        re_kpc REAL NOT NULL,
        morph_proxy TEXT NOT NULL CHECK (morph_proxy IN ('early', 'late')),
        retrieved_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS images (
        image_id INTEGER PRIMARY KEY AUTOINCREMENT,
        objID INTEGER NOT NULL,
        image_type TEXT NOT NULL,
        band TEXT,
        file_path TEXT NOT NULL,
        pixel_scale_arcsec REAL,
        width_px INTEGER,
        height_px INTEGER,
        fetched_at TEXT NOT NULL,
        FOREIGN KEY (objID) REFERENCES galaxies(objID) ON DELETE CASCADE,
        UNIQUE (objID, image_type, band)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        objID INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL DEFAULT 'v1',
        predicted_class TEXT NOT NULL,
        confidence REAL,
        class_probabilities_json TEXT,
        predicted_at TEXT NOT NULL,
        FOREIGN KEY (objID) REFERENCES galaxies(objID) ON DELETE CASCADE,
        UNIQUE (objID, model_name, model_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_galaxies_redshift ON galaxies(redshift)",
    "CREATE INDEX IF NOT EXISTS idx_galaxies_morph_proxy ON galaxies(morph_proxy)",
    "CREATE INDEX IF NOT EXISTS idx_images_objid ON images(objID)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_objid_model ON predictions(objID, model_name)",
    """
    CREATE VIEW IF NOT EXISTS v_galaxy_dashboard AS
    SELECT
        g.objID,
        g.ra,
        g.dec,
        g.redshift,
        g.model_mag_u,
        g.model_mag_g,
        g.model_mag_r,
        g.model_mag_i,
        g.model_mag_z,
        g.petro_rad_r,
        g.petro_r50_r,
        g.petro_r90_r,
        g.concentration,
        g.frac_dev_r,
        g.log_stellar_mass,
        g.log_sfr,
        g.p_el_debiased,
        g.p_cs_debiased,
        g.color_gr,
        g.color_ug,
        g.abs_mag_r,
        g.re_kpc,
        g.morph_proxy,
        g.retrieved_at,
        img.file_path AS rgb_image_path,
        img.pixel_scale_arcsec AS rgb_pixel_scale_arcsec,
        pred.model_name AS latest_model_name,
        pred.predicted_class AS latest_predicted_class,
        pred.confidence AS latest_confidence
    FROM galaxies AS g
    LEFT JOIN images AS img
        ON img.image_id = (
            SELECT i2.image_id
            FROM images AS i2
            WHERE i2.objID = g.objID
              AND i2.image_type = 'rgb_cutout'
            ORDER BY i2.fetched_at DESC, i2.image_id DESC
            LIMIT 1
        )
    LEFT JOIN predictions AS pred
        ON pred.prediction_id = (
            SELECT p2.prediction_id
            FROM predictions AS p2
            WHERE p2.objID = g.objID
            ORDER BY p2.predicted_at DESC, p2.prediction_id DESC
            LIMIT 1
        )
    """,
)


def _to_sqlite_value(value: Any) -> Any:
    """Convert pandas/NumPy scalars and NaN values into native SQLite types."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


class GalaxyDatabase:
    """Context manager wrapping the normalized SQLite galaxy atlas database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the active SQLite connection."""
        if self._conn is None:
            raise RuntimeError("Database connection is not open. Use a 'with' block or connect().")
        return self._conn

    def connect(self) -> GalaxyDatabase:
        """Open the SQLite connection, enable foreign keys, and ensure schema exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.initialize_schema()
        return self

    def close(self) -> None:
        """Close the active SQLite connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> GalaxyDatabase:
        return self.connect()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def initialize_schema(self) -> None:
        """Idempotently create all tables, indexes, and views."""
        with self.conn:
            for statement in SCHEMA_DDL:
                self.conn.execute(statement)

    def upsert_galaxies(
        self,
        df: pd.DataFrame,
        retrieved_at: str | None = None,
    ) -> int:
        """Bulk upsert galaxy records into ``galaxies`` using parameterized SQL.

        Parameters
        ----------
        df:
            Cleaned galaxy DataFrame containing all required ``GALAXY_COLUMNS``
            (except ``retrieved_at``, which is populated automatically if omitted).
        retrieved_at:
            ISO-8601 UTC timestamp string. Defaults to current UTC time.
        """
        if df.empty:
            return 0

        timestamp = retrieved_at or datetime.now(timezone.utc).isoformat()
        work = df.copy()
        if "retrieved_at" not in work.columns:
            work["retrieved_at"] = timestamp

        missing = [col for col in GALAXY_COLUMNS if col not in work.columns]
        if missing:
            raise ValueError(f"Missing required galaxy columns for upsert: {missing}")

        placeholders = ", ".join("?" for _ in GALAXY_COLUMNS)
        col_list = ", ".join(GALAXY_COLUMNS)
        update_assignments = ", ".join(
            f"{col} = excluded.{col}" for col in GALAXY_COLUMNS if col != "objID"
        )
        sql = (
            f"INSERT INTO galaxies ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT(objID) DO UPDATE SET {update_assignments}"
        )

        rows = [
            tuple(_to_sqlite_value(val) for val in row)
            for row in work[list(GALAXY_COLUMNS)].itertuples(index=False, name=None)
        ]

        with self.conn:
            self.conn.executemany(sql, rows)

        return len(rows)

    def upsert_images(
        self,
        records: pd.DataFrame | Sequence[Mapping[str, Any]],
        fetched_at: str | None = None,
    ) -> int:
        """Bulk upsert image asset records into ``images`` using parameterized SQL."""
        work = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        if work.empty:
            return 0

        timestamp = fetched_at or datetime.now(timezone.utc).isoformat()
        if "fetched_at" not in work.columns:
            work["fetched_at"] = timestamp

        missing = [col for col in IMAGE_COLUMNS if col not in work.columns]
        if missing:
            raise ValueError(f"Missing required image columns for upsert: {missing}")

        placeholders = ", ".join("?" for _ in IMAGE_COLUMNS)
        col_list = ", ".join(IMAGE_COLUMNS)
        sql = (
            f"INSERT INTO images ({col_list}) VALUES ({placeholders}) "
            "ON CONFLICT(objID, image_type, band) DO UPDATE SET "
            "file_path = excluded.file_path, "
            "pixel_scale_arcsec = excluded.pixel_scale_arcsec, "
            "width_px = excluded.width_px, "
            "height_px = excluded.height_px, "
            "fetched_at = excluded.fetched_at"
        )

        rows = [
            tuple(_to_sqlite_value(val) for val in row)
            for row in work[list(IMAGE_COLUMNS)].itertuples(index=False, name=None)
        ]
        with self.conn:
            self.conn.executemany(sql, rows)

        return len(rows)

    def get_images_summary(self) -> pd.DataFrame:
        """Return counts and pixel scale summary grouped by image_type and band."""
        sql = """
        SELECT
            image_type,
            band,
            COUNT(*) AS image_count,
            AVG(pixel_scale_arcsec) AS avg_pixel_scale_arcsec,
            AVG(width_px) AS avg_width_px,
            AVG(height_px) AS avg_height_px
        FROM images
        GROUP BY image_type, band
        ORDER BY image_type, band
        """
        return self.query_df(sql)

    def upsert_predictions(
        self,
        records: pd.DataFrame | Sequence[Mapping[str, Any]],
        predicted_at: str | None = None,
    ) -> int:
        """Bulk upsert ML/DL morphology predictions into ``predictions`` via parameterized SQL."""
        work = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        if work.empty:
            return 0

        timestamp = predicted_at or datetime.now(timezone.utc).isoformat()
        if "predicted_at" not in work.columns:
            work["predicted_at"] = timestamp
        if "model_version" not in work.columns:
            work["model_version"] = "v1"
        if "class_probabilities_json" not in work.columns:
            work["class_probabilities_json"] = None

        missing = [col for col in PREDICTION_COLUMNS if col not in work.columns]
        if missing:
            raise ValueError(f"Missing required prediction columns for upsert: {missing}")

        placeholders = ", ".join("?" for _ in PREDICTION_COLUMNS)
        col_list = ", ".join(PREDICTION_COLUMNS)
        sql = (
            f"INSERT INTO predictions ({col_list}) VALUES ({placeholders}) "
            "ON CONFLICT(objID, model_name, model_version) DO UPDATE SET "
            "predicted_class = excluded.predicted_class, "
            "confidence = excluded.confidence, "
            "class_probabilities_json = excluded.class_probabilities_json, "
            "predicted_at = excluded.predicted_at"
        )

        rows = []
        for row in work[list(PREDICTION_COLUMNS)].itertuples(index=False, name=None):
            converted = []
            for col_name, val in zip(PREDICTION_COLUMNS, row):
                if col_name == "class_probabilities_json" and isinstance(val, Mapping):
                    converted.append(json.dumps(val, sort_keys=True))
                else:
                    converted.append(_to_sqlite_value(val))
            rows.append(tuple(converted))

        with self.conn:
            self.conn.executemany(sql, rows)

        return len(rows)

    def get_predictions_summary(self) -> pd.DataFrame:
        """Return prediction counts and mean confidence grouped by model and predicted class."""
        sql = """
        SELECT
            model_name,
            model_version,
            predicted_class,
            COUNT(*) AS galaxy_count,
            AVG(confidence) AS avg_confidence
        FROM predictions
        GROUP BY model_name, model_version, predicted_class
        ORDER BY model_name, galaxy_count DESC
        """
        return self.query_df(sql)

    def get_galaxies_missing_images(
        self,
        image_type: str = "rgb_cutout",
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Return catalog galaxies that do not yet have a registered image of ``image_type``."""
        params: list[Any] = [image_type]
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(int(limit))

        sql = f"""
        SELECT
            g.objID,
            g.ra,
            g.dec,
            g.redshift,
            g.model_mag_r,
            g.petro_r50_r,
            g.morph_proxy,
            g.p_el_debiased,
            g.p_cs_debiased
        FROM galaxies AS g
        WHERE NOT EXISTS (
            SELECT 1
            FROM images AS i
            WHERE i.objID = g.objID
              AND i.image_type = ?
        )
        ORDER BY g.objID
        {limit_clause}
        """
        return self.query_df(sql, params=params)

    def set_metadata(self, items: Mapping[str, Any]) -> None:
        """Persist key-value metadata records using parameterized upserts."""
        if not items:
            return

        sql = (
            "INSERT INTO metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
        rows = [
            (
                str(key),
                val if isinstance(val, str) else json.dumps(val, sort_keys=True),
            )
            for key, val in items.items()
        ]
        with self.conn:
            self.conn.executemany(sql, rows)

    def get_metadata(self) -> dict[str, str]:
        """Return all stored metadata key-value pairs."""
        cursor = self.conn.execute("SELECT key, value FROM metadata ORDER BY key")
        return {row["key"]: row["value"] for row in cursor.fetchall()}

    def query_df(
        self,
        sql: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Execute a parameterized SELECT query and return the result as a DataFrame."""
        return pd.read_sql_query(sql, self.conn, params=params)

    def get_morphology_summary(self) -> pd.DataFrame:
        """Compute GROUP BY morph_proxy summary statistics across physical scaling parameters."""
        sql = """
        SELECT
            morph_proxy,
            COUNT(*) AS galaxy_count,
            AVG(log_stellar_mass) AS avg_log_stellar_mass,
            AVG(re_kpc) AS avg_re_kpc,
            AVG(color_gr) AS avg_color_gr,
            AVG(log_sfr) AS avg_log_sfr,
            AVG(concentration) AS avg_concentration
        FROM galaxies
        GROUP BY morph_proxy
        ORDER BY morph_proxy
        """
        return self.query_df(sql)

    def get_table_counts(self) -> dict[str, int]:
        """Return row counts for all core tables."""
        tables = ("galaxies", "images", "predictions", "metadata")
        counts: dict[str, int] = {}
        for table in tables:
            row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            counts[table] = int(row["n"])
        return counts
