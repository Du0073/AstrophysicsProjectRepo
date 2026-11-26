"""SDSS DR17 SQL acquisition, validation, and physical parameter derivation."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astroquery.sdss import SDSSClass

from .config import ProjectConfig
from .database import GalaxyDatabase

logger = logging.getLogger(__name__)

# Official SkyServer REST SQL endpoint (astroquery's legacy x_results.aspx path
# returns HTML 403 Forbidden on current SDSS DR17 SkyServer deployments).
SKYSERVER_SQL_URL_SUFFIX = "/dr{dr}/SkyServerWS/SearchTools/SqlSearch"

# Mapping from raw SDSS DR17 SQL column names (case-insensitive via lower()) to
# canonical snake_case schema names.
RAW_TO_CANONICAL_COLUMNS: dict[str, str] = {
    "objid": "objID",
    "ra": "ra",
    "dec": "dec",
    "z": "redshift",
    "modelmag_u": "model_mag_u",
    "modelmag_g": "model_mag_g",
    "modelmag_r": "model_mag_r",
    "modelmag_i": "model_mag_i",
    "modelmag_z": "model_mag_z",
    "petrorad_r": "petro_rad_r",
    "petror50_r": "petro_r50_r",
    "petror90_r": "petro_r90_r",
    "fracdev_r": "frac_dev_r",
    "lgm_tot_p50": "log_stellar_mass",
    "sfr_tot_p50": "log_sfr",
    "p_el_debiased": "p_el_debiased",
    "p_cs_debiased": "p_cs_debiased",
}

REQUIRED_CANONICAL_COLUMNS: tuple[str, ...] = (
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
    "frac_dev_r",
    "log_stellar_mass",
    "log_sfr",
)

OPTIONAL_ZOO_COLUMNS: tuple[str, ...] = (
    "p_el_debiased",
    "p_cs_debiased",
)


class SDSSCatalogFetcher:
    """Fetch, clean, and derive physical properties for SDSS DR17 galaxies."""

    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or ProjectConfig()

    def build_sql_query(self, top_n: int | None = None) -> str:
        """Build the SDSS DR17 SkyServer SQL query with configured selection cuts.

        Parameters
        ----------
        top_n:
            Maximum number of rows returned by SDSS ``SELECT TOP``. Defaults to
            ``config.default_top_n``.
        """
        limit = self.config.default_top_n if top_n is None else int(top_n)
        if limit <= 0:
            raise ValueError(f"top_n must be a positive integer, got {limit}.")

        cfg = self.config
        return f"""SELECT TOP {limit}
    p.objID,
    p.ra,
    p.dec,
    s.z,
    p.modelMag_u,
    p.modelMag_g,
    p.modelMag_r,
    p.modelMag_i,
    p.modelMag_z,
    p.petroRad_r,
    p.petroR50_r,
    p.petroR90_r,
    p.fracDeV_r,
    gse.lgm_tot_p50,
    gse.sfr_tot_p50,
    z.p_el_debiased,
    z.p_cs_debiased
FROM PhotoObj AS p
INNER JOIN SpecObj AS s
    ON s.bestObjID = p.objID
INNER JOIN galSpecExtra AS gse
    ON gse.specObjID = s.specObjID
LEFT JOIN zooSpec AS z
    ON z.objid = p.objID
WHERE
    p.type = {cfg.photo_type}
    AND s.class = '{cfg.spec_class}'
    AND s.zWarning = {cfg.z_warning}
    AND s.z BETWEEN {cfg.z_min} AND {cfg.z_max}
    AND p.modelMag_r < {cfg.rmag_max}
ORDER BY p.objID"""

    def fetch_raw_catalog(
        self,
        top_n: int | None = None,
        save_raw: bool = True,
        max_retries: int = 3,
        backoff_s: float = 3.0,
    ) -> tuple[pd.DataFrame, str]:
        """Execute the SDSS DR17 SQL query with retry handling for transient timeouts.

        Returns
        -------
        tuple[pd.DataFrame, str]
            Raw catalog DataFrame and the exact SQL query string executed.
        """
        sql = self.build_sql_query(top_n=top_n)
        sdss_client = SDSSClass()
        sdss_client.QUERY_URL_SUFFIX_DR_NEW = SKYSERVER_SQL_URL_SUFFIX
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Querying SDSS DR%d (top_n=%s, attempt %d/%d)...",
                    self.config.data_release,
                    top_n or self.config.default_top_n,
                    attempt,
                    max_retries,
                )
                table = sdss_client.query_sql(
                    sql,
                    data_release=self.config.data_release,
                    timeout=self.config.sdss_timeout_s,
                    cache=False,
                )
                if table is None or len(table) == 0:
                    raise RuntimeError("SDSS DR17 query returned no rows.")

                df_raw = table.to_pandas()
                if save_raw:
                    self.save_raw_csv(df_raw)
                return df_raw, sql
            except Exception as exc:  # noqa: BLE001 — handle network/SkyServer errors
                last_error = exc
                logger.warning(
                    "SDSS query attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(backoff_s * attempt)

        raise RuntimeError(
            f"SDSS DR{self.config.data_release} query failed after {max_retries} attempts: {last_error}"
        ) from last_error

    def save_raw_csv(self, df_raw: pd.DataFrame, path: Path | None = None) -> Path:
        """Persist raw SDSS query results to CSV."""
        out_path = path or self.config.raw_catalog_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(out_path, index=False)
        logger.info("Saved %d raw rows to %s", len(df_raw), out_path)
        return out_path

    def clean_and_derive(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Validate schema, filter SDSS sentinel values, and compute physical parameters.

        Derived columns (stored as unitless float values):
        - ``concentration``: ``petro_r90_r / petro_r50_r``
        - ``color_gr``: ``model_mag_g - model_mag_r``
        - ``color_ug``: ``model_mag_u - model_mag_g``
        - ``abs_mag_r``: ``model_mag_r - distmod(redshift)``
        - ``re_kpc``: physical half-light radius from ``petro_r50_r`` [arcsec] via
          ``FlatLambdaCDM.kpc_proper_per_arcmin(z) / 60``
        - ``morph_proxy``: ``"early"`` if ``frac_dev_r >= 0.5`` else ``"late"``
        """
        df = self._normalize_columns(df_raw)

        missing = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required SDSS columns after normalization: {missing}")

        for col in OPTIONAL_ZOO_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        # Coerce numeric types cleanly (preserving integer objID)
        df["objID"] = pd.to_numeric(df["objID"], errors="coerce")
        numeric_cols = [c for c in REQUIRED_CANONICAL_COLUMNS if c != "objID"] + list(
            OPTIONAL_ZOO_COLUMNS
        )
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean nullable Galaxy Zoo vote fractions (replace sentinels with NaN)
        for col in OPTIONAL_ZOO_COLUMNS:
            invalid_zoo = (~np.isfinite(df[col])) | (df[col] < 0.0) | (df[col] > 1.0)
            df.loc[invalid_zoo, col] = np.nan

        # Build validity mask for required physical and photometric fields
        valid = np.ones(len(df), dtype=bool)
        for col in REQUIRED_CANONICAL_COLUMNS:
            valid &= np.isfinite(df[col].to_numpy(dtype=float))

        # Sentinel and physical range checks (SDSS uses -9999 / -99.9 sentinels)
        mag_cols = [
            "model_mag_u",
            "model_mag_g",
            "model_mag_r",
            "model_mag_i",
            "model_mag_z",
        ]
        for col in mag_cols:
            valid &= (df[col] > -100.0) & (df[col] < 50.0)

        valid &= (df["redshift"] >= self.config.z_min) & (df["redshift"] <= self.config.z_max)
        valid &= df["model_mag_r"] < self.config.rmag_max
        valid &= (df["petro_rad_r"] > 0.0) & (df["petro_r50_r"] > 0.0) & (df["petro_r90_r"] > 0.0)
        valid &= (df["frac_dev_r"] >= 0.0) & (df["frac_dev_r"] <= 1.0)
        valid &= df["log_stellar_mass"] > 0.0
        valid &= df["log_sfr"] > -90.0

        df_clean = df.loc[valid].copy()
        df_clean["objID"] = df_clean["objID"].astype("int64")
        df_clean = df_clean.drop_duplicates(subset=["objID"], keep="first").reset_index(drop=True)

        if df_clean.empty:
            return self._empty_derived_frame()

        cosmo = self.config.cosmology
        redshifts = df_clean["redshift"].to_numpy(dtype=float)

        dist_mod = cosmo.distmod(redshifts).value
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshifts).value / 60.0

        df_clean["concentration"] = (
            df_clean["petro_r90_r"].to_numpy(dtype=float)
            / df_clean["petro_r50_r"].to_numpy(dtype=float)
        )
        df_clean["color_gr"] = df_clean["model_mag_g"] - df_clean["model_mag_r"]
        df_clean["color_ug"] = df_clean["model_mag_u"] - df_clean["model_mag_g"]
        df_clean["abs_mag_r"] = df_clean["model_mag_r"].to_numpy(dtype=float) - dist_mod
        df_clean["re_kpc"] = df_clean["petro_r50_r"].to_numpy(dtype=float) * kpc_per_arcsec
        df_clean["morph_proxy"] = np.where(
            df_clean["frac_dev_r"].to_numpy(dtype=float) >= self.config.frac_dev_early_threshold,
            "early",
            "late",
        )

        ordered_cols = [
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
        ]
        return df_clean[ordered_cols]

    def run_pipeline(
        self,
        top_n: int | None = None,
        db_path: Path | None = None,
    ) -> dict[str, Any]:
        """Fetch SDSS DR17 catalog, derive physical parameters, and populate SQLite."""
        self.config.ensure_directories()
        df_raw, sql = self.fetch_raw_catalog(top_n=top_n, save_raw=True)
        df_clean = self.clean_and_derive(df_raw)
        retrieved_at = datetime.now(timezone.utc).isoformat()

        target_db = db_path or self.config.db_path
        with GalaxyDatabase(target_db) as db:
            upserted = db.upsert_galaxies(df_clean, retrieved_at=retrieved_at)
            db.set_metadata(
                {
                    "sdss_sql_query": sql,
                    "cosmology": self.config.cosmology_metadata(),
                    "quality_cuts": self.config.quality_cuts_metadata(),
                    "raw_row_count": len(df_raw),
                    "clean_row_count": len(df_clean),
                    "dropped_row_count": len(df_raw) - len(df_clean),
                    "upserted_row_count": upserted,
                    "last_retrieved_at": retrieved_at,
                }
            )
            summary = db.get_morphology_summary()

        return {
            "raw_count": len(df_raw),
            "clean_count": len(df_clean),
            "upserted_count": upserted,
            "db_path": str(target_db),
            "raw_csv_path": str(self.config.raw_catalog_path),
            "summary": summary,
        }

    @staticmethod
    def _normalize_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize raw SDSS or already-canonical column names to canonical schema names."""
        canonical_set = set(REQUIRED_CANONICAL_COLUMNS) | set(OPTIONAL_ZOO_COLUMNS)
        rename_map: dict[str, str] = {}
        for col in df_raw.columns:
            if col in canonical_set:
                continue
            mapped = RAW_TO_CANONICAL_COLUMNS.get(str(col).strip().lower())
            if mapped is not None:
                rename_map[col] = mapped
        return df_raw.rename(columns=rename_map).copy()

    @staticmethod
    def _empty_derived_frame() -> pd.DataFrame:
        """Return an empty DataFrame with the full canonical derived schema."""
        return pd.DataFrame(
            columns=[
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
            ]
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 SDSS DR17 acquisition and populate data/galaxy_atlas.db."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of galaxies to fetch from SDSS DR17 (default: 2500; use 100 for smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for Phase 1 SDSS DR17 ingestion."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()
    fetcher = SDSSCatalogFetcher()
    result = fetcher.run_pipeline(top_n=args.top_n)
    print(
        f"Ingested {result['clean_count']}/{result['raw_count']} galaxies into {result['db_path']}."
    )
    print("\nGROUP BY morph_proxy summary:")
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
