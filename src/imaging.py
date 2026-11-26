"""Phase 2 — Automated SkyServer RGB cutouts & multi-band FITS (g, r, i) processing."""

from __future__ import annotations

import argparse
import bz2
import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from astroquery.sdss import SDSSClass
from PIL import Image

from .acquisition import SKYSERVER_SQL_URL_SUFFIX
from .config import ProjectConfig
from .database import GalaxyDatabase

logger = logging.getLogger(__name__)

SKYSERVER_CUTOUT_URL = "https://skyserver.sdss.org/dr{dr}/SkyServerWS/ImgCutout/getjpeg"
LEGACY_SDSS_CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.jpg"
SDSS_SAS_FRAME_URL = (
    "https://data.sdss.org/sas/dr{dr}/eboss/photoObj/frames/"
    "{rerun}/{run}/{camcol}/frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"
)
FITS_BANDS: tuple[str, ...] = ("g", "r", "i")


def validate_image_bytes(raw_bytes: bytes) -> tuple[int, int]:
    """Verify raw bytes decode to a valid image and return ``(width, height)``."""
    if not raw_bytes or len(raw_bytes) < 128:
        raise ValueError("Downloaded image payload is empty or unexpectedly small.")
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img.verify()
    with Image.open(io.BytesIO(raw_bytes)) as img:
        width, height = img.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid decoded image dimensions: {width}x{height}")
        return int(width), int(height)


def extract_wcs_cutout(
    data: np.ndarray,
    header: fits.Header,
    ra: float,
    dec: float,
    size_px: int = 128,
) -> tuple[np.ndarray, fits.Header]:
    """Extract a sky-centered WCS ``Cutout2D`` from a 2D FITS science frame.

    Uses ``mode='partial'`` with ``fill_value=np.nan`` so galaxies near a frame
    edge still produce an exact ``(size_px, size_px)`` array whose out-of-bounds
    pixels are handled cleanly by :func:`mask_and_clean_cutout`.
    """
    wcs = WCS(header)
    position = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
    cutout = Cutout2D(
        np.asarray(data, dtype=np.float32),
        position=position,
        size=(int(size_px), int(size_px)),
        wcs=wcs,
        mode="partial",
        fill_value=np.nan,
    )
    out_header = header.copy()
    out_header.update(cutout.wcs.to_header())
    return np.asarray(cutout.data, dtype=np.float32), out_header


def mask_and_clean_cutout(
    data: np.ndarray,
    subtract_sky: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Mask NaN/Inf pixels using robust sky statistics and optionally subtract sky background.

    Parameters
    ----------
    data:
        2D floating-point cutout array.
    subtract_sky:
        When ``True``, subtracts the sigma-clipped sky median so background
        flux sits near zero prior to Lupton arcsinh stretching.

    Returns
    -------
    tuple[np.ndarray, dict[str, float]]
        Cleaned ``float32`` array and diagnostic metrics (``nan_fraction``,
        ``sky_median``, ``sky_std``).
    """
    arr = np.asarray(data, dtype=np.float32).copy()
    finite_mask = np.isfinite(arr)
    nan_fraction = float(1.0 - np.mean(finite_mask))

    if not np.any(finite_mask):
        return np.zeros_like(arr, dtype=np.float32), {
            "nan_fraction": 1.0,
            "sky_median": 0.0,
            "sky_std": 0.0,
        }

    finite_values = arr[finite_mask]
    _, sky_median, sky_std = sigma_clipped_stats(finite_values, sigma=3.0, maxiters=5)
    sky_median = float(sky_median)
    sky_std = float(sky_std)

    if nan_fraction > 0.0:
        arr[~finite_mask] = sky_median

    if subtract_sky:
        arr = arr - sky_median

    return arr.astype(np.float32), {
        "nan_fraction": nan_fraction,
        "sky_median": sky_median,
        "sky_std": sky_std,
    }


def synthesize_lupton_rgb(
    g_data: np.ndarray,
    r_data: np.ndarray,
    i_data: np.ndarray,
    q: float = 8.0,
    stretch: float = 0.5,
    minimum: float = 0.0,
) -> np.ndarray:
    """Synthesize a ``uint8`` Lupton arcsinh-stretched RGB image from ``(i, r, g)`` bands."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb = make_lupton_rgb(
            np.asarray(i_data, dtype=np.float32),
            np.asarray(r_data, dtype=np.float32),
            np.asarray(g_data, dtype=np.float32),
            Q=float(q),
            stretch=float(stretch),
            minimum=float(minimum),
        )
    return np.asarray(rgb, dtype=np.uint8)


class SkyServerRGBDownloader:
    """Resumable, multi-threaded downloader for SDSS DR17 SkyServer RGB cutouts."""

    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or ProjectConfig()
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": "SDSS-Galaxy-Atlas/0.2"})
            self._thread_local.session = session
        return session

    def build_cutout_url(
        self,
        ra: float,
        dec: float,
        scale: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> str:
        """Build the official SDSS DR17 SkyServer JPEG cutout URL."""
        base = SKYSERVER_CUTOUT_URL.format(dr=self.config.data_release)
        params = {
            "ra": f"{float(ra):.6f}",
            "dec": f"{float(dec):.6f}",
            "scale": f"{self.config.pixel_scale_arcsec if scale is None else float(scale):.4f}",
            "width": int(self.config.cutout_size_px if width is None else width),
            "height": int(self.config.cutout_size_px if height is None else height),
        }
        return f"{base}?{urlencode(params)}"

    def build_legacy_sdss_url(
        self,
        ra: float,
        dec: float,
        scale: float | None = None,
        size: int | None = None,
    ) -> str:
        """Build the Legacy Survey SDSS-layer JPEG cutout mirror URL."""
        params = {
            "ra": f"{float(ra):.6f}",
            "dec": f"{float(dec):.6f}",
            "layer": "sdss",
            "pixscale": f"{self.config.pixel_scale_arcsec if scale is None else float(scale):.4f}",
            "size": int(self.config.cutout_size_px if size is None else size),
        }
        return f"{LEGACY_SDSS_CUTOUT_URL}?{urlencode(params)}"

    def _probe_skyserver(self, ra: float, dec: float) -> bool:
        """Check whether SkyServer ImgCutout responds cleanly before bulk downloads."""
        try:
            resp = self._get_session().get(self.build_cutout_url(ra, dec), timeout=(6, 12))
            resp.raise_for_status()
            validate_image_bytes(resp.content)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _relative_posix_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.config.project_root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def download_single(
        self,
        obj_id: int,
        ra: float,
        dec: float,
        overwrite: bool = False,
        max_retries: int = 3,
        backoff_s: float = 1.5,
        prefer_legacy_mirror: bool = False,
    ) -> tuple[dict[str, Any] | None, str]:
        """Download or validate a single galaxy RGB cutout.

        Returns
        -------
        tuple[dict[str, Any] | None, str]
            ``(image_record, status)`` where ``status`` is one of
            ``'downloaded'``, ``'skipped'``, or ``'failed'``.
        """
        out_path = self.config.rgb_cutouts_dir / f"{int(obj_id)}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not overwrite and out_path.exists():
            try:
                width, height = validate_image_bytes(out_path.read_bytes())
                return (
                    {
                        "objID": int(obj_id),
                        "image_type": "rgb_cutout",
                        "band": "gri",
                        "file_path": self._relative_posix_path(out_path),
                        "pixel_scale_arcsec": self.config.pixel_scale_arcsec,
                        "width_px": width,
                        "height_px": height,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "skipped",
                )
            except Exception:  # noqa: BLE001 — re-download corrupt local file
                logger.warning("Re-downloading corrupt local JPEG: %s", out_path)

        urls = (
            [self.build_legacy_sdss_url(ra=ra, dec=dec), self.build_cutout_url(ra=ra, dec=dec)]
            if prefer_legacy_mirror
            else [self.build_cutout_url(ra=ra, dec=dec), self.build_legacy_sdss_url(ra=ra, dec=dec)]
        )
        session = self._get_session()

        for attempt in range(1, max_retries + 1):
            for url in urls:
                try:
                    resp = session.get(url, timeout=(6, self.config.image_timeout_s))
                    resp.raise_for_status()
                    width, height = validate_image_bytes(resp.content)
                    out_path.write_bytes(resp.content)
                    return (
                        {
                            "objID": int(obj_id),
                            "image_type": "rgb_cutout",
                            "band": "gri",
                            "file_path": self._relative_posix_path(out_path),
                            "pixel_scale_arcsec": self.config.pixel_scale_arcsec,
                            "width_px": width,
                            "height_px": height,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                        },
                        "downloaded",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "RGB cutout objID=%s url=%s attempt %d/%d failed: %s",
                        obj_id,
                        url,
                        attempt,
                        max_retries,
                        exc,
                    )
            if attempt < max_retries:
                time.sleep(backoff_s * attempt)

        logger.warning("Failed to download RGB cutout for objID=%s after %d tries.", obj_id, max_retries)
        return None, "failed"

    def download_for_database(
        self,
        db: GalaxyDatabase,
        limit: int | None = None,
        max_workers: int | None = None,
        overwrite: bool = False,
        commit_batch_size: int = 100,
    ) -> dict[str, int]:
        """Download RGB cutouts for catalog galaxies and register them in SQLite."""
        self.config.ensure_directories()
        if overwrite:
            limit_clause = "LIMIT ?" if limit is not None else ""
            params = (int(limit),) if limit is not None else None
            targets = db.query_df(
                f"SELECT objID, ra, dec FROM galaxies ORDER BY objID {limit_clause}",
                params=params,
            )
        else:
            targets = db.get_galaxies_missing_images(image_type="rgb_cutout", limit=limit)

        if targets.empty:
            logger.info("All requested galaxies already have registered rgb_cutout images.")
            return {"requested": 0, "downloaded": 0, "skipped": 0, "failed": 0}

        first_ra = float(targets.iloc[0]["ra"])
        first_dec = float(targets.iloc[0]["dec"])
        skyserver_ok = self._probe_skyserver(first_ra, first_dec)
        prefer_legacy_mirror = not skyserver_ok
        if prefer_legacy_mirror:
            logger.info(
                "SkyServer ImgCutout unreachable on probe; failing over to Legacy Survey SDSS layer."
            )

        workers = max(1, int(max_workers or self.config.rgb_max_workers))
        logger.info(
            "Starting SDSS RGB cutout acquisition for %d galaxies (workers=%d, mirror=%s)...",
            len(targets),
            workers,
            "legacy_sdss" if prefer_legacy_mirror else "skyserver",
        )

        stats = {"requested": len(targets), "downloaded": 0, "skipped": 0, "failed": 0}
        pending_records: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self.download_single,
                    int(row.objID),
                    float(row.ra),
                    float(row.dec),
                    overwrite,
                    3,
                    1.5,
                    prefer_legacy_mirror,
                ): int(row.objID)
                for row in targets.itertuples(index=False)
            }

            completed = 0
            for future in as_completed(futures):
                record, status = future.result()
                completed += 1
                stats[status] += 1
                if record is not None:
                    pending_records.append(record)

                if len(pending_records) >= commit_batch_size:
                    db.upsert_images(pending_records)
                    pending_records.clear()

                if completed % 250 == 0 or completed == len(targets):
                    logger.info(
                        "RGB progress: %d/%d (downloaded=%d, skipped=%d, failed=%d)",
                        completed,
                        len(targets),
                        stats["downloaded"],
                        stats["skipped"],
                        stats["failed"],
                    )

        if pending_records:
            db.upsert_images(pending_records)

        return stats


class MultiBandFITSProcessor:
    """Download and process multi-band SDSS DR17 FITS (g, r, i) frames for a showcase subset."""

    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or ProjectConfig()

    def _relative_posix_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.config.project_root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def select_showcase_galaxies(
        self,
        db: GalaxyDatabase,
        count: int | None = None,
    ) -> pd.DataFrame:
        """Select a balanced, bright showcase sample (half early/elliptical, half late/spiral)."""
        total = int(self.config.fits_showcase_count if count is None else count)
        if total <= 0:
            return pd.DataFrame()

        n_early = total // 2
        n_late = total - n_early

        early_df = db.query_df(
            """
            SELECT objID, ra, dec, redshift, model_mag_r, petro_r50_r,
                   morph_proxy, p_el_debiased, p_cs_debiased
            FROM galaxies
            WHERE morph_proxy = 'early'
            ORDER BY
                CASE WHEN p_el_debiased IS NOT NULL THEN 0 ELSE 1 END,
                COALESCE(p_el_debiased, frac_dev_r) DESC,
                model_mag_r ASC
            LIMIT ?
            """,
            params=(n_early,),
        )

        late_df = db.query_df(
            """
            SELECT objID, ra, dec, redshift, model_mag_r, petro_r50_r,
                   morph_proxy, p_el_debiased, p_cs_debiased
            FROM galaxies
            WHERE morph_proxy = 'late'
            ORDER BY
                CASE WHEN p_cs_debiased IS NOT NULL THEN 0 ELSE 1 END,
                COALESCE(p_cs_debiased, 1.0 - frac_dev_r) DESC,
                model_mag_r ASC
            LIMIT ?
            """,
            params=(n_late,),
        )

        return pd.concat([early_df, late_df], ignore_index=True)

    def fetch_frame_identifiers(
        self,
        obj_ids: list[int],
        max_retries: int = 3,
        backoff_s: float = 3.0,
    ) -> pd.DataFrame:
        """Query SDSS DR17 PhotoObj for ``(run, rerun, camcol, field)`` of showcase galaxies."""
        if not obj_ids:
            return pd.DataFrame(columns=["objID", "run", "rerun", "camcol", "field"])

        id_list = ", ".join(str(int(oid)) for oid in obj_ids)
        sql = f"""
        SELECT objID, run, rerun, camcol, field
        FROM PhotoObj
        WHERE objID IN ({id_list})
        """
        sdss_client = SDSSClass()
        sdss_client.QUERY_URL_SUFFIX_DR_NEW = SKYSERVER_SQL_URL_SUFFIX

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                table = sdss_client.query_sql(
                    sql,
                    data_release=self.config.data_release,
                    timeout=self.config.sdss_timeout_s,
                    cache=False,
                )
                if table is None or len(table) == 0:
                    raise RuntimeError("SDSS DR17 returned no frame identifiers.")
                df = table.to_pandas()
                df.columns = [str(c).strip() for c in df.columns]
                rename = {c: "objID" for c in df.columns if c.lower() == "objid"}
                df = df.rename(columns=rename)
                for col in ("objID", "run", "rerun", "camcol", "field"):
                    df[col] = df[col].astype("int64")
                return df[["objID", "run", "rerun", "camcol", "field"]]
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("Frame ID lookup attempt %d/%d failed: %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(backoff_s * attempt)

        raise RuntimeError(f"Failed to fetch SDSS frame IDs: {last_err}") from last_err

    def build_fits_frame_url(
        self,
        run: int,
        rerun: int,
        camcol: int,
        field: int,
        band: str,
    ) -> str:
        """Build the official SDSS DR17 SAS calibrated FITS frame URL."""
        return SDSS_SAS_FRAME_URL.format(
            dr=self.config.data_release,
            rerun=int(rerun),
            run=int(run),
            camcol=int(camcol),
            field=int(field),
            band=band,
        )

    def download_fits_frame(
        self,
        run: int,
        rerun: int,
        camcol: int,
        field: int,
        band: str,
        overwrite: bool = False,
        max_retries: int = 3,
    ) -> Path:
        """Download a compressed SDSS DR17 FITS frame to ``data/raw/fits/``."""
        filename = f"frame-{band}-{int(run):06d}-{int(camcol)}-{int(field):04d}.fits.bz2"
        out_path = self.config.fits_raw_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not overwrite and out_path.exists() and out_path.stat().st_size > 10_000:
            return out_path

        url = self.build_fits_frame_url(run=run, rerun=rerun, camcol=camcol, field=field, band=band)
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, timeout=max(self.config.image_timeout_s, 90))
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
                return out_path
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("FITS download %s attempt %d/%d failed: %s", filename, attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(2.5 * attempt)

        raise RuntimeError(f"Failed downloading {url}: {last_err}") from last_err

    def process_galaxy_fits(
        self,
        obj_id: int,
        ra: float,
        dec: float,
        run: int,
        rerun: int,
        camcol: int,
        field: int,
        overwrite: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Download ``g, r, i`` FITS frames, crop with ``Cutout2D``, mask NaNs, and build Lupton RGB."""
        self.config.ensure_directories()
        size_px = self.config.cutout_size_px
        cleaned_bands: dict[str, np.ndarray] = {}
        band_diagnostics: dict[str, dict[str, float]] = {}
        records: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat()

        for band in FITS_BANDS:
            raw_bz2_path = self.download_fits_frame(
                run=run,
                rerun=rerun,
                camcol=camcol,
                field=field,
                band=band,
                overwrite=overwrite,
            )
            with bz2.open(raw_bz2_path, "rb") as fh:
                with fits.open(fh) as hdul:
                    sci_data = hdul[0].data
                    sci_header = hdul[0].header

            cutout_data, cutout_header = extract_wcs_cutout(
                data=sci_data,
                header=sci_header,
                ra=ra,
                dec=dec,
                size_px=size_px,
            )
            cleaned, diag = mask_and_clean_cutout(cutout_data, subtract_sky=True)
            cleaned_bands[band] = cleaned
            band_diagnostics[band] = diag

            cutout_path = self.config.fits_cutouts_dir / f"{int(obj_id)}_{band}.fits"
            cutout_header["NANFRAC"] = (round(diag["nan_fraction"], 6), "Fraction of masked non-finite pixels")
            cutout_header["SKYMED"] = (round(diag["sky_median"], 6), "Subtracted sigma-clipped sky median")
            fits.PrimaryHDU(data=cleaned, header=cutout_header).writeto(cutout_path, overwrite=True)

            records.append(
                {
                    "objID": int(obj_id),
                    "image_type": "fits_cutout",
                    "band": band,
                    "file_path": self._relative_posix_path(cutout_path),
                    "pixel_scale_arcsec": self.config.pixel_scale_arcsec,
                    "width_px": size_px,
                    "height_px": size_px,
                    "fetched_at": now,
                }
            )

        lupton_rgb = synthesize_lupton_rgb(
            g_data=cleaned_bands["g"],
            r_data=cleaned_bands["r"],
            i_data=cleaned_bands["i"],
            q=self.config.lupton_q,
            stretch=self.config.lupton_stretch,
        )
        lupton_path = self.config.lupton_rgb_dir / f"{int(obj_id)}_lupton_rgb.png"
        Image.fromarray(lupton_rgb).save(lupton_path)

        records.append(
            {
                "objID": int(obj_id),
                "image_type": "lupton_rgb",
                "band": "gri",
                "file_path": self._relative_posix_path(lupton_path),
                "pixel_scale_arcsec": self.config.pixel_scale_arcsec,
                "width_px": int(lupton_rgb.shape[1]),
                "height_px": int(lupton_rgb.shape[0]),
                "fetched_at": now,
            }
        )

        return records, {"objID": int(obj_id), "bands": band_diagnostics}

    def run_showcase(
        self,
        db: GalaxyDatabase,
        count: int | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Select showcase galaxies, process their ``g, r, i`` FITS cutouts, and register in SQLite."""
        self.config.ensure_directories()
        showcase_df = self.select_showcase_galaxies(db, count=count)
        if showcase_df.empty:
            return {"selected": 0, "processed": 0, "failed": 0, "diagnostics": []}

        frame_ids = self.fetch_frame_identifiers(showcase_df["objID"].astype(int).tolist())
        merged = showcase_df.merge(frame_ids, on="objID", how="inner")

        processed = 0
        failed = 0
        diagnostics: list[dict[str, Any]] = []

        for idx, row in enumerate(merged.itertuples(index=False)):
            obj_id = int(row.objID)
            try:
                logger.info(
                    "Processing FITS showcase [%d/%d] objID=%s (%s, r=%.2f)...",
                    idx + 1,
                    len(merged),
                    obj_id,
                    row.morph_proxy,
                    float(row.model_mag_r),
                )
                records, diag = self.process_galaxy_fits(
                    obj_id=obj_id,
                    ra=float(row.ra),
                    dec=float(row.dec),
                    run=int(row.run),
                    rerun=int(row.rerun),
                    camcol=int(row.camcol),
                    field=int(row.field),
                    overwrite=overwrite,
                )
                db.upsert_images(records)
                diagnostics.append(diag)
                processed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("FITS showcase processing failed for objID=%s: %s", obj_id, exc)
                failed += 1

        return {
            "selected": len(showcase_df),
            "processed": processed,
            "failed": failed,
            "diagnostics": diagnostics,
        }


def plot_showcase_comparison(
    db: GalaxyDatabase,
    config: ProjectConfig | None = None,
    output_path: str | Path | None = None,
    max_galaxies: int = 8,
) -> plt.Figure:
    """Plot a side-by-side gallery comparing SkyServer RGB cutouts vs Lupton FITS composites."""
    cfg = config or ProjectConfig()
    df = db.query_df(
        """
        SELECT
            g.objID,
            g.redshift,
            g.morph_proxy,
            g.concentration,
            g.p_el_debiased,
            g.p_cs_debiased,
            rgb.file_path AS skyserver_path,
            lup.file_path AS lupton_path
        FROM galaxies AS g
        INNER JOIN images AS lup
            ON lup.objID = g.objID AND lup.image_type = 'lupton_rgb'
        LEFT JOIN images AS rgb
            ON rgb.objID = g.objID AND rgb.image_type = 'rgb_cutout'
        ORDER BY g.morph_proxy, g.model_mag_r ASC
        LIMIT ?
        """,
        params=(int(max_galaxies),),
    )

    if df.empty:
        raise ValueError("No Lupton RGB showcase images found in database.")

    n = len(df)
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 2.7 * n), squeeze=False)
    for row_idx, row in enumerate(df.itertuples(index=False)):
        sky_ax = axes[row_idx, 0]
        lup_ax = axes[row_idx, 1]

        sky_rel = row.skyserver_path
        if pd.notna(sky_rel) and (cfg.project_root / sky_rel).exists():
            sky_ax.imshow(Image.open(cfg.project_root / sky_rel))
        else:
            sky_ax.text(0.5, 0.5, "SkyServer RGB\nnot fetched", ha="center", va="center")
        sky_ax.set_xticks([])
        sky_ax.set_yticks([])

        lup_full = cfg.project_root / row.lupton_path
        if lup_full.exists():
            lup_ax.imshow(Image.open(lup_full))
        lup_ax.set_xticks([])
        lup_ax.set_yticks([])

        label = (
            f"{row.morph_proxy.upper()} | z={row.redshift:.3f} | "
            f"C={row.concentration:.2f}"
        )
        sky_ax.set_title(f"SDSS RGB Cutout\n{label}", fontsize=9)
        lup_ax.set_title(f"FITS Cutout2D + Lupton RGB\nobjID {int(row.objID)}", fontsize=9)

    fig.suptitle("Phase 2 Showcase: SkyServer RGB vs. Calibrated FITS Lupton RGB", fontsize=12, y=1.002)
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 image acquisition (SkyServer RGB cutouts + FITS g,r,i showcase)."
    )
    parser.add_argument(
        "--rgb-limit",
        type=int,
        default=None,
        help="Optional cap on number of RGB cutouts to fetch (default: all missing galaxies).",
    )
    parser.add_argument(
        "--skip-rgb",
        action="store_true",
        help="Skip bulk SkyServer RGB cutout download.",
    )
    parser.add_argument(
        "--fits-showcase",
        type=int,
        default=None,
        help="Number of galaxies for multi-band FITS g,r,i processing (default: 12; 0 to skip).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent download workers for SkyServer RGB cutouts (default: 6).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download existing image assets.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for Phase 2 automated image acquisition."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()
    cfg = ProjectConfig()
    cfg.ensure_directories()

    with GalaxyDatabase(cfg.db_path) as db:
        rgb_stats: dict[str, int] = {}
        fits_stats: dict[str, Any] = {}

        if not args.skip_rgb:
            rgb_downloader = SkyServerRGBDownloader(cfg)
            rgb_stats = rgb_downloader.download_for_database(
                db=db,
                limit=args.rgb_limit,
                max_workers=args.workers,
                overwrite=args.overwrite,
            )

        showcase_n = cfg.fits_showcase_count if args.fits_showcase is None else int(args.fits_showcase)
        if showcase_n > 0:
            fits_proc = MultiBandFITSProcessor(cfg)
            fits_stats = fits_proc.run_showcase(
                db=db,
                count=showcase_n,
                overwrite=args.overwrite,
            )

        db.set_metadata(
            {
                "phase2_imaging_config": cfg.imaging_metadata(),
                "phase2_rgb_stats": rgb_stats,
                "phase2_fits_processed": fits_stats.get("processed", 0),
                "phase2_last_run_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        print("\nPhase 2 Images Summary:")
        print(db.get_images_summary().to_string(index=False))


if __name__ == "__main__":
    main()
