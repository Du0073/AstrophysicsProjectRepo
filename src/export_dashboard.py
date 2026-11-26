"""Phase 4 — Export normalized SQLite views, predictions, and model metrics to web JSON/JS."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis import (
    classify_color_population,
    compute_correlation_matrix,
    fit_mass_size_power_law,
    PHYSICAL_CORR_LABELS,
)
from .config import ProjectConfig
from .database import GalaxyDatabase

logger = logging.getLogger(__name__)


def _round_opt(val: Any, digits: int = 4) -> float | None:
    if val is None or pd.isna(val):
        return None
    return round(float(val), digits)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def export_atlas_dashboard(
    config: ProjectConfig | None = None,
    web_dir: Path | None = None,
    copy_images: bool = True,
) -> dict[str, Any]:
    """Export SQLite ``v_galaxy_dashboard``, predictions, images, and metrics for ``web/``.

    Produces both:
    - ``web/data/atlas_data.json`` (standard JSON payload)
    - ``web/data/atlas_data.js`` (``window.ATLAS_DATA = ...`` for zero-CORS ``file://`` viewing)
    """
    cfg = config or ProjectConfig()
    out_web = Path(web_dir) if web_dir is not None else (cfg.project_root / "web")
    data_dir = out_web / "data"
    rgb_web_dir = out_web / "assets" / "rgb_cutouts"
    lupton_web_dir = out_web / "assets" / "lupton_rgb"
    data_dir.mkdir(parents=True, exist_ok=True)
    if copy_images:
        rgb_web_dir.mkdir(parents=True, exist_ok=True)
        lupton_web_dir.mkdir(parents=True, exist_ok=True)

    with GalaxyDatabase(cfg.db_path) as db:
        df_gal = db.query_df(
            """
            SELECT
                objID, ra, dec, redshift, abs_mag_r, color_gr, color_ug,
                log_stellar_mass, log_sfr, re_kpc, concentration, frac_dev_r,
                p_el_debiased, p_cs_debiased, morph_proxy, rgb_image_path
            FROM v_galaxy_dashboard
            ORDER BY objID
            """
        )
        df_lupton = db.query_df(
            "SELECT objID, file_path FROM images WHERE image_type = 'lupton_rgb'"
        )
        df_preds = db.query_df(
            """
            SELECT objID, model_name, predicted_class, confidence, class_probabilities_json
            FROM predictions
            ORDER BY objID, model_name
            """
        )
        meta = db.get_metadata()
        table_counts = db.get_table_counts()
        images_summary = db.get_images_summary().to_dict(orient="records")

    annotated = classify_color_population(df_gal)
    fits_out = fit_mass_size_power_law(annotated)
    corr_df = compute_correlation_matrix(annotated)
    crosstab = pd.crosstab(annotated["morph_proxy"], annotated["color_population"]).reindex(
        index=["early", "late"],
        columns=["red_sequence", "blue_cloud"],
        fill_value=0,
    )

    lupton_map: dict[int, str] = {}
    for row in df_lupton.itertuples(index=False):
        src_path = cfg.project_root / row.file_path
        if src_path.exists():
            if copy_images:
                dst = lupton_web_dir / src_path.name
                if not dst.exists() or dst.stat().st_size != src_path.stat().st_size:
                    shutil.copy2(src_path, dst)
            lupton_map[int(row.objID)] = f"assets/lupton_rgb/{src_path.name}"

    model_short_key = {
        "rf_photometric_baseline": "rf",
        "astronn_galaxy10_cnn": "astronn",
        "tensorflow_hubble_cnn": "tf",
    }
    preds_by_obj: dict[int, dict[str, Any]] = {}
    for row in df_preds.itertuples(index=False):
        oid = int(row.objID)
        short = model_short_key.get(str(row.model_name), str(row.model_name))
        probs: dict[str, Any] = {}
        if row.class_probabilities_json:
            try:
                probs = json.loads(row.class_probabilities_json)
            except Exception:  # noqa: BLE001
                probs = {}
        preds_by_obj.setdefault(oid, {})[short] = {
            "cls": str(row.predicted_class),
            "conf": _round_opt(row.confidence, 4),
            "probs": probs,
        }

    galaxies_payload: list[dict[str, Any]] = []
    showcase_payload: list[dict[str, Any]] = []
    copied_rgb = 0

    for row in annotated.itertuples(index=False):
        oid = int(row.objID)
        img_rel: str | None = None
        if pd.notna(row.rgb_image_path):
            src_rgb = cfg.project_root / str(row.rgb_image_path)
            if src_rgb.exists():
                if copy_images:
                    dst_rgb = rgb_web_dir / src_rgb.name
                    if not dst_rgb.exists() or dst_rgb.stat().st_size != src_rgb.stat().st_size:
                        shutil.copy2(src_rgb, dst_rgb)
                img_rel = f"assets/rgb_cutouts/{src_rgb.name}"
                copied_rgb += 1

        lup_rel = lupton_map.get(oid)
        rec = {
            "id": str(oid),
            "ra": _round_opt(row.ra, 4),
            "dec": _round_opt(row.dec, 4),
            "z": _round_opt(row.redshift, 4),
            "mr": _round_opt(row.abs_mag_r, 3),
            "gr": _round_opt(row.color_gr, 3),
            "ug": _round_opt(row.color_ug, 3),
            "lm": _round_opt(row.log_stellar_mass, 3),
            "lsfr": _round_opt(row.log_sfr, 3),
            "re": _round_opt(row.re_kpc, 3),
            "c": _round_opt(row.concentration, 3),
            "fdev": _round_opt(row.frac_dev_r, 3),
            "mp": str(row.morph_proxy),
            "cp": str(row.color_population),
            "pel": _round_opt(row.p_el_debiased, 3),
            "pcs": _round_opt(row.p_cs_debiased, 3),
            "img": img_rel,
            "lup": lup_rel,
            "pred": preds_by_obj.get(oid, {}),
        }
        galaxies_payload.append(rec)
        if lup_rel is not None:
            showcase_payload.append(rec)

    # Load model evaluation metrics from artifact JSONs (with DB metadata fallback)
    rf_metrics = _load_json_if_exists(
        cfg.project_root / "classifiers" / "astronn_and_astroml" / "artifacts" / "rf_metrics.json"
    )
    astronn_metrics = _load_json_if_exists(
        cfg.project_root / "classifiers" / "astronn_and_astroml" / "artifacts" / "astronn_metrics.json"
    )
    tf_metrics = _load_json_if_exists(
        cfg.project_root / "classifiers" / "tensorflow_cnn" / "artifacts" / "tf_cnn_metrics.json"
    )

    models_payload: dict[str, Any] = {}
    if rf_metrics:
        models_payload["rf"] = rf_metrics
    if astronn_metrics:
        models_payload["astronn"] = astronn_metrics
    if tf_metrics:
        models_payload["tf"] = tf_metrics

    cosmology_meta = cfg.cosmology_metadata()
    if "cosmology" in meta:
        try:
            cosmology_meta = json.loads(meta["cosmology"])
        except Exception:  # noqa: BLE001
            pass

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "title": "SDSS Galaxy Morphology & Scaling Relations Atlas",
            "data_release": f"SDSS DR{cfg.data_release}",
            "cosmology": cosmology_meta,
            "quality_cuts": cfg.quality_cuts_metadata(),
            "table_counts": table_counts,
            "images_summary": images_summary,
        },
        "summary": {
            "total_galaxies": int(len(annotated)),
            "early_count": int((annotated["morph_proxy"] == "early").sum()),
            "late_count": int((annotated["morph_proxy"] == "late").sum()),
            "rgb_cutouts_count": int(copied_rgb),
            "fits_showcase_count": int(len(showcase_payload)),
            "zoo_matched_count": int(annotated["p_el_debiased"].notna().sum()),
            "color_population_crosstab": {
                m: {c: int(crosstab.loc[m, c]) for c in crosstab.columns} for m in crosstab.index
            },
            "mass_size_fits": fits_out,
            "correlation_matrix": {
                "columns": list(corr_df.columns),
                "labels": [PHYSICAL_CORR_LABELS.get(c, c) for c in corr_df.columns],
                "values": [[round(float(v), 3) for v in row] for row in corr_df.to_numpy()],
            },
        },
        "models": models_payload,
        "showcase": showcase_payload,
        "galaxies": galaxies_payload,
    }

    json_text = json.dumps(payload, separators=(",", ":"))
    json_path = data_dir / "atlas_data.json"
    js_path = data_dir / "atlas_data.js"
    json_path.write_text(json_text, encoding="utf-8")
    js_path.write_text(f"window.ATLAS_DATA = {json_text};\n", encoding="utf-8")

    logger.info(
        "Exported %d galaxies (%d with RGB, %d FITS showcase) -> %s (%.1f KB)",
        len(galaxies_payload),
        copied_rgb,
        len(showcase_payload),
        json_path,
        json_path.stat().st_size / 1024.0,
    )
    return {
        "json_path": str(json_path),
        "js_path": str(js_path),
        "size_kb": round(json_path.stat().st_size / 1024.0, 1),
        "galaxies": len(galaxies_payload),
        "rgb_count": copied_rgb,
        "showcase_count": len(showcase_payload),
        "models": list(models_payload.keys()),
    }


def main() -> None:
    """CLI entry point for Phase 4 dashboard data export."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Export SQLite galaxy atlas to web JSON/JS.")
    parser.add_argument("--no-copy-images", action="store_true")
    args = parser.parse_args()
    res = export_atlas_dashboard(copy_images=not args.no_copy_images)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
