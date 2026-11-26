"""Unit tests for statistical dashboard, both classifier tracks, and Phase 4 web export."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from classifiers.astronn_and_astroml.astronn_classifier import (
    build_astronn_galaxy10_architecture,
    mc_dropout_predict,
)
from classifiers.astronn_and_astroml.tabular_baseline import PhotometricMorphologyClassifier
from classifiers.tensorflow_cnn.cnn_model import build_hubble_cnn
from classifiers.tensorflow_cnn.hubble_taxonomy import (
    HUBBLE_CLASSES,
    aggregate_10class_probs_to_hubble,
    map_galaxy10_to_hubble,
)
from src.acquisition import SDSSCatalogFetcher
from src.analysis import (
    classify_color_population,
    compute_correlation_matrix,
    fit_mass_size_power_law,
    plot_statistical_dashboard,
)
from src.config import ProjectConfig
from src.database import GalaxyDatabase
from src.export_dashboard import export_atlas_dashboard


def _make_synthetic_catalog(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        is_early = i < n // 2
        z = float(rng.uniform(0.025, 0.095))
        mr = float(rng.uniform(14.5, 17.2))
        gr = float(rng.normal(0.92 if is_early else 0.58, 0.06))
        ug = float(rng.normal(1.65 if is_early else 1.15, 0.10))
        r50 = float(rng.uniform(2.0, 4.0) if is_early else rng.uniform(3.2, 5.5))
        c = float(rng.uniform(2.7, 3.3) if is_early else rng.uniform(1.9, 2.4))
        rows.append(
            {
                "objID": 1237648702972360000 + i,
                "ra": 180.0 + 0.01 * i,
                "dec": 0.5 + 0.01 * i,
                "z": z,
                "modelMag_u": mr + gr + ug,
                "modelMag_g": mr + gr,
                "modelMag_r": mr,
                "modelMag_i": mr - 0.35,
                "modelMag_z": mr - 0.60,
                "petroRad_r": r50 * 2.3,
                "petroR50_r": r50,
                "petroR90_r": r50 * c,
                "fracDeV_r": 0.85 if is_early else 0.15,
                "lgm_tot_p50": float(rng.normal(10.7 if is_early else 9.9, 0.25)),
                "sfr_tot_p50": float(rng.normal(-1.0 if is_early else 0.2, 0.35)),
                "p_el_debiased": 0.88 if is_early else 0.10,
                "p_cs_debiased": 0.08 if is_early else 0.85,
            }
        )
    return SDSSCatalogFetcher(ProjectConfig()).clean_and_derive(pd.DataFrame(rows))


def test_statistical_fits_and_dashboard_generation(tmp_path: Path) -> None:
    df = _make_synthetic_catalog(60)
    annotated = classify_color_population(df)
    assert set(annotated["color_population"].unique()).issubset({"red_sequence", "blue_cloud"})

    fits = fit_mass_size_power_law(df)
    assert {"early", "late"} == set(fits.keys())
    assert 0.0 <= fits["early"]["r_squared"] <= 1.0

    corr = compute_correlation_matrix(df)
    assert corr.shape == (9, 9)

    png_out = tmp_path / "dash.png"
    pdf_out = tmp_path / "dash.pdf"
    fig, metrics = plot_statistical_dashboard(df, output_png=png_out, output_pdf=pdf_out)
    assert png_out.exists() and png_out.stat().st_size > 10_000
    assert pdf_out.exists() and pdf_out.stat().st_size > 10_000
    assert metrics["n_total"] == 60
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_hubble_taxonomy_and_cnn_architectures() -> None:
    mapped = map_galaxy10_to_hubble(np.arange(10))
    assert mapped.shape == (10,)
    assert set(mapped.tolist()) == set(range(len(HUBBLE_CLASSES)))

    p10 = np.full((3, 10), 0.1, dtype=np.float32)
    p5 = aggregate_10class_probs_to_hubble(p10)
    assert p5.shape == (3, 5)
    assert np.allclose(p5.sum(axis=1), 1.0)

    dummy_imgs = np.random.default_rng(0).random((4, 69, 69, 3), dtype=np.float32)

    tf_cnn = build_hubble_cnn(input_shape=(69, 69, 3), num_classes=5, use_augmentation=True)
    out_tf = tf_cnn.predict(dummy_imgs, verbose=0)
    assert out_tf.shape == (4, 5)
    assert np.allclose(out_tf.sum(axis=1), 1.0, atol=1e-4)

    astro_cnn = build_astronn_galaxy10_architecture(input_shape=(69, 69, 3), num_classes=10)
    mean_p, std_p = mc_dropout_predict(astro_cnn, dummy_imgs, mc_passes=3, batch_size=4)
    assert mean_p.shape == (4, 10)
    assert std_p.shape == (4, 10)
    assert np.allclose(mean_p.sum(axis=1), 1.0, atol=1e-4)


def test_tabular_rf_predictions_and_phase4_export(tmp_path: Path) -> None:
    cfg = ProjectConfig(project_root=tmp_path)
    cfg.ensure_directories()
    df = _make_synthetic_catalog(80)

    with GalaxyDatabase(cfg.db_path) as db:
        db.upsert_galaxies(df)
        clf = PhotometricMorphologyClassifier(cfg, random_state=42)
        metrics = clf.train_and_evaluate(db, save_artifacts=False, write_predictions=True)
        assert metrics["test_accuracy"] >= 0.85
        assert db.get_table_counts()["predictions"] == 80

    res = export_atlas_dashboard(config=cfg, web_dir=tmp_path / "web", copy_images=False)
    assert res["galaxies"] == 80
    payload = json.loads(Path(res["json_path"]).read_text(encoding="utf-8"))
    assert len(payload["galaxies"]) == 80
    # 64-bit objID must be exported as exact string for JS safety
    assert payload["galaxies"][0]["id"] == "1237648702972360000"
    assert payload["galaxies"][0]["pred"]["rf"]["cls"] in {"Elliptical", "Spiral"}
    assert Path(res["js_path"]).read_text(encoding="utf-8").startswith("window.ATLAS_DATA = ")
