"""Tabular photometric galaxy morphology baseline (scikit-learn / astroML paradigm).

Uses physical and photometric features from ``data/galaxy_atlas.db`` (excluding
``frac_dev_r`` to prevent leakage against the structural proxy) to train
Random Forest and Gradient Boosting baselines, evaluate against Galaxy Zoo
consensus labels, and persist predictions into SQLite ``predictions``.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.config import ProjectConfig
from src.database import GalaxyDatabase

logger = logging.getLogger(__name__)

# Photometric & physical features (deliberately excludes frac_dev_r to prevent target leakage)
PHOTOMETRIC_FEATURES: tuple[str, ...] = (
    "color_gr",
    "color_ug",
    "abs_mag_r",
    "re_kpc",
    "concentration",
    "log_stellar_mass",
    "log_sfr",
    "redshift",
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


class PhotometricMorphologyClassifier:
    """Classical photometric galaxy morphology classifier using scikit-learn."""

    def __init__(
        self,
        config: ProjectConfig | None = None,
        random_state: int = 42,
    ) -> None:
        self.config = config or ProjectConfig()
        self.random_state = int(random_state)
        self.rf_model = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.gb_model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.06,
            random_state=self.random_state,
        )

    @staticmethod
    def build_galaxy_zoo_labels(df: pd.DataFrame, threshold: float = 0.60) -> pd.DataFrame:
        """Create high-confidence Galaxy Zoo ground-truth labels (Elliptical vs. Spiral).

        Galaxies with ``p_el_debiased >= threshold`` are labeled ``Elliptical``;
        galaxies with ``p_cs_debiased >= threshold`` are labeled ``Spiral``.
        Falls back to ``morph_proxy`` only if Galaxy Zoo votes are absent.
        """
        zoo_valid = df.dropna(subset=["p_el_debiased", "p_cs_debiased"]).copy()
        conditions = [
            zoo_valid["p_el_debiased"] >= threshold,
            zoo_valid["p_cs_debiased"] >= threshold,
        ]
        zoo_valid["target_label"] = np.select(conditions, ["Elliptical", "Spiral"], default="")
        labeled = zoo_valid.loc[zoo_valid["target_label"] != ""].copy()
        if len(labeled) < 30:
            fallback = df.copy()
            fallback["target_label"] = np.where(
                fallback["morph_proxy"] == "early", "Elliptical", "Spiral"
            )
            return fallback
        return labeled

    def train_and_evaluate(
        self,
        db: GalaxyDatabase,
        save_artifacts: bool = True,
        write_predictions: bool = True,
    ) -> dict[str, Any]:
        """Train Random Forest & Gradient Boosting baselines, evaluate, and save to SQLite."""
        df_all = db.query_df(
            """
            SELECT objID, redshift, abs_mag_r, color_gr, color_ug,
                   log_stellar_mass, log_sfr, re_kpc, concentration,
                   frac_dev_r, p_el_debiased, p_cs_debiased, morph_proxy
            FROM galaxies
            ORDER BY objID
            """
        )
        if df_all.empty:
            raise ValueError("Database galaxies table is empty.")

        labeled = self.build_galaxy_zoo_labels(df_all)
        X = labeled[list(PHOTOMETRIC_FEATURES)].to_numpy(dtype=float)
        y = labeled["target_label"].to_numpy()
        class_names = sorted(np.unique(y).tolist())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=self.random_state
        )

        # Fit Random Forest (primary tabular baseline) and Gradient Boosting (benchmark)
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)

        rf_preds = self.rf_model.predict(X_test)
        rf_proba = self.rf_model.predict_proba(X_test)
        gb_preds = self.gb_model.predict(X_test)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        rf_cv_acc = cross_val_score(self.rf_model, X, y, cv=cv, scoring="accuracy")

        pos_idx = list(self.rf_model.classes_).index("Spiral")
        y_test_bin = (y_test == "Spiral").astype(int)
        rf_auc = float(roc_auc_score(y_test_bin, rf_proba[:, pos_idx]))

        cm = confusion_matrix(y_test, rf_preds, labels=class_names)
        importances = {
            feat: round(float(imp), 4)
            for feat, imp in sorted(
                zip(PHOTOMETRIC_FEATURES, self.rf_model.feature_importances_),
                key=lambda kv: kv[1],
                reverse=True,
            )
        }

        metrics: dict[str, Any] = {
            "model_name": "rf_photometric_baseline",
            "model_family": "scikit-learn RandomForest (astroML photometric baseline)",
            "target_source": "Galaxy Zoo debiased consensus (p >= 0.60)",
            "features": list(PHOTOMETRIC_FEATURES),
            "classes": class_names,
            "n_labeled": int(len(labeled)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_accuracy": round(float(accuracy_score(y_test, rf_preds)), 4),
            "macro_f1": round(float(f1_score(y_test, rf_preds, average="macro")), 4),
            "roc_auc": round(rf_auc, 4),
            "cv5_accuracy_mean": round(float(rf_cv_acc.mean()), 4),
            "cv5_accuracy_std": round(float(rf_cv_acc.std()), 4),
            "gb_benchmark_accuracy": round(float(accuracy_score(y_test, gb_preds)), 4),
            "confusion_matrix": cm.tolist(),
            "feature_importances": importances,
            "classification_report": classification_report(
                y_test, rf_preds, labels=class_names, output_dict=True
            ),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Predict on ALL catalog galaxies and upsert into SQLite predictions table
        if write_predictions:
            X_all = df_all[list(PHOTOMETRIC_FEATURES)].to_numpy(dtype=float)
            all_proba = self.rf_model.predict_proba(X_all)
            rf_classes = list(self.rf_model.classes_)
            now = datetime.now(timezone.utc).isoformat()
            records = []
            for row_idx, obj_id in enumerate(df_all["objID"].astype("int64")):
                probs = {cls: round(float(all_proba[row_idx, j]), 4) for j, cls in enumerate(rf_classes)}
                best_idx = int(np.argmax(all_proba[row_idx]))
                records.append(
                    {
                        "objID": int(obj_id),
                        "model_name": "rf_photometric_baseline",
                        "model_version": "v1",
                        "predicted_class": rf_classes[best_idx],
                        "confidence": round(float(all_proba[row_idx, best_idx]), 4),
                        "class_probabilities_json": probs,
                        "predicted_at": now,
                    }
                )
            db.upsert_predictions(records, predicted_at=now)
            db.set_metadata({"phase3_rf_metrics": metrics})
            metrics["n_catalog_predicted"] = len(records)

        if save_artifacts:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            with (ARTIFACTS_DIR / "rf_photometric_baseline.pkl").open("wb") as fh:
                pickle.dump(self.rf_model, fh)
            (ARTIFACTS_DIR / "rf_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            self._plot_diagnostics(cm, class_names, importances)

        return metrics

    def _plot_diagnostics(
        self,
        cm: np.ndarray,
        class_names: list[str],
        importances: dict[str, float],
    ) -> None:
        """Save confusion matrix + feature importance figure to outputs/figures/."""
        self.config.ensure_directories()
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0],
        )
        axes[0].set_xlabel("Predicted Class")
        axes[0].set_ylabel("Galaxy Zoo True Class")
        axes[0].set_title("Random Forest Photometric Baseline — Confusion Matrix")

        feats = list(importances.keys())[::-1]
        vals = [importances[f] for f in feats]
        axes[1].barh(feats, vals, color="#2471a3", edgecolor="white")
        axes[1].set_xlabel("Gini Feature Importance")
        axes[1].set_title("Most Informative Physical & Photometric Features")
        axes[1].grid(alpha=0.25, axis="x")

        fig.tight_layout()
        fig.savefig(self.config.figures_dir / "rf_confusion_and_importance.png", dpi=150)
        plt.close(fig)


def main() -> None:
    """CLI entry point for the tabular photometric Random Forest classifier."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Train scikit-learn photometric morphology baseline.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ProjectConfig()
    clf = PhotometricMorphologyClassifier(cfg, random_state=args.seed)
    with GalaxyDatabase(cfg.db_path) as db:
        metrics = clf.train_and_evaluate(db)
    print(
        f"Random Forest Test Accuracy: {metrics['test_accuracy']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | "
        f"5-fold CV: {metrics['cv5_accuracy_mean']:.4f} ± {metrics['cv5_accuracy_std']:.4f}"
    )


if __name__ == "__main__":
    main()
