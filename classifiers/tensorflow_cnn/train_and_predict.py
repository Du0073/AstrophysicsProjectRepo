"""Training, evaluation, and SDSS catalog inference pipeline for the custom TensorFlow CNN."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from astroNN.datasets.galaxy10sdss import load_data as load_galaxy10_sdss
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import ProjectConfig
from src.database import GalaxyDatabase

from .cnn_model import build_hubble_cnn
from .hubble_taxonomy import HUBBLE_CLASSES, map_galaxy10_to_hubble

logger = logging.getLogger(__name__)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


class TensorFlowHubbleTrainer:
    """End-to-end trainer and catalog predictor for the custom TensorFlow Hubble CNN."""

    def __init__(
        self,
        config: ProjectConfig | None = None,
        random_state: int = 42,
    ) -> None:
        self.config = config or ProjectConfig()
        self.random_state = int(random_state)
        self.model: tf.keras.Model | None = None

    def load_hubble_dataset(
        self,
        max_samples: int | None = 12000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load SDSS Galaxy10 images and map labels to the 5 Hubble Sequence categories."""
        images, labels_10 = load_galaxy10_sdss()
        images = np.asarray(images, dtype=np.float32) / 255.0
        labels_5 = map_galaxy10_to_hubble(labels_10)

        if max_samples is not None and max_samples < len(labels_5):
            _, images, _, labels_5 = train_test_split(
                images,
                labels_5,
                test_size=int(max_samples),
                stratify=labels_5,
                random_state=self.random_state,
            )
        return images, labels_5

    def train_and_evaluate(
        self,
        epochs: int = 14,
        batch_size: int = 64,
        max_samples: int | None = 12000,
        save_artifacts: bool = True,
    ) -> dict[str, Any]:
        """Train the custom TensorFlow CNN with class balancing and astronomical augmentation."""
        tf.keras.utils.set_random_seed(self.random_state)
        images, labels = self.load_hubble_dataset(max_samples=max_samples)

        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=0.30, stratify=labels, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=self.random_state
        )

        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        self.model = build_hubble_cnn(
            input_shape=images.shape[1:],
            num_classes=len(HUBBLE_CLASSES),
            learning_rate=1e-3,
            use_augmentation=True,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
            ),
        ]

        logger.info(
            "Training custom TensorFlow Hubble CNN on %d train / %d val images (%d epochs)...",
            len(X_train),
            len(X_val),
            epochs,
        )
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2,
        )

        probs_test = self.model.predict(X_test, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(probs_test, axis=1)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(HUBBLE_CLASSES))))

        metrics: dict[str, Any] = {
            "model_name": "tensorflow_hubble_cnn",
            "model_family": "Custom TensorFlow/Keras CNN (4-block + 360° Astro Augmentation)",
            "dataset": "SDSS Galaxy10 -> 5-Class Hubble Sequence (69x69 RGB)",
            "hubble_classes": list(HUBBLE_CLASSES),
            "n_parameters": int(self.model.count_params()),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "epochs_run": len(history.history["loss"]),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred,
                labels=list(range(len(HUBBLE_CLASSES))),
                target_names=list(HUBBLE_CLASSES),
                output_dict=True,
                zero_division=0,
            ),
            "history": {
                k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()
            },
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        if save_artifacts:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            self.model.save(ARTIFACTS_DIR / "tensorflow_hubble_cnn.keras")
            (ARTIFACTS_DIR / "tf_cnn_metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            self._plot_diagnostics(history.history, cm)

        return metrics

    def predict_catalog_cutouts(self, db: GalaxyDatabase, batch_size: int = 64) -> int:
        """Classify all downloaded SDSS DR17 RGB cutouts and store predictions in SQLite."""
        if self.model is None:
            model_path = ARTIFACTS_DIR / "tensorflow_hubble_cnn.keras"
            if not model_path.exists():
                raise FileNotFoundError(f"Trained TensorFlow CNN not found at {model_path}")
            self.model = tf.keras.models.load_model(model_path)

        df_img = db.query_df(
            """
            SELECT objID, file_path
            FROM images
            WHERE image_type = 'rgb_cutout'
            ORDER BY objID
            """
        )
        if df_img.empty:
            logger.warning("No rgb_cutout images registered in database.")
            return 0

        obj_ids: list[int] = []
        tensors: list[np.ndarray] = []
        for row in df_img.itertuples(index=False):
            full_path = self.config.project_root / row.file_path
            if not full_path.exists():
                continue
            with Image.open(full_path) as img:
                resized = img.convert("RGB").resize((69, 69), Image.Resampling.BILINEAR)
                tensors.append(np.asarray(resized, dtype=np.float32) / 255.0)
                obj_ids.append(int(row.objID))

        if not tensors:
            return 0

        probs = self.model.predict(np.stack(tensors, axis=0), batch_size=batch_size, verbose=0)
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for i, obj_id in enumerate(obj_ids):
            best = int(np.argmax(probs[i]))
            prob_map = {
                cls: round(float(probs[i, j]), 4)
                for j, cls in enumerate(HUBBLE_CLASSES)
            }
            records.append(
                {
                    "objID": obj_id,
                    "model_name": "tensorflow_hubble_cnn",
                    "model_version": "v1",
                    "predicted_class": HUBBLE_CLASSES[best],
                    "confidence": round(float(probs[i, best]), 4),
                    "class_probabilities_json": prob_map,
                    "predicted_at": now,
                }
            )

        upserted = db.upsert_predictions(records, predicted_at=now)
        logger.info("Upserted %d custom TensorFlow CNN predictions into SQLite.", upserted)
        return upserted

    def _plot_diagnostics(self, history: dict[str, list[float]], cm: np.ndarray) -> None:
        """Save training/validation curves and Hubble Sequence confusion matrix."""
        self.config.ensure_directories()
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

        epochs_x = range(1, len(history["accuracy"]) + 1)
        axes[0].plot(epochs_x, history["accuracy"], marker="o", label="Train")
        axes[0].plot(epochs_x, history["val_accuracy"], marker="s", label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Custom TensorFlow CNN — Accuracy")
        axes[0].legend(frameon=False)
        axes[0].grid(alpha=0.25)

        axes[1].plot(epochs_x, history["loss"], marker="o", label="Train")
        axes[1].plot(epochs_x, history["val_loss"], marker="s", label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Sparse Categorical Cross-Entropy")
        axes[1].set_title("Custom TensorFlow CNN — Loss")
        axes[1].legend(frameon=False)
        axes[1].grid(alpha=0.25)

        short_labels = ["Elliptical", "S0 / Edge-on", "Tight Sa", "Medium Sb", "Loose Sc/Irr"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=short_labels,
            yticklabels=short_labels,
            ax=axes[2],
        )
        axes[2].set_xlabel("Predicted Hubble Class")
        axes[2].set_ylabel("True Hubble Class")
        axes[2].set_title("Hubble Sequence Confusion Matrix")

        fig.tight_layout()
        fig.savefig(self.config.figures_dir / "tf_cnn_evaluation.png", dpi=150)
        plt.close(fig)


def main() -> None:
    """CLI entry point for the custom TensorFlow Hubble Sequence CNN."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Train custom TensorFlow Hubble CNN and classify SDSS cutouts.")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--max-samples", type=int, default=12000)
    args = parser.parse_args()

    cfg = ProjectConfig()
    trainer = TensorFlowHubbleTrainer(cfg)
    metrics = trainer.train_and_evaluate(epochs=args.epochs, max_samples=args.max_samples)
    with GalaxyDatabase(cfg.db_path) as db:
        n_pred = trainer.predict_catalog_cutouts(db)
        metrics["n_catalog_predicted"] = n_pred
        db.set_metadata({"phase3_tf_cnn_metrics": metrics})
    (ARTIFACTS_DIR / "tf_cnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Custom TensorFlow CNN -> Test Accuracy: {metrics['test_accuracy']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f} | Params: {metrics['n_parameters']:,} | "
        f"Catalog Predicted: {n_pred}"
    )


if __name__ == "__main__":
    main()
