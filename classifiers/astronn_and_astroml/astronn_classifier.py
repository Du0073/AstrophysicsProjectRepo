"""astroNN-based Galaxy10 SDSS CNN classifier with Monte Carlo Dropout uncertainty.

Uses the ``astroNN`` astronomy deep-learning package (``astroNN.datasets.galaxy10sdss``
and ``astroNN.models.Galaxy10CNN`` architecture / ``galaxy10cls_lookup``) to train on
SDSS Galaxy10 imagery, evaluate 10-class and 5-class Hubble Sequence accuracy, and run
probabilistic inference across the downloaded SDSS DR17 cutouts in ``data/galaxy_atlas.db``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from astroNN.datasets.galaxy10sdss import galaxy10cls_lookup, load_data as load_galaxy10_sdss
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from classifiers.tensorflow_cnn.hubble_taxonomy import (
    HUBBLE_CLASSES,
    aggregate_10class_probs_to_hubble,
    map_galaxy10_to_hubble,
)
from src.config import ProjectConfig
from src.database import GalaxyDatabase

logger = logging.getLogger(__name__)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def build_astronn_galaxy10_architecture(
    input_shape: tuple[int, int, int] = (69, 69, 3),
    num_classes: int = 10,
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    """Construct the canonical astroNN Galaxy10CNN 4-layer architecture with MC Dropout.

    Follows the exact ``astroNN.models.Galaxy10CNN`` layer topology (2 Conv2D + MaxPool
    stages + 2 Dense layers with dropout retained at inference for Monte Carlo Bayesian
    uncertainty estimation) in native Keras 3 / TensorFlow 2.x.
    """
    inputs = tf.keras.Input(shape=input_shape, name="galaxy_rgb_69")
    x = tf.keras.layers.Conv2D(
        16, kernel_size=(3, 3), padding="same", activation="relu", name="astronn_conv1"
    )(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="astronn_pool1")(x)
    x = tf.keras.layers.Conv2D(
        32, kernel_size=(3, 3), padding="same", activation="relu", name="astronn_conv2"
    )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="astronn_pool2")(x)
    x = tf.keras.layers.Flatten(name="astronn_flatten")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="astronn_dense1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="astronn_mc_dropout1")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="astronn_dense2")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="astronn_mc_dropout2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="galaxy10_probs")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="AstroNN_Galaxy10CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def mc_dropout_predict(
    model: tf.keras.Model,
    images: np.ndarray,
    mc_passes: int = 10,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform Monte Carlo Dropout forward passes to estimate predictive mean & uncertainty."""
    preds_stack = []
    for _ in range(mc_passes):
        batch_outputs = []
        for start in range(0, len(images), batch_size):
            batch = tf.convert_to_tensor(images[start : start + batch_size], dtype=tf.float32)
            out = model(batch, training=True).numpy()
            batch_outputs.append(out)
        preds_stack.append(np.concatenate(batch_outputs, axis=0))
    stacked = np.stack(preds_stack, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


class AstroNNGalaxy10Pipeline:
    """Train, evaluate, and run inference with the astroNN Galaxy10 SDSS classifier."""

    def __init__(
        self,
        config: ProjectConfig | None = None,
        random_state: int = 42,
    ) -> None:
        self.config = config or ProjectConfig()
        self.random_state = int(random_state)
        self.model: tf.keras.Model | None = None

    def load_dataset(
        self,
        max_samples: int | None = 12000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load Galaxy10 SDSS via ``astroNN.datasets.galaxy10sdss`` and normalize to [0, 1]."""
        images, labels = load_galaxy10_sdss()
        images = np.asarray(images, dtype=np.float32) / 255.0
        labels = np.asarray(labels, dtype=np.int64)

        if max_samples is not None and max_samples < len(labels):
            _, images, _, labels = train_test_split(
                images,
                labels,
                test_size=int(max_samples),
                stratify=labels,
                random_state=self.random_state,
            )
        return images, labels

    def train_and_evaluate(
        self,
        epochs: int = 12,
        batch_size: int = 64,
        max_samples: int | None = 12000,
        save_artifacts: bool = True,
    ) -> dict[str, Any]:
        """Train the astroNN Galaxy10CNN model and report 10-class & 5-class Hubble metrics."""
        tf.keras.utils.set_random_seed(self.random_state)
        images, labels = self.load_dataset(max_samples=max_samples)

        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=0.30, stratify=labels, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=self.random_state
        )

        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        self.model = build_astronn_galaxy10_architecture(input_shape=images.shape[1:], num_classes=10)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
        ]

        logger.info(
            "Training astroNN Galaxy10CNN on %d train / %d val SDSS images (%d epochs)...",
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

        # Evaluate deterministic & MC Dropout predictions
        mean_probs_10, std_probs_10 = mc_dropout_predict(self.model, X_test, mc_passes=8)
        y_pred_10 = np.argmax(mean_probs_10, axis=1)

        y_test_hubble = map_galaxy10_to_hubble(y_test)
        probs_hubble = aggregate_10class_probs_to_hubble(mean_probs_10)
        y_pred_hubble = np.argmax(probs_hubble, axis=1)

        cm_hubble = confusion_matrix(y_test_hubble, y_pred_hubble, labels=list(range(len(HUBBLE_CLASSES))))
        mean_epistemic_std = float(np.mean(std_probs_10[np.arange(len(y_pred_10)), y_pred_10]))

        metrics: dict[str, Any] = {
            "model_name": "astronn_galaxy10_cnn",
            "model_family": "astroNN Galaxy10CNN (4-layer Bayesian MC-Dropout CNN)",
            "dataset": "astroNN Galaxy10 SDSS (69x69 RGB)",
            "galaxy10_classes": {str(i): galaxy10cls_lookup(i) for i in range(10)},
            "hubble_classes": list(HUBBLE_CLASSES),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "epochs_run": len(history.history["loss"]),
            "test_accuracy_10class": round(float(accuracy_score(y_test, y_pred_10)), 4),
            "macro_f1_10class": round(float(f1_score(y_test, y_pred_10, average="macro")), 4),
            "test_accuracy": round(float(accuracy_score(y_test_hubble, y_pred_hubble)), 4),
            "macro_f1": round(float(f1_score(y_test_hubble, y_pred_hubble, average="macro")), 4),
            "mean_mc_epistemic_std": round(mean_epistemic_std, 4),
            "confusion_matrix": cm_hubble.tolist(),
            "history": {
                k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()
            },
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        if save_artifacts:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            self.model.save(ARTIFACTS_DIR / "astronn_galaxy10_cnn.keras")
            (ARTIFACTS_DIR / "astronn_metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            self._plot_diagnostics(history.history, cm_hubble)

        return metrics

    def predict_catalog_cutouts(
        self,
        db: GalaxyDatabase,
        mc_passes: int = 8,
    ) -> int:
        """Run MC-Dropout inference on all downloaded RGB cutouts and upsert to SQLite."""
        if self.model is None:
            model_path = ARTIFACTS_DIR / "astronn_galaxy10_cnn.keras"
            if not model_path.exists():
                raise FileNotFoundError(f"Trained astroNN model not found at {model_path}")
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

        batch = np.stack(tensors, axis=0)
        mean_p10, std_p10 = mc_dropout_predict(self.model, batch, mc_passes=mc_passes)
        probs_hubble = aggregate_10class_probs_to_hubble(mean_p10)

        now = datetime.now(timezone.utc).isoformat()
        records = []
        for i, obj_id in enumerate(obj_ids):
            best_h = int(np.argmax(probs_hubble[i]))
            best_10 = int(np.argmax(mean_p10[i]))
            prob_map = {
                cls: round(float(probs_hubble[i, j]), 4)
                for j, cls in enumerate(HUBBLE_CLASSES)
            }
            prob_map["_galaxy10_top_class"] = galaxy10cls_lookup(best_10)
            prob_map["_mc_uncertainty_std"] = round(float(std_p10[i, best_10]), 4)
            records.append(
                {
                    "objID": obj_id,
                    "model_name": "astronn_galaxy10_cnn",
                    "model_version": "v1",
                    "predicted_class": HUBBLE_CLASSES[best_h],
                    "confidence": round(float(probs_hubble[i, best_h]), 4),
                    "class_probabilities_json": prob_map,
                    "predicted_at": now,
                }
            )

        upserted = db.upsert_predictions(records, predicted_at=now)
        logger.info("Upserted %d astroNN Galaxy10CNN predictions into SQLite.", upserted)
        return upserted

    def _plot_diagnostics(self, history: dict[str, list[float]], cm_hubble: np.ndarray) -> None:
        """Save training curves and 5-class Hubble confusion matrix figure."""
        self.config.ensure_directories()
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

        epochs_x = range(1, len(history["accuracy"]) + 1)
        axes[0].plot(epochs_x, history["accuracy"], marker="o", label="Train Accuracy")
        axes[0].plot(epochs_x, history["val_accuracy"], marker="s", label="Val Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("10-Class Accuracy")
        axes[0].set_title("astroNN Galaxy10CNN — Learning Curves")
        axes[0].legend(frameon=False)
        axes[0].grid(alpha=0.25)

        short_labels = ["Elliptical", "S0 / Edge-on", "Tight Sa", "Medium Sb", "Loose Sc/Irr"]
        sns.heatmap(
            cm_hubble,
            annot=True,
            fmt="d",
            cmap="Purples",
            xticklabels=short_labels,
            yticklabels=short_labels,
            ax=axes[1],
        )
        axes[1].set_xlabel("Predicted Hubble Class")
        axes[1].set_ylabel("True Hubble Class")
        axes[1].set_title("astroNN Galaxy10CNN — Hubble Sequence Confusion Matrix")

        fig.tight_layout()
        fig.savefig(self.config.figures_dir / "astronn_cnn_evaluation.png", dpi=150)
        plt.close(fig)


def main() -> None:
    """CLI entry point for the astroNN Galaxy10 SDSS CNN pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Train astroNN Galaxy10CNN and classify SDSS cutouts.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=12000)
    args = parser.parse_args()

    cfg = ProjectConfig()
    pipe = AstroNNGalaxy10Pipeline(cfg)
    metrics = pipe.train_and_evaluate(epochs=args.epochs, max_samples=args.max_samples)
    with GalaxyDatabase(cfg.db_path) as db:
        n_pred = pipe.predict_catalog_cutouts(db)
        metrics["n_catalog_predicted"] = n_pred
        db.set_metadata({"phase3_astronn_metrics": metrics})
    (ARTIFACTS_DIR / "astronn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"astroNN Galaxy10CNN -> 5-Class Hubble Accuracy: {metrics['test_accuracy']:.4f} | "
        f"10-Class Accuracy: {metrics['test_accuracy_10class']:.4f} | "
        f"Catalog Predicted: {n_pred}"
    )


if __name__ == "__main__":
    main()
