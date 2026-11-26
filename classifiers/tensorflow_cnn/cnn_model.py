"""Custom TensorFlow / Keras Convolutional Neural Network for Hubble Sequence classification.

Architecture directly extends the official TensorFlow CNN tutorial
(https://www.tensorflow.org/tutorials/images/cnn) for astronomical imaging:
- Built-in spatial-invariance augmentation (random horizontal/vertical flips + 360° rotations)
  because galaxies have no preferred orientation on the sky.
- Progressive Conv2D + BatchNormalization + ReLU + MaxPooling2D blocks.
- GlobalAveragePooling2D + Dropout regularization + Softmax classification over the
  5 Hubble Sequence categories.
"""

from __future__ import annotations

import tensorflow as tf


def build_hubble_cnn(
    input_shape: tuple[int, int, int] = (69, 69, 3),
    num_classes: int = 5,
    learning_rate: float = 1e-3,
    use_augmentation: bool = True,
) -> tf.keras.Model:
    """Build and compile a custom TensorFlow/Keras CNN for galaxy morphology."""
    layers: list[tf.keras.layers.Layer] = [
        tf.keras.layers.Input(shape=input_shape, name="sdss_rgb_input"),
    ]

    if use_augmentation:
        layers.extend(
            [
                tf.keras.layers.RandomFlip(
                    "horizontal_and_vertical", name="astro_random_flip"
                ),
                tf.keras.layers.RandomRotation(
                    factor=0.5, fill_mode="reflect", name="astro_random_rot360"
                ),
            ]
        )

    # Convolutional Base (inspired by TensorFlow CNN tutorial + BatchNorm for stability)
    layers.extend(
        [
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv_block1_a"),
            tf.keras.layers.BatchNormalization(name="bn_block1_a"),
            tf.keras.layers.ReLU(name="relu_block1_a"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool_block1"),

            tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv_block2_a"),
            tf.keras.layers.BatchNormalization(name="bn_block2_a"),
            tf.keras.layers.ReLU(name="relu_block2_a"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool_block2"),

            tf.keras.layers.Conv2D(96, (3, 3), padding="same", use_bias=False, name="conv_block3_a"),
            tf.keras.layers.BatchNormalization(name="bn_block3_a"),
            tf.keras.layers.ReLU(name="relu_block3_a"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool_block3"),
            tf.keras.layers.Dropout(0.25, name="dropout_conv"),

            tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv_block4_a"),
            tf.keras.layers.BatchNormalization(name="bn_block4_a"),
            tf.keras.layers.ReLU(name="relu_block4_a"),

            # Dense Classification Head
            tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_features"),
            tf.keras.layers.Dropout(0.40, name="dropout_head"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="hubble_probs"),
        ]
    )

    model = tf.keras.Sequential(layers, name="TensorFlow_Hubble_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
