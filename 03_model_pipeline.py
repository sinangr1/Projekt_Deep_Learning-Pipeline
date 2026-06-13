from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_data_augmentation() -> tf.keras.Sequential:
    """Create data augmentation pipeline."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.10, 0.10),
        ],
        name="data_augmentation",
    )


def build_baseline_model(
    img_size: tuple[int, int] = (160, 160),
    data_augmentation: tf.keras.Sequential | None = None,
) -> tf.keras.Model:
    """Build the baseline CNN model."""
    data_augmentation = data_augmentation or build_data_augmentation()
    input_shape = (img_size[0], img_size[1], 3)

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            data_augmentation,
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="baseline_cnn",
    )

    return model


def build_optimized_model(
    img_size: tuple[int, int] = (160, 160),
    data_augmentation: tf.keras.Sequential | None = None,
) -> tf.keras.Model:
    """Build the optimized CNN model."""
    data_augmentation = data_augmentation or build_data_augmentation()
    input_shape = (img_size[0], img_size[1], 3)

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            data_augmentation,
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.0001)),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="optimized_cnn",
    )

    return model
