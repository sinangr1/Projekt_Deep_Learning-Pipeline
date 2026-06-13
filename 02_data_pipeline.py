from __future__ import annotations

from pathlib import Path

import tensorflow as tf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def load_image_datasets(
    data_dir: str | Path,
    img_size: tuple[int, int] = (160, 160),
    batch_size: int = 32,
    seed: int = 42,
):
    """Load train, validation and test datasets from data_split folders."""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    for directory in [train_dir, val_dir, test_dir]:
        if not directory.exists():
            raise FileNotFoundError(f"Ordner nicht gefunden: {directory}")

    print("Loading datasets...")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
        seed=seed,
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_dataset.class_names
    print("Class names:", class_names)

    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=autotune)
    val_dataset = val_dataset.prefetch(buffer_size=autotune)
    test_dataset = test_dataset.prefetch(buffer_size=autotune)

    return train_dataset, val_dataset, test_dataset, class_names


def count_images_by_class(directory: str | Path, class_names: list[str]) -> dict[str, int]:
    """Count image files for every class folder."""
    directory = Path(directory)
    class_counts: dict[str, int] = {}

    for class_name in class_names:
        class_directory = directory / class_name
        if not class_directory.exists():
            class_counts[class_name] = 0
            continue

        class_counts[class_name] = sum(
            1
            for file in class_directory.rglob("*")
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )

    return class_counts


def calculate_imbalance_ratio(class_counts: dict[str, int]) -> float:
    """Return majority/minority class ratio."""
    count_values = [count for count in class_counts.values() if count > 0]
    if len(count_values) < 2:
        return 1.0
    return float(max(count_values) / min(count_values))


def analyze_class_distribution(data_dir: str | Path, class_names: list[str]) -> dict:
    """Analyze train and validation class distribution."""
    data_dir = Path(data_dir)

    training_class_counts = count_images_by_class(data_dir / "train", class_names)
    validation_class_counts = count_images_by_class(data_dir / "val", class_names)
    test_class_counts = count_images_by_class(data_dir / "test", class_names)

    training_imbalance_ratio = calculate_imbalance_ratio(training_class_counts)
    validation_imbalance_ratio = calculate_imbalance_ratio(validation_class_counts)
    test_imbalance_ratio = calculate_imbalance_ratio(test_class_counts)

    print("\nTraining class distribution:")
    for class_name, count in training_class_counts.items():
        print(f"{class_name}: {count}")

    print("\nValidation class distribution:")
    for class_name, count in validation_class_counts.items():
        print(f"{class_name}: {count}")

    print("\nTest class distribution:")
    for class_name, count in test_class_counts.items():
        print(f"{class_name}: {count}")

    print(f"\nTraining imbalance ratio:    {training_imbalance_ratio:.2f}")
    print(f"Validation imbalance ratio:  {validation_imbalance_ratio:.2f}")
    print(f"Test imbalance ratio:        {test_imbalance_ratio:.2f}")

    return {
        "training_class_counts": training_class_counts,
        "validation_class_counts": validation_class_counts,
        "test_class_counts": test_class_counts,
        "training_imbalance_ratio": training_imbalance_ratio,
        "validation_imbalance_ratio": validation_imbalance_ratio,
        "test_imbalance_ratio": test_imbalance_ratio,
    }
