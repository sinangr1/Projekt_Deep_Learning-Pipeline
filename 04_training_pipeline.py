from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """Compile a binary CNN model."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_baseline_callbacks(model_path: str | Path) -> list[tf.keras.callbacks.Callback]:
    """Callbacks for baseline training."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def get_optimized_callbacks(model_path: str | Path) -> list[tf.keras.callbacks.Callback]:
    """Callbacks for optimized training."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1,
        ),
    ]


def train_model(
    model: tf.keras.Model,
    train_dataset,
    val_dataset,
    epochs: int,
    callbacks: list[tf.keras.callbacks.Callback],
    model_name: str,
):
    """Train a model and return Keras History."""
    print(f"\nTraining {model_name}...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history


def evaluate_model(model: tf.keras.Model, dataset, dataset_name: str) -> dict[str, float]:
    """Evaluate model and return metric dictionary."""
    print(f"\nEvaluating model on {dataset_name}...")
    results = model.evaluate(dataset, return_dict=True)

    print(f"\n{dataset_name} results:")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value:.4f}")

    return {name: float(value) for name, value in results.items()}


def analyze_training_history(history) -> dict[str, float | int]:
    """Analyze final train/validation accuracy and loss gap."""
    final_training_accuracy = float(history.history["accuracy"][-1])
    final_validation_accuracy = float(history.history["val_accuracy"][-1])
    final_training_loss = float(history.history["loss"][-1])
    final_validation_loss = float(history.history["val_loss"][-1])

    best_validation_loss = float(min(history.history["val_loss"]))
    best_validation_loss_epoch = int(history.history["val_loss"].index(best_validation_loss) + 1)

    accuracy_gap = float(final_training_accuracy - final_validation_accuracy)
    loss_gap = float(final_validation_loss - final_training_loss)

    print("\nTraining and validation comparison:")
    print(f"Final training accuracy:     {final_training_accuracy:.4f}")
    print(f"Final validation accuracy:   {final_validation_accuracy:.4f}")
    print(f"Accuracy gap:                {accuracy_gap:.4f}")
    print(f"\nFinal training loss:         {final_training_loss:.4f}")
    print(f"Final validation loss:       {final_validation_loss:.4f}")
    print(f"Loss gap:                    {loss_gap:.4f}")
    print(f"\nBest validation loss:        {best_validation_loss:.4f}")
    print(f"Best validation loss epoch:  {best_validation_loss_epoch}")

    return {
        "final_training_accuracy": final_training_accuracy,
        "final_validation_accuracy": final_validation_accuracy,
        "final_training_loss": final_training_loss,
        "final_validation_loss": final_validation_loss,
        "best_validation_loss": best_validation_loss,
        "best_validation_loss_epoch": best_validation_loss_epoch,
        "accuracy_gap": accuracy_gap,
        "loss_gap": loss_gap,
    }


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1-score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))
