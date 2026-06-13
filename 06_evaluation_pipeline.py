from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score


def plot_training_curves(history, output_dir: str | Path, prefix: str) -> dict[str, Path]:
    """Save accuracy and loss curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_path = output_dir / f"{prefix}_accuracy.png"
    loss_path = output_dir / f"{prefix}_loss.png"

    plt.figure()
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Training and Validation Accuracy - {prefix}")
    plt.savefig(accuracy_path, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training and Validation Loss - {prefix}")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {accuracy_path}")
    print(f"Saved plot: {loss_path}")

    return {"accuracy_plot": accuracy_path, "loss_plot": loss_path}


def get_true_labels(dataset) -> np.ndarray:
    """Extract true labels from a Keras dataset."""
    y_true = []
    for _, labels in dataset:
        y_true.extend(labels.numpy().astype(int).reshape(-1))
    return np.array(y_true)


def create_confusion_matrix_for_model(
    trained_model,
    test_dataset,
    class_names: list[str],
    model_name: str,
    output_path: str | Path,
) -> dict:
    """Create, print and save confusion matrix and classification report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"CONFUSION MATRIX - {model_name}")
    print("=" * 70)

    y_true = get_true_labels(test_dataset)
    y_pred_probabilities = trained_model.predict(test_dataset).reshape(-1)
    y_pred = (y_pred_probabilities >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nInterpretation:")
    print("[[True class 0, False class 1],")
    print(" [False class 0, True class 1]]")
    print(f"Class order: {class_names}")

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report_text)

    if "zebra" in class_names:
        zebra_class_index = class_names.index("zebra")
    else:
        zebra_class_index = 1

    zebra_f1 = f1_score(y_true, y_pred, pos_label=zebra_class_index, zero_division=0)
    print(f"\nF1-score for zebra class: {zebra_f1:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix: {output_path}")

    return {
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
        "zebra_f1_score": float(zebra_f1),
        "output_path": str(output_path),
    }


def print_final_summary(
    baseline_validation_results: dict,
    optimized_validation_results: dict,
    optimized_test_results: dict,
    baseline_validation_f1: float,
    optimized_validation_f1: float,
    optimized_test_f1: float,
    history_analysis: dict,
    class_distribution: dict,
) -> None:
    """Print final comparison summary."""
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)

    print("\nBASELINE MODEL - VALIDATION RESULTS")
    print("-" * 70)
    for metric_name, metric_value in baseline_validation_results.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print(f"f1_score: {baseline_validation_f1:.4f}")

    print("\nTraining and validation comparison:")
    print(f"Final training accuracy:     {history_analysis['final_training_accuracy']:.4f}")
    print(f"Final validation accuracy:   {history_analysis['final_validation_accuracy']:.4f}")
    print(f"Accuracy gap:                {history_analysis['accuracy_gap']:.4f}")
    print(f"\nFinal training loss:         {history_analysis['final_training_loss']:.4f}")
    print(f"Final validation loss:       {history_analysis['final_validation_loss']:.4f}")
    print(f"Loss gap:                    {history_analysis['loss_gap']:.4f}")
    print(f"\nBest validation loss:        {history_analysis['best_validation_loss']:.4f}")
    print(f"Best validation loss epoch:  {history_analysis['best_validation_loss_epoch']}")

    print("\nTraining class distribution:")
    for class_name, count in class_distribution["training_class_counts"].items():
        print(f"{class_name}: {count}")

    print("\nValidation class distribution:")
    for class_name, count in class_distribution["validation_class_counts"].items():
        print(f"{class_name}: {count}")

    print(f"\nTraining imbalance ratio:    {class_distribution['training_imbalance_ratio']:.2f}")
    print(f"Validation imbalance ratio:  {class_distribution['validation_imbalance_ratio']:.2f}")

    print("\n" + "-" * 70)
    print("OPTIMIZED MODEL - VALIDATION RESULTS")
    print("-" * 70)
    for metric_name, metric_value in optimized_validation_results.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print(f"f1_score: {optimized_validation_f1:.4f}")

    print("\n" + "-" * 70)
    print("OPTIMIZED MODEL - TEST RESULTS")
    print("-" * 70)
    for metric_name, metric_value in optimized_test_results.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print(f"f1_score: {optimized_test_f1:.4f}")

    print("\n" + "-" * 70)
    print("BASELINE VS OPTIMIZED VALIDATION COMPARISON")
    print("-" * 70)
    print(f"Baseline validation accuracy:  {baseline_validation_results['accuracy']:.4f}")
    print(f"Optimized validation accuracy: {optimized_validation_results['accuracy']:.4f}")
    print(f"Accuracy improvement:          {optimized_validation_results['accuracy'] - baseline_validation_results['accuracy']:+.4f}")

    print(f"\nBaseline validation loss:      {baseline_validation_results['loss']:.4f}")
    print(f"Optimized validation loss:     {optimized_validation_results['loss']:.4f}")
    print(f"Loss difference:               {optimized_validation_results['loss'] - baseline_validation_results['loss']:+.4f}")

    print(f"\nBaseline validation precision: {baseline_validation_results['precision']:.4f}")
    print(f"Optimized validation precision:{optimized_validation_results['precision']:.4f}")
    print(f"Precision difference:          {optimized_validation_results['precision'] - baseline_validation_results['precision']:+.4f}")

    print(f"\nBaseline validation recall:    {baseline_validation_results['recall']:.4f}")
    print(f"Optimized validation recall:   {optimized_validation_results['recall']:.4f}")
    print(f"Recall improvement:            {optimized_validation_results['recall'] - baseline_validation_results['recall']:+.4f}")

    print(f"\nBaseline validation F1-score:  {baseline_validation_f1:.4f}")
    print(f"Optimized validation F1-score: {optimized_validation_f1:.4f}")
    print(f"F1-score improvement:          {optimized_validation_f1 - baseline_validation_f1:+.4f}")
    print("=" * 70)


def save_summary_json(summary: dict, output_path: str | Path) -> Path:
    """Save JSON summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, default=str)

    print(f"Saved summary: {output_path}")
    return output_path
