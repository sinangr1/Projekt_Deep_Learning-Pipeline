from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data_split"
DEFAULT_MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _load_step_module(filename: str, module_name: str):
    path = PROJECT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Konnte Modul nicht laden: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data_step = _load_step_module("02_data_pipeline.py", "zebra_data_pipeline")
model_step = _load_step_module("03_model_pipeline.py", "zebra_model_pipeline")
training_step = _load_step_module("04_training_pipeline.py", "zebra_training_pipeline")
optimization_step = _load_step_module("05_optimization_decision.py", "zebra_optimization_decision")
evaluation_step = _load_step_module("06_evaluation_pipeline.py", "zebra_evaluation_pipeline")


def run_zebra_cnn_pipeline(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    model_output_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    img_size: tuple[int, int] = (160, 160),
    batch_size: int = 32,
    baseline_epochs: int = 20,
    optimized_epochs: int = 40,
    baseline_learning_rate: float = 0.001,
    optimized_learning_rate: float = 0.0005,
    target_validation_accuracy: float = 0.85,
) -> dict:
    """Run the full zebra/no_zebra CNN pipeline in a strict order."""
    data_dir = Path(data_dir)
    model_output_dir = Path(model_output_dir)
    output_dir = Path(output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_model_path = model_output_dir / "baseline_cnn.keras"
    optimized_model_path = model_output_dir / "optimized_cnn.keras"

    print("\n1. Load datasets")
    train_dataset, val_dataset, test_dataset, class_names = data_step.load_image_datasets(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
    )

    print("\n2. Analyze class distribution")
    class_distribution = data_step.analyze_class_distribution(data_dir, class_names)

    print("\n3. Build data augmentation")
    data_augmentation = model_step.build_data_augmentation()

    print("\n4. Build baseline CNN model")
    baseline_model = model_step.build_baseline_model(
        img_size=img_size,
        data_augmentation=data_augmentation,
    )
    baseline_model.summary()

    print("\n5. Compile baseline model")
    training_step.compile_model(baseline_model, learning_rate=baseline_learning_rate)

    print("\n6. Train baseline model")
    baseline_history = training_step.train_model(
        model=baseline_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=baseline_epochs,
        callbacks=training_step.get_baseline_callbacks(baseline_model_path),
        model_name="baseline CNN model",
    )

    print("\n7. Evaluate baseline model on validation data")
    baseline_validation_results = training_step.evaluate_model(
        baseline_model,
        val_dataset,
        dataset_name="validation dataset",
    )

    print("\n8. Analyze baseline training history")
    baseline_history_analysis = training_step.analyze_training_history(baseline_history)

    print("\n9. Decide if optimization is needed")
    optimization_decision = optimization_step.decide_optimization_need(
        history_analysis=baseline_history_analysis,
        training_imbalance_ratio=class_distribution["training_imbalance_ratio"],
        target_validation_accuracy=target_validation_accuracy,
    )
    optimization_step.print_optimization_decision(optimization_decision)

    print("\n10. Evaluate baseline model on test data")
    baseline_test_results = training_step.evaluate_model(
        baseline_model,
        test_dataset,
        dataset_name="test dataset",
    )
    baseline_model.save(baseline_model_path)
    print(f"Baseline model saved to: {baseline_model_path}")

    print("\n11. Build optimized CNN model")
    optimized_model = model_step.build_optimized_model(
        img_size=img_size,
        data_augmentation=data_augmentation,
    )
    optimized_model.summary()

    print("\n12. Compile optimized model")
    training_step.compile_model(optimized_model, learning_rate=optimized_learning_rate)

    print("\n13. Train optimized model")
    optimized_history = training_step.train_model(
        model=optimized_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=optimized_epochs,
        callbacks=training_step.get_optimized_callbacks(optimized_model_path),
        model_name="optimized CNN model",
    )

    print("\n14. Evaluate optimized model on validation data")
    optimized_validation_results = training_step.evaluate_model(
        optimized_model,
        val_dataset,
        dataset_name="optimized validation dataset",
    )

    print("\n15. Compare baseline and optimized model")
    print("\n" + "=" * 60)
    print("BASELINE VS OPTIMIZED MODEL")
    print("=" * 60)
    print(f"Baseline validation accuracy:  {baseline_validation_results['accuracy']:.4f}")
    print(f"Optimized validation accuracy: {optimized_validation_results['accuracy']:.4f}")
    print(f"\nBaseline validation loss:      {baseline_validation_results['loss']:.4f}")
    print(f"Optimized validation loss:     {optimized_validation_results['loss']:.4f}")
    print(f"\nAccuracy difference:           {optimized_validation_results['accuracy'] - baseline_validation_results['accuracy']:+.4f}")
    print(f"Loss difference:               {optimized_validation_results['loss'] - baseline_validation_results['loss']:+.4f}")
    print("=" * 60)

    print("\n16. Evaluate optimized model on test data")
    optimized_test_results = training_step.evaluate_model(
        optimized_model,
        test_dataset,
        dataset_name="optimized test dataset",
    )
    optimized_model.save(optimized_model_path)
    print(f"Optimized model saved to: {optimized_model_path}")

    print("\n17. Calculate F1 scores")
    baseline_validation_f1 = training_step.calculate_f1_score(
        baseline_validation_results["precision"],
        baseline_validation_results["recall"],
    )
    optimized_validation_f1 = training_step.calculate_f1_score(
        optimized_validation_results["precision"],
        optimized_validation_results["recall"],
    )
    optimized_test_f1 = training_step.calculate_f1_score(
        optimized_test_results["precision"],
        optimized_test_results["recall"],
    )

    print("\n18. Print final summary")
    evaluation_step.print_final_summary(
        baseline_validation_results=baseline_validation_results,
        optimized_validation_results=optimized_validation_results,
        optimized_test_results=optimized_test_results,
        baseline_validation_f1=baseline_validation_f1,
        optimized_validation_f1=optimized_validation_f1,
        optimized_test_f1=optimized_test_f1,
        history_analysis=baseline_history_analysis,
        class_distribution=class_distribution,
    )

    print("\n19. Plot training curves")
    baseline_plots = evaluation_step.plot_training_curves(
        baseline_history,
        output_dir=output_dir,
        prefix="baseline",
    )
    optimized_plots = evaluation_step.plot_training_curves(
        optimized_history,
        output_dir=output_dir,
        prefix="optimized",
    )

    print("\n20. Create confusion matrices")
    baseline_confusion = evaluation_step.create_confusion_matrix_for_model(
        trained_model=baseline_model,
        test_dataset=test_dataset,
        class_names=class_names,
        model_name="Baseline Model on Test Data",
        output_path=output_dir / "confusion_matrix_baseline_test.png",
    )

    optimized_confusion = evaluation_step.create_confusion_matrix_for_model(
        trained_model=optimized_model,
        test_dataset=test_dataset,
        class_names=class_names,
        model_name="Optimized Model on Test Data",
        output_path=output_dir / "confusion_matrix_optimized_test.png",
    )

    summary = {
        "class_names": class_names,
        "img_size": img_size,
        "batch_size": batch_size,
        "baseline_epochs": baseline_epochs,
        "optimized_epochs": optimized_epochs,
        "class_distribution": class_distribution,
        "baseline_validation_results": baseline_validation_results,
        "baseline_test_results": baseline_test_results,
        "baseline_history_analysis": baseline_history_analysis,
        "optimization_decision": optimization_decision,
        "optimized_validation_results": optimized_validation_results,
        "optimized_test_results": optimized_test_results,
        "f1_scores": {
            "baseline_validation_f1": baseline_validation_f1,
            "optimized_validation_f1": optimized_validation_f1,
            "optimized_test_f1": optimized_test_f1,
        },
        "exports": {
            "baseline_model": str(baseline_model_path),
            "optimized_model": str(optimized_model_path),
            "baseline_plots": {k: str(v) for k, v in baseline_plots.items()},
            "optimized_plots": {k: str(v) for k, v in optimized_plots.items()},
            "baseline_confusion_matrix": baseline_confusion["output_path"],
            "optimized_confusion_matrix": optimized_confusion["output_path"],
        },
        "confusion_matrices": {
            "baseline": baseline_confusion,
            "optimized": optimized_confusion,
        },
    }

    evaluation_step.save_summary_json(summary, output_dir / "zebra_cnn_summary.json")

    print("\nPipeline finished successfully.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ordered Zebra CNN pipeline.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Ordner mit train/val/test Unterordnern.")
    parser.add_argument("--model-output-dir", default=str(DEFAULT_MODEL_OUTPUT_DIR), help="Ordner für gespeicherte Modelle.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Ordner für Plots und Reports.")
    parser.add_argument("--img-size", type=int, default=160, help="Bildgrösse, z.B. 160 bedeutet 160x160.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--baseline-epochs", type=int, default=20, help="Epochs für Baseline-Modell.")
    parser.add_argument("--optimized-epochs", type=int, default=40, help="Epochs für optimiertes Modell.")
    parser.add_argument("--target-validation-accuracy", type=float, default=0.85, help="Zielwert für Validation Accuracy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_zebra_cnn_pipeline(
        data_dir=args.data_dir,
        model_output_dir=args.model_output_dir,
        output_dir=args.output_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        baseline_epochs=args.baseline_epochs,
        optimized_epochs=args.optimized_epochs,
        target_validation_accuracy=args.target_validation_accuracy,
    )


if __name__ == "__main__":
    main()
