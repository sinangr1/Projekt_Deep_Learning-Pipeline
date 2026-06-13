from __future__ import annotations


def decide_optimization_need(
    history_analysis: dict,
    training_imbalance_ratio: float,
    target_validation_accuracy: float = 0.85,
) -> dict:
    """Decide whether optimization is needed based on model behavior."""
    optimization_needed = False
    optimization_reasons: list[str] = []
    recommended_actions: list[str] = []

    accuracy_gap = history_analysis["accuracy_gap"]
    final_training_accuracy = history_analysis["final_training_accuracy"]
    final_validation_accuracy = history_analysis["final_validation_accuracy"]
    final_training_loss = history_analysis["final_training_loss"]
    final_validation_loss = history_analysis["final_validation_loss"]
    best_validation_loss = history_analysis["best_validation_loss"]

    if accuracy_gap > 0.08:
        optimization_needed = True
        optimization_reasons.append(
            "There is a large gap between training accuracy and validation accuracy. This may indicate overfitting."
        )
        recommended_actions.append(
            "Increase dropout, add L2 regularization, use Batch Normalization, or reduce model complexity."
        )

    if final_validation_loss > best_validation_loss * 1.10:
        optimization_needed = True
        optimization_reasons.append(
            "The final validation loss is significantly higher than the best validation loss. The model may have become worse in later epochs."
        )
        recommended_actions.append(
            "Use EarlyStopping, reduce patience, or lower the learning rate."
        )

    if final_training_accuracy < 0.75 and final_validation_accuracy < 0.75:
        optimization_needed = True
        optimization_reasons.append(
            "Both training accuracy and validation accuracy are low. This may indicate underfitting."
        )
        recommended_actions.append(
            "Increase model capacity, train for more epochs, or adjust the learning rate."
        )

    if final_validation_accuracy < target_validation_accuracy:
        optimization_needed = True
        optimization_reasons.append(
            f"Validation accuracy is below the target value of {target_validation_accuracy:.2f}."
        )
        recommended_actions.append(
            "Tune the learning rate, model architecture, data augmentation, and regularization settings."
        )

    if training_imbalance_ratio >= 1.5:
        optimization_needed = True
        optimization_reasons.append(
            "The training dataset has class imbalance. The model may become biased toward the majority class."
        )
        recommended_actions.append(
            "Use class weights or add more images for the minority class."
        )

    return {
        "optimization_needed": optimization_needed,
        "optimization_reasons": optimization_reasons,
        "recommended_actions": recommended_actions,
        "target_validation_accuracy": float(target_validation_accuracy),
    }


def print_optimization_decision(decision: dict) -> None:
    """Print readable optimization decision."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION DECISION")
    print("=" * 60)

    if decision["optimization_needed"]:
        print("Result: OPTIMIZATION IS NEEDED")

        print("\nWhy optimization is needed:")
        for index, reason in enumerate(decision["optimization_reasons"], start=1):
            print(f"{index}. {reason}")

        print("\nRecommended optimization direction:")
        for index, action in enumerate(decision["recommended_actions"], start=1):
            print(f"{index}. {action}")
    else:
        print("Result: OPTIMIZATION IS NOT NECESSARY RIGHT NOW")
        print(
            "\nThe validation performance is close to the training performance. "
            "There is no strong sign of overfitting, underfitting, or class imbalance."
        )

    print("=" * 60)
