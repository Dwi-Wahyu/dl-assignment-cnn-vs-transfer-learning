"""Model evaluation, classification reporting, confusion matrix plotting, and comparison analytics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    model_name: str,
    output_dir: str,
    training_time_sec: float = 0.0,
) -> dict[str, Any]:
    """Perform comprehensive model evaluation on the independent test split.

    Generates:
      1. Overall test loss and accuracy.
      2. Detailed scikit-learn classification report (precision, recall, f1, support).
      3. Seaborn confusion matrix heatmap.
      4. Structured metric dictionary.

    Args:
        model: Evaluated tf.keras.Model.
        test_ds: Evaluation test dataset.
        class_names: List of class category labels in order of indices.
        model_name: Identifier for model naming and file saving.
        output_dir: Target directory for artifacts.
        training_time_sec: Recorded training time in seconds. Defaults to 0.0.

    Returns:
        dict[str, Any]: Dictionary containing evaluation metrics and model capacity metadata.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[*] Evaluating '{model_name}' on test set...")

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"[+] Test Loss: {loss:.4f} | Test Accuracy: {acc * 100:.2f}%")

    # Gather ground-truth labels and model predictions
    y_true_batches = []
    y_pred_batches = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_batches.append(np.argmax(preds, axis=1))
        y_true_batches.append(labels.numpy())

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)

    # Classification report
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    report_file = os.path.join(output_dir, f"{model_name.lower()}_classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report_str)
        f.write("\n")
    print(f"[+] Saved classification report to: {report_file}")
    print("\n" + report_str + "\n")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("True Class", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_file = os.path.join(output_dir, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(cm_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved confusion matrix to: {cm_file}")

    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    total_params = sum(tf.keras.backend.count_params(w) for w in model.weights)

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "loss": float(loss),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "training_time_sec": float(training_time_sec),
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
    }


def plot_model_comparison(results: list[dict[str, Any]], output_dir: str) -> str:
    """Generate comparative 2x2 bar charts across key metrics for all models.

    Metrics visualized: Accuracy, Loss, Macro F1, Training Time.

    Args:
        results: List of evaluation metric dictionaries.
        output_dir: Destination directory.

    Returns:
        str: Absolute path to the saved comparison figure.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_names = [res["model_name"] for res in results]

    metrics = [
        ("accuracy", "Test Accuracy", "Accuracy", "#2b5c8f"),
        ("loss", "Test Loss", "Loss", "#c23b22"),
        ("macro_f1", "Macro F1-Score", "F1-Score", "#2a9d8f"),
        ("training_time_sec", "Training Time (seconds)", "Seconds", "#e76f51"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (key, title, ylabel, color) in enumerate(metrics):
        ax = axes[idx]
        values = [res[key] for res in results]
        bars = ax.bar(model_names, values, color=color, width=0.55, edgecolor="black", alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        # Label values on top of bars
        for bar in bars:
            height = bar.get_height()
            label_text = f"{height:.4f}" if key in ("accuracy", "loss", "macro_f1") else f"{height:.1f}s"
            ax.annotate(
                label_text,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="semibold",
                fontsize=9,
            )

    plt.suptitle("Model Benchmark Comparison: Custom CNN vs MobileNetV2", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_file = os.path.join(output_dir, "model_comparison_charts.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison charts to: {out_file}")
    return out_file


def save_results_table(results: list[dict[str, Any]], csv_path: str) -> pd.DataFrame:
    """Format evaluation results into a clean DataFrame and save to CSV.

    Args:
        results: List of evaluation metric dictionaries.
        csv_path: Target CSV file path.

    Returns:
        pd.DataFrame: Formatted results table.
    """
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)

    # Format numeric columns for presentation
    rename_cols = {
        "model_name": "Model",
        "accuracy": "Accuracy",
        "loss": "Loss",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
        "training_time_sec": "Training Time (s)",
        "trainable_params": "Trainable Params",
        "total_params": "Total Params",
    }
    df.rename(columns=rename_cols, inplace=True)
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved benchmark results table to: {csv_path}")
    return df
