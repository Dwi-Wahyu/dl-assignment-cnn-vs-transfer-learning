"""Training orchestration, callback creation, seed initialization, and history logging.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def set_global_seed(seed: int = 42) -> None:
    """Set global random seed across Python, NumPy, and TensorFlow for full reproducibility.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[+] Global reproducibility seed set to: {seed}")


def make_callbacks(
    checkpoint_path: str,
    patience: int = 5,
) -> list[tf.keras.callbacks.Callback]:
    """Create production training callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.

    Args:
        checkpoint_path: Destination file path for best weights checkpoint (must end in .keras).
        patience: Epoch patience for early stopping. Defaults to 5.

    Returns:
        list[tf.keras.callbacks.Callback]: Configured Keras callbacks.
    """
    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    return [early_stopping, checkpoint, reduce_lr]


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 30,
    callbacks: list[tf.keras.callbacks.Callback] | None = None,
) -> Tuple[tf.keras.callbacks.History, float]:
    """Train model on train_ds, evaluating on val_ds, and record total elapsed time.

    Args:
        model: Compiled tf.keras.Model.
        train_ds: Training tf.data.Dataset.
        val_ds: Validation tf.data.Dataset.
        epochs: Maximum number of training epochs. Defaults to 30.
        callbacks: Optional list of Keras callbacks.

    Returns:
        Tuple[tf.keras.callbacks.History, float]: Training history and elapsed time in seconds.
    """
    print(f"[*] Commencing training for '{model.name}' (max epochs={epochs})...")
    start_time = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"[+] Finished training '{model.name}' in {elapsed_time:.2f} seconds.")
    return history, elapsed_time


def save_history(
    history: tf.keras.callbacks.History | dict[str, Any],
    csv_path: str,
) -> None:
    """Save training history metrics to a CSV file.

    Args:
        history: Keras History object or metrics dictionary.
        csv_path: Destination path for CSV file.
    """
    data = history.history if hasattr(history, "history") else history
    df = pd.DataFrame(data)
    df.index.name = "epoch"
    df.reset_index(inplace=True)
    df["epoch"] = df["epoch"] + 1

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved training history to: {csv_path}")


def plot_training_curves(
    history: tf.keras.callbacks.History | dict[str, Any],
    model_name: str,
    output_dir: str,
) -> str:
    """Generate and save Loss and Accuracy learning curves.

    Args:
        history: Keras History object or history dictionary.
        model_name: Name of model for chart title and file naming.
        output_dir: Output directory to save the image.

    Returns:
        str: Absolute path to the saved figure.
    """
    data = history.history if hasattr(history, "history") else history
    epochs_range = range(1, len(data.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs_range, data.get("loss", []), label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in data:
        axes[0].plot(epochs_range, data.get("val_loss", []), label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title(f"{model_name} - Loss Curve", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Accuracy plot
    acc_key = "accuracy" if "accuracy" in data else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in data else "val_acc"
    axes[1].plot(epochs_range, data.get(acc_key, []), label="Train Accuracy", color="#2ca02c", linewidth=2)
    if val_acc_key in data:
        axes[1].plot(epochs_range, data.get(val_acc_key, []), label="Val Accuracy", color="#d62728", linewidth=2, linestyle="--")
    axes[1].set_title(f"{model_name} - Accuracy Curve", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model_name.lower()}_learning_curves.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved learning curves to: {out_file}")
    return out_file
