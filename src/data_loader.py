"""Dataset loader, downloader, resolver, and preprocessing utilities for Intel Image Classification.

Handles downloading via kagglehub, kaggle cli, or google drive zip,
resolves double-nested directories (seg_train/seg_train, seg_test/seg_test),
and builds tf.data pipelines for Custom CNN and MobileNetV2.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Tuple

import tensorflow as tf

EXPECTED_CLASSES = {"buildings", "forest", "glacier", "mountain", "sea", "street"}


def download_dataset_kagglehub(handle: str = "puneet6060/intel-image-classification") -> str:
    """Download Intel Image Classification dataset using the kagglehub library.

    Args:
        handle: Kaggle dataset handle (owner/dataset-name).

    Returns:
        str: Absolute path to the downloaded dataset root directory.

    Raises:
        ImportError: If kagglehub is not installed.
        RuntimeError: If download fails.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed. Install it via 'pip install kagglehub' to use this downloader."
        ) from exc

    try:
        print(f"[*] Downloading dataset '{handle}' via kagglehub...")
        download_path = kagglehub.dataset_download(handle)
        print(f"[+] Download complete: {download_path}")
        return str(download_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to download dataset '{handle}' via kagglehub: {exc}") from exc


def download_dataset_kaggle_cli(
    handle: str = "puneet6060/intel-image-classification",
    output_dir: str = "/content/dataset",
) -> str:
    """Download and unzip dataset using the Kaggle CLI.

    Requires Kaggle credentials (either ~/.kaggle/kaggle.json or
    KAGGLE_USERNAME and KAGGLE_KEY environment variables).

    Args:
        handle: Kaggle dataset handle (owner/dataset-name).
        output_dir: Target directory to extract the dataset.

    Returns:
        str: Absolute path to the output directory.

    Raises:
        RuntimeError: If Kaggle CLI command execution fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        handle,
        "--unzip",
        "-p",
        output_dir,
    ]
    print(f"[*] Running Kaggle CLI command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle CLI failed with return code {result.returncode}.\n"
            f"Stderr: {result.stderr.strip()}\n"
            f"Stdout: {result.stdout.strip()}\n"
            "Ensure KAGGLE_USERNAME and KAGGLE_KEY are set or ~/.kaggle/kaggle.json exists."
        )
    print(f"[+] Kaggle CLI download completed in: {output_dir}")
    return output_dir


def extract_dataset_from_drive(zip_path: str, output_dir: str = "/content/dataset") -> str:
    """Extract a zipped dataset from Google Drive or local storage into output_dir.

    Args:
        zip_path: Path to the .zip archive file.
        output_dir: Destination directory.

    Returns:
        str: Absolute path to the extracted destination directory.

    Raises:
        FileNotFoundError: If zip_path does not exist.
        zipfile.BadZipFile: If zip_path is corrupted.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip archive not found at: {zip_path}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Extracting '{zip_path}' to '{output_dir}'...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)
    print(f"[+] Extraction complete in: {output_dir}")
    return output_dir


def _is_valid_class_dir(directory: Path, expected_classes: set[str] = EXPECTED_CLASSES) -> bool:
    """Check if directory contains subdirectories matching the target dataset classes."""
    if not directory.is_dir():
        return False
    subdirs = {d.name.lower() for d in directory.iterdir() if d.is_dir()}
    # Matches if at least majority of expected classes exist
    return len(expected_classes.intersection(subdirs)) >= 4


def resolve_dataset_dirs(root: str) -> Tuple[str, str]:
    """Recursively search root directory to resolve correct train and test split directories.

    Handles double-nested structures such as:
      seg_train/seg_train/<class>/
      seg_test/seg_test/<class>/
    and ignores unused folders like seg_pred/seg_pred/.
    Does not depend on hardcoded version numbers or nested depths.

    Args:
        root: Root search directory.

    Returns:
        Tuple[str, str]: (train_dir, test_dir) absolute paths.

    Raises:
        FileNotFoundError: If root directory or valid train/test splits cannot be found.
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root_path}")

    train_candidates: list[Path] = []
    test_candidates: list[Path] = []

    # Traverse all directories recursively
    for path in root_path.rglob("*"):
        if not path.is_dir():
            continue

        name_lower = path.name.lower()
        # Explicit check for seg_train and seg_test folders containing class subfolders
        if "seg_train" in name_lower and _is_valid_class_dir(path):
            train_candidates.append(path)
        elif "seg_test" in name_lower and _is_valid_class_dir(path):
            test_candidates.append(path)

    # Fallback in case directory names were slightly altered during unzip
    if not train_candidates:
        for path in root_path.rglob("*"):
            if path.is_dir() and "train" in str(path).lower() and _is_valid_class_dir(path):
                train_candidates.append(path)

    if not test_candidates:
        for path in root_path.rglob("*"):
            if (
                path.is_dir()
                and ("test" in str(path).lower() or "val" in str(path).lower())
                and _is_valid_class_dir(path)
            ):
                test_candidates.append(path)

    if not train_candidates:
        raise FileNotFoundError(
            f"Could not find valid 'seg_train' directory containing classes under: {root_path}"
        )
    if not test_candidates:
        raise FileNotFoundError(
            f"Could not find valid 'seg_test' directory containing classes under: {root_path}"
        )

    # In double-nested structures, pick the deepest path (e.g. seg_train/seg_train over seg_train)
    train_dir = max(train_candidates, key=lambda p: len(p.parts))
    test_dir = max(test_candidates, key=lambda p: len(p.parts))

    print(f"[+] Resolved train_dir: {train_dir}")
    print(f"[+] Resolved test_dir : {test_dir}")
    return str(train_dir), str(test_dir)


def build_datasets(
    train_dir: str,
    test_dir: str,
    img_size: Tuple[int, int] = (150, 150),
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Build tf.data.Dataset pipelines for training, validation, and testing.

    Validation is split strictly from train_dir (test_dir remains untouched).

    Args:
        train_dir: Path to train directory.
        test_dir: Path to test directory.
        img_size: Target image dimensions (height, width). Defaults to (150, 150).
        batch_size: Mini-batch size. Defaults to 32.
        val_split: Proportion of training data for validation. Defaults to 0.2.
        seed: Random seed for deterministic train/val split. Defaults to 42.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
            (train_ds, val_ds, test_ds, class_names)
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names

    # Enable caching and prefetching for optimal throughput
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def get_augmentation_layer() -> tf.keras.Sequential:
    """Return data augmentation layer with RandomFlip, RandomRotation, and RandomZoom."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", name="aug_random_flip"),
            tf.keras.layers.RandomRotation(0.1, name="aug_random_rotation"),
            tf.keras.layers.RandomZoom(0.1, name="aug_random_zoom"),
        ],
        name="data_augmentation",
    )


def preprocess_custom_cnn(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Normalize image pixel values from [0, 255] to [0.0, 1.0] for Custom CNN."""
    rescaling_layer = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling_custom_cnn")
    return ds.map(
        lambda images, labels: (rescaling_layer(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def preprocess_mobilenet(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Preprocess image inputs for MobileNetV2 using tf.keras.applications.mobilenet_v2.preprocess_input.

    Scales pixel values from [0, 255] to [-1.0, 1.0].
    """
    preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
    return ds.map(
        lambda images, labels: (preprocess_fn(tf.cast(images, tf.float32)), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def print_dataset_summary(class_names: list[str], splits: dict[str, Any]) -> None:
    """Print structured summary of classes and split sizes."""
    print("=" * 60)
    print("DATASET SUMMARY: Intel Image Classification (6 Classes)")
    print("=" * 60)
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print("-" * 60)
    for split_name, count in splits.items():
        print(f"  - {split_name:<16}: {count}")
    print("=" * 60)
