"""Grad-CAM (Gradient-weighted Class Activation Mapping) visualization module.

Supports both custom sequential/functional CNN architectures and nested
transfer learning models (e.g. MobileNetV2 backbone).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model) -> Tuple[str, bool]:
    """Identify the name of the final convolutional layer in the model.

    Args:
        model: Target Keras model.

    Returns:
        Tuple[str, bool]: (layer_name, is_nested)
    """
    # Check top-level layers in reverse
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name, False
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    return sub_layer.name, True

    # Fallback to common layer names
    for name in ["conv4_1", "conv3_2", "out_relu", "Conv_1"]:
        for layer in model.layers:
            if layer.name == name:
                return name, False
            if hasattr(layer, "layers") and any(l.name == name for l in layer.layers):
                return name, True

    raise ValueError("Could not automatically locate a convolutional layer in the model.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str | None = None,
    pred_index: int | None = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for a given input image array and model.

    Handles both flat convolutional networks and nested transfer learning models.

    Args:
        img_array: Batch tensor of shape (1, H, W, C) properly preprocessed.
        model: tf.keras.Model.
        last_conv_layer_name: Name of target feature map layer. If None, auto-detected.
        pred_index: Target class index to inspect. If None, uses top predicted class.

    Returns:
        np.ndarray: 2D array (H, W) with normalized activation intensity in [0, 1].
    """
    if last_conv_layer_name is None:
        last_conv_layer_name, is_nested = find_last_conv_layer(model)
    else:
        is_nested = any(
            hasattr(l, "layers") and any(sl.name == last_conv_layer_name for sl in l.layers)
            for l in model.layers
        )

    if not is_nested:
        # Standard flat model
        grad_model = tf.keras.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
    else:
        # Nested model (e.g. MobileNetV2 backbone inside outer model)
        nested_layer = next(
            l for l in model.layers
            if hasattr(l, "layers") and any(sl.name == last_conv_layer_name for sl in l.layers)
        )
        target_layer = nested_layer.get_layer(last_conv_layer_name)

        # Feature model extracts target feature map and backbone output
        feature_model = tf.keras.Model(
            inputs=[nested_layer.inputs],
            outputs=[target_layer.output, nested_layer.output],
        )

        # Reconstruct classification head
        head_input = tf.keras.Input(shape=nested_layer.output.shape[1:])
        x = head_input
        # Find index of nested layer in model.layers
        nested_idx = model.layers.index(nested_layer)
        for layer in model.layers[nested_idx + 1:]:
            x = layer(x)
        classifier_model = tf.keras.Model(inputs=head_input, outputs=x)

        with tf.GradientTape() as tape:
            conv_outputs, backbone_outputs = feature_model(img_array)
            tape.watch(conv_outputs)
            predictions = classifier_model(backbone_outputs)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    # Global Average Pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the convolutional feature channels by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain only features that contribute positively to the class
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def display_gradcam(
    img_path_or_array: str | np.ndarray,
    heatmap: np.ndarray,
    output_path: str | None = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Superimpose Grad-CAM heatmap onto the original RGB image.

    Args:
        img_path_or_array: File path to image or numpy image array (H, W, 3).
        heatmap: 2D numpy array heatmap from make_gradcam_heatmap.
        output_path: Optional path to save the superimposed image. Defaults to None.
        alpha: Blending weight for heatmap overlay (0.0 to 1.0). Defaults to 0.4.

    Returns:
        np.ndarray: Blended RGB image array uint8 in range [0, 255].
    """
    if isinstance(img_path_or_array, str):
        img = Image.open(img_path_or_array).convert("RGB")
        img = np.array(img, dtype=np.float32)
    else:
        img = np.array(img_path_or_array, dtype=np.float32)
        if img.max() <= 1.0:
            img = img * 255.0

    # Resize heatmap to match original image dimensions
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_resized = heatmap_img.resize((img.shape[1], img.shape[0]), resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_resized) / 255.0

    # Colorize using Jet colormap
    try:
        cmap = cm.colormaps["jet"]
    except AttributeError:
        cmap = cm.get_cmap("jet")

    jet_colors = cmap(heatmap_resized)[:, :, :3]  # Discard alpha
    jet_heatmap = np.uint8(255 * jet_colors)

    # Superimpose heatmap on original image
    superimposed = np.uint8(img * (1.0 - alpha) + jet_heatmap * alpha)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(superimposed).save(output_path)
        print(f"[+] Saved Grad-CAM visualization to: {output_path}")

    return superimposed
