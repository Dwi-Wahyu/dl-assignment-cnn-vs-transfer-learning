"""Model architectures for Intel Image Classification:
1. Custom CNN from scratch
2. Transfer Learning with MobileNetV2 (ImageNet)
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (150, 150, 3),
    num_classes: int = 6,
) -> tf.keras.Model:
    """Build Custom Convolutional Neural Network from scratch.

    Architecture consists of 4 convolutional blocks with BatchNormalization,
    MaxPooling, and Dropout, terminating in GlobalAveragePooling2D and Dense layers.

    Args:
        input_shape: Input tensor shape (H, W, C). Defaults to (150, 150, 3).
        num_classes: Number of output classification categories. Defaults to 6.

    Returns:
        tf.keras.Model: Compiled or uncompiled Custom CNN model named 'CustomCNN'.
    """
    inputs = tf.keras.Input(shape=input_shape, name="custom_cnn_input")

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1_1")(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = tf.keras.layers.Dropout(0.2, name="drop1")(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2_1")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = tf.keras.layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3_1")(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = tf.keras.layers.Dropout(0.3, name="drop3")(x)

    # Block 4 (Feature refinement)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv4_1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4_1")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = tf.keras.layers.Dropout(0.3, name="drop4")(x)

    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_dense")(x)
    x = tf.keras.layers.Dropout(0.4, name="drop_dense")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CustomCNN")
    return model


def build_transfer_learning_model(
    input_shape: Tuple[int, int, int] = (150, 150, 3),
    num_classes: int = 6,
    fine_tune_at: int | None = None,
) -> tf.keras.Model:
    """Build Transfer Learning model using pre-trained MobileNetV2 (ImageNet).

    Args:
        input_shape: Input image dimensions. Defaults to (150, 150, 3).
        num_classes: Number of target classes. Defaults to 6.
        fine_tune_at: Layer index from which to unfreeze base layers for fine-tuning.
            If None, all base layers remain frozen (feature extraction stage).

    Returns:
        tf.keras.Model: Configured MobileNetV2 transfer learning model named 'MobileNetV2_TransferLearning'.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    if fine_tune_at is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        # Freeze all layers prior to fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="mobilenet_input")
    # Note: training=False ensures BatchNormalization layers remain in inference mode
    # even when unfreezing weights during fine-tuning (avoids updating running statistics).
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_TransferLearning")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Compile model with Adam optimizer, SparseCategoricalCrossentropy loss, and Accuracy metric.

    Args:
        model: Model to be compiled.
        learning_rate: Initial learning rate for Adam optimizer. Defaults to 1e-3.

    Returns:
        tf.keras.Model: Compiled model instance.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def count_trainable_params(model: tf.keras.Model) -> Tuple[int, int]:
    """Calculate the number of trainable and total parameters of a Keras model.

    Args:
        model: Model instance.

    Returns:
        Tuple[int, int]: (trainable_parameters, total_parameters).
    """
    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    total_count = trainable_count + non_trainable_count
    return int(trainable_count), int(total_count)
