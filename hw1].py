import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ---------------------------------------------------
# 1. TENSOR MANIPULATIONS AND RESHAPING
# ---------------------------------------------------

print("\n--- Task 1: Tensor Manipulations ---")

tensor = tf.random.uniform(shape=(4, 6))
print("Original tensor shape:", tensor.shape)
print("Original tensor rank:", tf.rank(tensor).numpy())

reshaped_tensor = tf.reshape(tensor, (2, 3, 4))
print("Reshaped tensor shape:", reshaped_tensor.shape)
print("Reshaped tensor rank:", tf.rank(reshaped_tensor).numpy())

transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
print("Transposed tensor shape:", transposed_tensor.shape)
print("Transposed tensor rank:", tf.rank(transposed_tensor).numpy())

small_tensor = tf.random.uniform(shape=(1, 4))
broadcasted_tensor = tf.broadcast_to(small_tensor, (3, 2, 4))
result = transposed_tensor + broadcasted_tensor

print("Small tensor shape:", small_tensor.shape)
print("Broadcasted tensor shape:", broadcasted_tensor.shape)
print("Result after addition shape:", result.shape)


# ---------------------------------------------------
# 2. LOSS FUNCTIONS AND HYPERPARAMETER TUNING
# ---------------------------------------------------

print("\n--- Task 2: Loss Functions ---")

# Each row represents one item with three possible classes.
y_true = tf.constant([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
])

y_pred_good = tf.constant([
    [0.90, 0.05, 0.05],
    [0.10, 0.80, 0.10]
])

y_pred_changed = tf.constant([
    [0.60, 0.20, 0.20],
    [0.30, 0.50, 0.20]
])

mse = tf.keras.losses.MeanSquaredError()
cce = tf.keras.losses.CategoricalCrossentropy()

mse_good = mse(y_true, y_pred_good).numpy()
cce_good = cce(y_true, y_pred_good).numpy()

mse_changed = mse(y_true, y_pred_changed).numpy()
cce_changed = cce(y_true, y_pred_changed).numpy()

print("Good predictions:")
print("MSE =", mse_good)
print("CCE =", cce_good)

print("\nChanged predictions:")
print("MSE =", mse_changed)
print("CCE =", cce_changed)

labels = ["MSE", "Cross-Entropy"]
good_losses = [mse_good, cce_good]
changed_losses = [mse_changed, cce_changed]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, good_losses, width, label="Good Predictions")
plt.bar(x + width / 2, changed_losses, width, label="Changed Predictions")
plt.xticks(x, labels)
plt.ylabel("Loss Value")
plt.title("Loss Comparison")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------
# 3. TRAIN MNIST WITH ADAM AND SGD
# ---------------------------------------------------

print("\n--- Task 3: Adam vs. SGD ---")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])


adam_model = create_model()
adam_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

sgd_model = create_model()
sgd_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

adam_history = adam_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=128,
    verbose=1
)

sgd_history = sgd_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=128,
    verbose=1
)

plt.figure(figsize=(8, 5))
plt.plot(adam_history.history["accuracy"], label="Adam Training Accuracy")
plt.plot(adam_history.history["val_accuracy"], label="Adam Validation Accuracy")
plt.plot(sgd_history.history["accuracy"], label="SGD Training Accuracy")
plt.plot(sgd_history.history["val_accuracy"], label="SGD Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST Accuracy: Adam vs. SGD")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------
# 4. TENSORBOARD LOGGING
# ---------------------------------------------------

print("\n--- Task 4: TensorBoard ---")

tensorboard_model = create_model()
tensorboard_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

history = tensorboard_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=128,
    callbacks=[tensorboard_callback],
    verbose=1
)

print("\nTensorBoard logs saved in:", log_dir)
print("Run this command in the terminal:")
print("tensorboard --logdir logs/fit")
