"""
CNN Image Classification on CIFAR-10
=====================================
Author  : Sasikala Madhu
Dataset : CIFAR-10 (10 classes, 60,000 images)
Framework: TensorFlow / Keras
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# ------------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------------
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Training set shape : {x_train.shape}")   # (50000, 32, 32, 3)
print(f"Test set shape     : {x_test.shape}")     # (10000, 32, 32, 3)

# CIFAR-10 class labels
CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# ------------------------------------------------------------------
# 2. Visualise First 24 Training Images
# ------------------------------------------------------------------
fig = plt.figure(figsize=(20, 5))
for i in range(24):
    ax = fig.add_subplot(3, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(x_train[i])
    ax.set_title(CIFAR10_LABELS[int(y_train[i])])
plt.suptitle("Sample Training Images", fontsize=14)
plt.tight_layout()
plt.savefig("sample_images.png", dpi=100)
plt.show()
print("Sample images saved to sample_images.png")

# ------------------------------------------------------------------
# 3. Normalise Pixel Values (0–255 → 0.0–1.0)
# ------------------------------------------------------------------
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

# ------------------------------------------------------------------
# 4. One-Hot Encode Labels
# ------------------------------------------------------------------
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)

print(f"Number of classes  : {num_classes}")

# ------------------------------------------------------------------
# 5. Split Training Set into Train + Validation
# ------------------------------------------------------------------
x_valid, x_train = x_train[:5000], x_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

print(f"Training samples   : {x_train.shape[0]}")
print(f"Validation samples : {x_valid.shape[0]}")
print(f"Test samples       : {x_test.shape[0]}")

# ------------------------------------------------------------------
# 6. Build CNN Model Architecture
# ------------------------------------------------------------------
model = Sequential([
    # Block 1
    Conv2D(filters=16, kernel_size=2, padding='same', activation='relu',
           input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=2),

    # Block 2
    Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    # Block 3
    Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Dropout(0.3),

    # Fully Connected Layers
    Flatten(),
    Dense(500, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

model.summary()

# ------------------------------------------------------------------
# 7. Compile the Model
# ------------------------------------------------------------------
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# ------------------------------------------------------------------
# 8. Train the Model
# ------------------------------------------------------------------
checkpointer = ModelCheckpoint(
    filepath='model.weights.best.keras',
    save_best_only=True,
    verbose=1
)

print("\nStarting training...")
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=15,
    validation_data=(x_valid, y_valid),
    callbacks=[checkpointer],
    verbose=1,
    shuffle=True
)

# ------------------------------------------------------------------
# 9. Plot Training History
# ------------------------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png", dpi=100)
plt.show()
print("Training history plot saved to training_history.png")

# ------------------------------------------------------------------
# 10. Evaluate on Test Set
# ------------------------------------------------------------------
model.load_weights('model.weights.best.keras')
score = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss     : {score[0]:.4f}")
print(f"Test Accuracy : {100 * score[1]:.2f}%")

# ------------------------------------------------------------------
# 11. Predict on Test Images
# ------------------------------------------------------------------
y_hat = model.predict(x_test)

# Visualise 9 test predictions
fig = plt.figure(figsize=(12, 8))
for idx in range(9):
    ax = fig.add_subplot(3, 3, idx + 1, xticks=[], yticks=[])
    ax.imshow(x_test[idx])
    predicted_label = CIFAR10_LABELS[np.argmax(y_hat[idx])]
    true_label      = CIFAR10_LABELS[np.argmax(y_test[idx])]
    color = 'green' if predicted_label == true_label else 'red'
    ax.set_title(f"Pred: {predicted_label}\nTrue: {true_label}", color=color)

plt.suptitle("Test Predictions (Green=Correct, Red=Wrong)", fontsize=13)
plt.tight_layout()
plt.savefig("test_predictions.png", dpi=100)
plt.show()
print("Test predictions saved to test_predictions.png")
