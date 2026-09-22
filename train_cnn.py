import numpy as np
import dataset
from tensorflow import keras
from tensorflow.keras import layers

# Keras wants (clips, height, width, channels). We have 1 channel.
X_train = dataset.X_train[..., np.newaxis]
X_test = dataset.X_test[..., np.newaxis]
y_train = dataset.y_train
y_test = dataset.y_test
num_classes = len(dataset.EMOTIONS)

print("CNN input:", X_train.shape, X_test.shape)

model = keras.Sequential(
    [
        layers.Input(shape=(128, 64, 1)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.3f}   test accuracy: {acc * 100:.1f}%")
print("(33 clips, 2 speakers — this is a pipeline check, not a real score.)")
