import numpy as np
import dataset
from tensorflow import keras
from tensorflow.keras import layers

X_train = dataset.X_train[..., np.newaxis]
X_val = dataset.X_val[..., np.newaxis]
X_test = dataset.X_test[..., np.newaxis]
y_train = dataset.y_train
y_val = dataset.y_val
y_test = dataset.y_test
num_classes = len(dataset.EMOTIONS)
n_mels, n_frames = dataset.X_train.shape[1], dataset.X_train.shape[2]

print("CNN input train/val/test:", X_train.shape, X_val.shape, X_test.shape)

# Pool frequency harder than time, then LSTM reads the remaining time steps.
# Input is (mel bins, time frames, 1). After the pools: freq=8, time=32, channels=32.
model = keras.Sequential(
    [
        layers.Input(shape=(n_mels, n_frames, 1)),
        layers.GaussianNoise(0.05),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((4, 1)),
        layers.Permute((2, 1, 3)),
        layers.Reshape((n_frames // 4, (n_mels // 16) * 32)),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    ),
]

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Val  loss: {val_loss:.3f}   val accuracy:  {val_acc * 100:.1f}%")
print(f"Test loss: {loss:.3f}   test accuracy: {acc * 100:.1f}%")
print(
    f"(speaker-independent: {len(y_train)} train / {len(y_val)} val / "
    f"{len(y_test)} test clips — unseen val/test voices.)"
)
