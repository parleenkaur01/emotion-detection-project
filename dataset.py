from pathlib import Path

import librosa
import numpy as np

RAW_DIR = Path("data/raw")
FIXED_FRAMES = 64
TRAIN_ACTOR = "1001"
TEST_ACTOR = "1002"

# Turn emotion text into a number. Same mapping every time (sorted names).
EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
label_map = {name: i for i, name in enumerate(EMOTIONS)}


def pad_or_crop(mel_db, n_frames=FIXED_FRAMES):
    n_mels, t = mel_db.shape
    out = np.zeros((n_mels, n_frames), dtype=mel_db.dtype)
    use = min(t, n_frames)
    out[:, :use] = mel_db[:, :use]
    return out


X_train, y_train = [], []
X_test, y_test = [], []

for path in sorted(RAW_DIR.glob("*.wav")):
    actor, sentence, emotion, intensity = path.stem.split("_")
    if emotion not in label_map:
        print("skip unknown emotion", path.name)
        continue

    y, sr = librosa.load(path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fixed = pad_or_crop(mel_db)

    if actor == TRAIN_ACTOR:
        X_train.append(fixed)
        y_train.append(label_map[emotion])
    elif actor == TEST_ACTOR:
        X_test.append(fixed)
        y_test.append(label_map[emotion])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Scale using TRAIN stats only, then apply the same numbers to test.
train_mean = float(X_train.mean())
train_std = float(X_train.std())
if train_std == 0:
    train_std = 1.0

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

print("label map:", label_map)
print("train clips:", len(y_train), "  X_train:", X_train.shape, "  y_train:", y_train.shape)
print("test clips:", len(y_test), "  X_test:", X_test.shape, "  y_test:", y_test.shape)
print("y_train (numbers):", y_train)
print("y_test  (numbers):", y_test)
print("train mean (before scale):", round(train_mean, 3), "  train std:", round(train_std, 3))
print("X_train after: mean", round(float(X_train.mean()), 3), " std", round(float(X_train.std()), 3))
print("X_test  after: mean", round(float(X_test.mean()), 3), " std", round(float(X_test.std()), 3))
