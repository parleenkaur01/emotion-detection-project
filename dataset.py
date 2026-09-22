from pathlib import Path

import librosa
import numpy as np

RAW_DIR = Path("data/raw")
FIXED_FRAMES = 128

# Speaker-independent: 80% of actors → train+val, 20% → test.
all_actors = sorted({p.stem.split("_")[0] for p in RAW_DIR.glob("*.wav")})
n_holdout = int(len(all_actors) * 0.8)
train_val_actors = all_actors[:n_holdout]
TEST_ACTORS = set(all_actors[n_holdout:])
n_val = max(1, int(len(train_val_actors) * 0.15))
VAL_ACTORS = set(train_val_actors[-n_val:])
TRAIN_ACTORS = set(train_val_actors[:-n_val])

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
label_map = {name: i for i, name in enumerate(EMOTIONS)}


def pad_or_crop(mel_db, n_frames=FIXED_FRAMES):
    n_mels, t = mel_db.shape
    out = np.zeros((n_mels, n_frames), dtype=mel_db.dtype)
    use = min(t, n_frames)
    out[:, :use] = mel_db[:, :use]
    return out


X_train, y_train = [], []
X_val, y_val = [], []
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

    if actor in TRAIN_ACTORS:
        X_train.append(fixed)
        y_train.append(label_map[emotion])
    elif actor in VAL_ACTORS:
        X_val.append(fixed)
        y_val.append(label_map[emotion])
    elif actor in TEST_ACTORS:
        X_test.append(fixed)
        y_test.append(label_map[emotion])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Scale using TRAIN stats only.
train_mean = float(X_train.mean())
train_std = float(X_train.std()) or 1.0
X_train = (X_train - train_mean) / train_std
X_val = (X_val - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

print("label map:", label_map)
print(
    f"actors: {len(all_actors)} total  |  "
    f"train {len(TRAIN_ACTORS)} ({min(TRAIN_ACTORS)}–{max(TRAIN_ACTORS)})  |  "
    f"val {len(VAL_ACTORS)} ({min(VAL_ACTORS)}–{max(VAL_ACTORS)})  |  "
    f"test {len(TEST_ACTORS)} ({min(TEST_ACTORS)}–{max(TEST_ACTORS)})"
)
print("train clips:", len(y_train), "  X_train:", X_train.shape)
print("val clips:  ", len(y_val), "  X_val:", X_val.shape)
print("test clips: ", len(y_test), "  X_test:", X_test.shape)
print("train mean (before scale):", round(train_mean, 3), "  train std:", round(train_std, 3))
print("X_train after: mean", round(float(X_train.mean()), 3), " std", round(float(X_train.std()), 3))
print("X_val   after: mean", round(float(X_val.mean()), 3), " std", round(float(X_val.std()), 3))
print("X_test  after: mean", round(float(X_test.mean()), 3), " std", round(float(X_test.std()), 3))
