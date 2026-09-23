import sys
from pathlib import Path

import librosa
import numpy as np
from tensorflow import keras

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
LABELS = {name: i for i, name in enumerate(EMOTIONS)}
FIXED_FRAMES = 128
RAW_DIR = Path("data/raw")

# Same speaker split as dataset.py, using filenames only.
all_actors = sorted({p.stem.split("_")[0] for p in RAW_DIR.glob("*.wav")})
n_holdout = int(len(all_actors) * 0.8)
train_val_actors = all_actors[:n_holdout]
TEST_ACTORS = set(all_actors[n_holdout:])
n_val = max(1, int(len(train_val_actors) * 0.15))
VAL_ACTORS = set(train_val_actors[-n_val:])
TRAIN_ACTORS = set(train_val_actors[:-n_val])


def pad_or_crop(mel_db, n_frames=FIXED_FRAMES):
    n_mels, t = mel_db.shape
    out = np.zeros((n_mels, n_frames), dtype=mel_db.dtype)
    use = min(t, n_frames)
    out[:, :use] = mel_db[:, :use]
    return out


if len(sys.argv) != 2:
    print("usage: python predict.py path/to/clip.wav")
    raise SystemExit(1)

stats_path = Path("norm_stats.npz")
if not stats_path.is_file():
    print("missing norm_stats.npz — run python dataset.py once to save the training mean and std")
    raise SystemExit(1)

path = Path(sys.argv[1])
if not path.is_file():
    print("file not found:", path)
    raise SystemExit(1)

stats = np.load(stats_path)
train_mean = float(stats["train_mean"])
train_std = float(stats["train_std"])

audio, sr = librosa.load(path, sr=16000)
mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
mel_db = librosa.power_to_db(mel, ref=np.max)
fixed = pad_or_crop(mel_db)
fixed = (fixed - train_mean) / train_std
sample = fixed[np.newaxis, ..., np.newaxis]

model = keras.models.load_model("emotion_cnn.keras")
probs = model.predict(sample, verbose=0)[0]
order = np.argsort(probs)[::-1]
first, second = int(order[0]), int(order[1])
guess = EMOTIONS[first]
# Less than 10 points between the top two means the model is not sure.
unsure = float(probs[first] - probs[second]) < 0.10

parts = path.stem.split("_")
emotion = parts[2] if len(parts) >= 3 and parts[2] in LABELS else None
actor = parts[0] if parts else ""
if actor in TEST_ACTORS:
    split = "unseen test voice"
elif actor in VAL_ACTORS:
    split = "unseen val voice"
elif actor in TRAIN_ACTORS:
    split = "train voice"
else:
    split = "voice not in this dataset"

print(f"file: {path.name}   actor {actor} ({split})")
if emotion:
    print(f"true emotion: {emotion}")
if unsure:
    other = EMOTIONS[second]
    print(
        f"unsure: {guess} {probs[first] * 100:.1f}%  or  "
        f"{other} {probs[second] * 100:.1f}%"
    )
else:
    print(f"guess: {guess}")
for name, prob in zip(EMOTIONS, probs):
    print(f"  {name}  {prob * 100:5.1f}%")
