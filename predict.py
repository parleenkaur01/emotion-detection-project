import sys
from pathlib import Path

import librosa
import numpy as np
import dataset
from tensorflow import keras

if len(sys.argv) != 2:
    print("usage: python predict.py path/to/clip.wav")
    raise SystemExit(1)

path = Path(sys.argv[1])
if not path.is_file():
    print("file not found:", path)
    raise SystemExit(1)

audio, sr = librosa.load(path, sr=16000)
mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
mel_db = librosa.power_to_db(mel, ref=np.max)
fixed = dataset.pad_or_crop(mel_db)
fixed = (fixed - dataset.train_mean) / dataset.train_std
sample = fixed[np.newaxis, ..., np.newaxis]

model = keras.models.load_model("emotion_cnn.keras")
probs = model.predict(sample, verbose=0)[0]
order = np.argsort(probs)[::-1]
first, second = int(order[0]), int(order[1])
guess = dataset.EMOTIONS[first]
# Less than 10 points between the top two means the model is not sure.
unsure = float(probs[first] - probs[second]) < 0.10

parts = path.stem.split("_")
emotion = parts[2] if len(parts) >= 3 and parts[2] in dataset.label_map else None
actor = parts[0] if parts else ""
if actor in dataset.TEST_ACTORS:
    split = "unseen test voice"
elif actor in dataset.VAL_ACTORS:
    split = "unseen val voice"
elif actor in dataset.TRAIN_ACTORS:
    split = "train voice"
else:
    split = "voice not in this dataset"

print(f"file: {path.name}   actor {actor} ({split})")
if emotion:
    print(f"true emotion: {emotion}")
if unsure:
    other = dataset.EMOTIONS[second]
    print(
        f"unsure: {guess} {probs[first] * 100:.1f}%  or  "
        f"{other} {probs[second] * 100:.1f}%"
    )
else:
    print(f"guess: {guess}")
for name, prob in zip(dataset.EMOTIONS, probs):
    print(f"  {name}  {prob * 100:5.1f}%")
