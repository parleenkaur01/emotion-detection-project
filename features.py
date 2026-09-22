from pathlib import Path

import librosa
import numpy as np

RAW_DIR = Path("data/raw")

print(f"{'file':32}  {'actor':6}  {'emotion':8}  {'seconds':8}  shape")
print("-" * 80)

for path in sorted(RAW_DIR.glob("*.wav")):
    actor, sentence, emotion, intensity = path.stem.split("_")
    y, sr = librosa.load(path, sr=16000)
    seconds = round(len(y) / sr, 2)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    print(f"{path.name:32}  {actor:6}  {emotion:8}  {seconds:8}  {mel_db.shape}")
