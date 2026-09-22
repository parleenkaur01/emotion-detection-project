from pathlib import Path

import librosa
import numpy as np

RAW_DIR = Path("data/raw")
# ~2 seconds of spectrogram: 16_000 Hz / hop 512 ≈ 31 frames per second, 31*2 ≈ 64.
# Our CREMA-D clips are about 1.5–3 seconds, so 64 crops a bit or pads a bit — not a huge empty canvas.
FIXED_FRAMES = 64


def pad_or_crop(mel_db, n_frames=FIXED_FRAMES):
    n_mels, t = mel_db.shape
    out = np.zeros((n_mels, n_frames), dtype=mel_db.dtype)
    use = min(t, n_frames)
    out[:, :use] = mel_db[:, :use]
    return out


print(
    f"{'file':32}  {'actor':6}  {'emotion':8}  {'seconds':8}  "
    f"{'raw_shape':12}  padded_shape"
)
print("-" * 100)

for path in sorted(RAW_DIR.glob("*.wav")):
    actor, sentence, emotion, intensity = path.stem.split("_")
    y, sr = librosa.load(path, sr=16000)
    seconds = round(len(y) / sr, 2)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fixed = pad_or_crop(mel_db)

    print(
        f"{path.name:32}  {actor:6}  {emotion:8}  {seconds:8}  "
        f"{str(mel_db.shape):12}  {fixed.shape}"
    )
