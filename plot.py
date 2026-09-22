import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

path = "data/raw/1001_DFA_HAP_XX.wav"
y, sr = librosa.load(path, sr=16000)

print("samples:", len(y), "seconds:", round(len(y) / sr, 2), "sr:", sr)

mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
mel_db = librosa.power_to_db(mel, ref=np.max)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(y)
axes[0].set_title("waveform")
librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000, ax=axes[1])
axes[1].set_title("mel spectrogram")
plt.tight_layout()
plt.show()
