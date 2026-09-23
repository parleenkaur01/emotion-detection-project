# Emotion detection from speech

Guess the emotion in a short voice clip: angry, disgust, fear, happy, neutral, or sad.

The clips are from [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D). Each file is one actor saying a fixed sentence in one emotion. The filename holds the answer, for example `1002_DFA_HAP_XX.wav`:

| Piece | Meaning |
|---|---|
| `1002` | who spoke |
| `DFA` | which sentence |
| `HAP` | emotion |
| `XX` | intensity (not used) |

Put the `.wav` files in `data/raw/`. That folder is not in git.

## How the voices are split

The test speakers are never used in training, so the score is not “did it memorize this person.”

| Split | Actors | Role |
|---|---|---|
| Train | `1001`–`1062` | the model learns from these voices |
| Val | `1063`–`1072` | used to stop training |
| Test | `1073`–`1091` | unseen voices, reported accuracy |

Each clip becomes a mel spectrogram (128 frequency bins by 128 time frames). A small CNN plus a bidirectional LSTM reads that picture and outputs six scores.

## Run

From the project folder, with the virtualenv active:

```bash
python train_cnn.py
```

This trains, prints test accuracy and a confusion matrix, and saves:

- `emotion_cnn.keras` — the trained model
- `norm_stats.npz` — the training mean and standard deviation, so a new clip is scaled the same way

Guess one file without training again:

```bash
python predict.py data/raw/1002_DFA_HAP_XX.wav
```

If the top two scores are less than 10 points apart, the script prints `unsure` and both emotions. Otherwise it prints one `guess`.

## Result

On the unseen test speakers the model reaches about **55%** accuracy. Chance for six emotions is about 17%.

Angry is the clearest class. Happy, fear, and disgust are the ones it mixes up. Happy and fear are both loud and tense, and the words in a CREMA-D pair are the same, so only the tone can separate them. A close call looks like:

```text
true emotion: HAP
unsure: FEA 49.3%  or  HAP 46.8%
```

## Other scripts

- `plot.py` — waveform and mel spectrogram for one file
- `features.py` — duration and spectrogram shape for every file
- `dataset.py` — builds the train, val, and test arrays and writes `norm_stats.npz`
