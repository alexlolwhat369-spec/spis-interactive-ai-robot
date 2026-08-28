# SPIS Interactive AI Robot — Detailed Guide

This is the in-depth companion to the [README](../README.md). It covers the
architecture, the data and training pipeline, the conversation brain, the
object-guessing game, the voice stack, privacy, and Raspberry Pi deployment.

- [Overview](#overview)
- [Architecture](#architecture)
- [Module map](#module-map)
- [Installation notes](#installation-notes)
- [Privacy](#privacy)
- [Gesture pipeline](#gesture-pipeline)
  - [Capturing samples](#capturing-samples)
  - [Optional public data (HaGRID)](#optional-public-data-hagrid)
  - [Training and evaluation](#training-and-evaluation)
  - [The live demo gate](#the-live-demo-gate)
- [Faces and reactions](#faces-and-reactions)
- [Conversation brain](#conversation-brain)
- [Object-guessing game](#object-guessing-game)
  - [Human game trials](#human-game-trials)
  - [Training the game questions](#training-the-game-questions)
  - [Reviewing learned suggestions](#reviewing-learned-suggestions)
- [Voice conversation](#voice-conversation)
- [The full interactive robot](#the-full-interactive-robot)
- [Deploying to Raspberry Pi](#deploying-to-raspberry-pi)
- [Testing](#testing)

---

## Overview

The robot recognizes five hand gestures with a camera and triggers a reaction.
The gesture model is trained on **hand coordinates**, not photos or facial
features. Trainable gestures:

- `wave`
- `thumbs_up`
- `peace`
- `stop`
- `heart` (two hands)

The `none` state means no hand is detected. If a hand does not look close enough
to the training examples, the system shows `unknown` instead of guessing.

During the live demo there is an extra gate: a gesture must be **close** to the
trained samples, get **enough votes** from the model, and stay **stable** for a
few frames. This prevents a casual hand from firing a reaction.

## Architecture

```text
Camera -> MediaPipe Hands -> 21 points per hand -> KNN model -> robot reaction
```

The full interactive loop adds voice and a language model:

```text
Camera ─► MediaPipe Hands ─► gesture gate ─┐
                                           ├─► RobotController ─► animated face
Microphone ─► Vosk/Windows STT ─► Ollama ──┘                 └─► Piper/Windows TTS
```

Training can be done on a laptop. Then these files are copied to the Raspberry
Pi 5: `src/`, `requirements.txt`, `models/hand_landmarker.task`, and
`model/gesture_knn.npz`. The Pi runs only `src/live_demo.py` (or the full
`interactive_robot.py`); it does not need the original samples.

## Module map

| File | Role |
|------|------|
| `setup_assets.py` | Download the official MediaPipe hand model once |
| `collect_samples.py` | Capture numeric hand-landmark samples to CSV |
| `import_hagrid.py` | Import up to 200 hand vectors per class from HaGRID annotations |
| `train.py` | Train the portable KNN gesture model + reports |
| `live_demo.py` | Run the trained gesture model against the camera |
| `face_demo.py` | Preview every face reaction without a camera |
| `interactive_robot.py` | Full demo: camera gestures + push-to-talk voice + face |
| `voice_demo.py` | Spoken conversation without gestures |
| `chat_console.py` | Type to the robot brain (no mic, no Ollama required) |
| `game_trial.py` | Record human game trials |
| `train_object_game.py` | Calibrate game questions from trials |
| `review_object_suggestions.py` | Review objects the robot proposed learning |
| `diagnose_camera.py` | Technical camera test (no video saved or shown) |
| `hand_tracker.py`, `gesture_features.py`, `gesture_model.py`, `gesture_gate.py` | Gesture core |
| `robot_state.py`, `robot_face.py`, `robot_runtime.py`, `conversation.py` | Robot brain + rendering |
| `speech.py`, `speech_input.py`, `push_to_talk.py`, `microphone_meter.py` | Voice I/O |
| `object_game.py`, `object_game_training.py`, `object_learning.py`, `object_review.py` | Game logic |
| `music.py` | Local playlist playback |

Every entry point supports `--help`.

## Installation notes

Use Python 3.11 or newer.

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Raspberry Pi OS / Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`opencv-contrib-python` is the **only** OpenCV distribution that should be
installed here. Installing it alongside `opencv-python` can overwrite files of
the `cv2` module.

Then download the official hand-detection model once. The file is saved locally;
detection needs no internet afterwards.

```bash
python src/setup_assets.py            # add --force to replace an existing model
```

## Privacy

The program stores only the gesture name and 126 numeric hand coordinates
(21 points × 3 axes × up to 2 hands). It stores **no frames and no faces**.
Detection runs on the device.

However, the current version of MediaPipe declares that it may send performance
and usage metrics; during local testing a connection attempt for that telemetry
was observed. For a fair without explicit visitor consent, download the
dependencies and the model beforehand and run the Raspberry Pi **offline** during
the demo.

## Gesture pipeline

### Capturing samples

Repeat capture for each gesture. Use good lighting and vary the angle and
distance of the hand a little. `heart` needs both hands fully visible.

```bash
python src/collect_samples.py --label wave --samples 180
python src/collect_samples.py --label thumbs_up --samples 180
python src/collect_samples.py --label peace --samples 180
python src/collect_samples.py --label stop --samples 180
python src/collect_samples.py --label heart --samples 220
```

In the camera window press **`C`** to save a sample and **`Q`** to quit. No frames
are stored — `data/landmarks.csv` holds only the hand-point numbers.

For more comfortable capture add `--auto`: it takes examples only after a short
interval passes and your hands changed enough, avoiding many near-identical copies
of the same pose. Tunables: `--cooldown-ms` (default 350), `--min-distance`
(default 0.18), `--camera` (default 0).

```bash
python src/collect_samples.py --label thumbs_up --samples 180 --auto
```

Capture uses OpenCV's default camera opening — the same path used for the first
working collection. If a new computer or camera misbehaves, `src/diagnose_camera.py`
offers a technical test without saving or showing video (not part of the normal
capture flow).

### Optional public data (HaGRID)

Do **not** download HaGRID images — each image class can take tens of GB. If you
already have the official HaGRID **annotations** in a folder, this importer
extracts up to 200 hand vectors for each of: `like -> thumbs_up`, `peace`,
`stop`, `hand_heart -> heart`. It stores no images, user IDs, or personal
metadata.

```bash
python src/import_hagrid.py --annotations-dir PATH_TO_ANNOTATIONS --max-per-class 200
python src/train.py --dataset data/hagrid_landmarks.csv
```

The result is written to `data/hagrid_landmarks.csv`; keep it separate from
`data/landmarks.csv`, which holds your voluntary captures. For the final demo,
also use samples from your own camera, especially for `wave` — a motion gesture
with no equivalent class in HaGRID.

### Training and evaluation

```bash
python src/train.py
```

Flags: `--dataset` (default `data/landmarks.csv`), `--model`
(default `model/gesture_knn.npz`), `--test-fraction` (0.2), `--seed` (42),
`--k` (5). It creates:

- `model/gesture_knn.npz` — the model for the demo and the Raspberry Pi.
- `reports/evaluation.json` — accuracy and per-gesture metrics.
- `reports/confusion_matrix.csv` — the confusion matrix.

Do not present a single accuracy number. Also show the confusion matrix, how many
samples exist per class, and error examples. The last local evaluation used 969
samples: **97.4%** accuracy over 194 held-out samples. The demo gate keeps 181 of
those correct samples and rejects poses that are not close enough.

### The live demo gate

```bash
python src/live_demo.py                 # --camera 0, --model model/gesture_knn.npz
```

`heart` draws a blue heart and fires the `heart` reaction. The animated face
opens a second **800×480** window, suitable for a small HDMI screen. Voice is
enabled only if the system has `espeak-ng`/`espeak`; the mic + speech recognition
integration is validated with the hardware. Press **`Q`** to quit.

## Faces and reactions

The robot has ten reactions (`robot_state.py`):
`idle`, `listening`, `thinking`, `speaking`, `happy`, `proud`, `confused`,
`heart`, `annoyed`, `curious`, plus actions `start_game`, `stop`, `play_music`.

Preview every reaction without a webcam:

```bash
python src/face_demo.py
```

The screen advances automatically. Use ← / → or `A` / `D` to change reaction and
**`Q`** to close.

## Conversation brain

Test the robot's brain without a microphone or Ollama:

```bash
python src/chat_console.py
```

Add `--speak` to have the robot read its replies with the machine's local voice
(Windows uses the built-in synthesizer; Raspberry Pi uses `espeak-ng` when
installed). This test still takes typed text; the mic and transcription are
connected after the device is validated.

```bash
python src/chat_console.py --speak
```

The local mode understands greetings, jokes, praise, mild insults, confusion,
interest, a music request, the object game, and stopping. It reacts with a happy
face to praise, an annoyed-but-kind face to insults, a confused face when it does
not understand, and star eyes for clear interest. To use a model already pulled
by Ollama:

```bash
python src/chat_console.py --ollama-model MODEL_NAME
```

The app validates that Ollama returns an allowed reply, reaction, and action. If
Ollama is unavailable, it falls back to local mode.

## Object-guessing game

The game chooses questions by information gain and accepts `yes`, `probably`,
`maybe`, `probably not`, and `no`. It uses `data/object_catalog.json`; you can
extend the list **without retraining any model**. The current curated catalog has
60 objects, and the algorithm has no hard object limit.

If the game misses a guess, the robot asks for the correct object. If that object
is already in the catalog, it anonymously stores the round's answers in
`data/game_trials.jsonl` to improve questions after review. If it is a new object,
it also asks for a feature that distinguishes it from the wrong object and stores
the suggestion in `data/pending_object_suggestions.jsonl` for human review before
adding it to the catalog.

Music uses `assets/music/playlist.json`, but you must place original or licensed
tracks in that folder.

### Human game trials

A simulation confirms the data is consistent but does not replace a real person
answering questions. For a test round, a tester privately picks one object,
answers the questions, and records only the normalized answers, the guess, and an
optional note about confusing questions. No audio, video, or names are stored.

```bash
python src/game_trial.py --list-targets
python src/game_trial.py --target laptop --trial-id round-01
```

Local results go to `data/game_trials.jsonl`. Do 10–15 rounds with different
objects before the fair and reword any questions people flag as confusing.

### Training the game questions

When you have enough rounds, first review the training without changing the robot:

```bash
python src/train_object_game.py --dry-run
```

Training holds out part of the rounds for evaluation. It calibrates a question
only when it has at least eight answers for objects that **do** have the feature
and eight for objects that **do not**. Running without `--dry-run` writes the
calibration to `data/object_game_calibration.json` and a report to
`reports/object_game_training.json`. The robot loads that calibration
automatically in the next game.

```bash
python src/train_object_game.py
```

### Reviewing learned suggestions

When the robot fails, its suggestion is kept **separate** from the catalog. List
and review first:

```bash
python src/review_object_suggestions.py
```

To approve, an adult or the team must choose a category, write the attributes, and
define a clear question. The command refuses a suggestion without that explicit
review:

```bash
python src/review_object_suggestions.py --approve 1 \
  --category technology \
  --attributes electronic,fits_hand,temperature_sensor \
  --attribute temperature_sensor \
  --question "Is it mainly used to measure temperature?"
```

To discard a bad suggestion:

```bash
python src/review_object_suggestions.py --reject 1
```

For a laptop and a first Raspberry Pi 5 test, the recommended model is
`llama3.2:1b`. Ollama downloads it once and then serves conversations on
`localhost` — this is not training from scratch. The Pi needs enough memory and a
real speed test before the fair.

## Voice conversation

The spoken demo uses the English recognizer already installed on Windows for the
laptop, and **Vosk** locally on Raspberry Pi. The default voice is **Piper** with
the local neural voice `en_US-lessac-medium`, softer and more natural than the
classic Windows voice. The voice model downloads once, then audio does not go to
any API. Use headphones so the mic does not hear the speakers during laptop tests.

```bash
python src/voice_demo.py --ollama-model llama3.2:1b
```

Fallback to the classic Windows voice:

```bash
python src/voice_demo.py --tts windows
```

List and pick a microphone / recognizer:

```bash
python src/voice_demo.py --list-microphones
python src/voice_demo.py --microphone 1
python src/voice_demo.py --recognizer vosk
```

Other flags: `--piper-voice`, `--voice`, `--voice-rate` (energy −10..10, default 4),
`--listen-seconds` (default 12), `--no-face`.

## The full interactive robot

For the fair, use the unified mode: the camera keeps detecting gestures and the
face keeps `listening`, `thinking`, and `speaking` during the conversation.
**Hold the space bar** while you talk and release it so the robot processes the
sentence. During conversation, gestures do not change the face; afterwards you must
remove your hands from the camera before making a new sign. Outside the
conversation, `heart` stays visible for 1.5 s so it does not vanish if your hands
leave the frame.

```bash
python src/interactive_robot.py --ollama-model spis-robot --recognizer vosk --microphone 1
```

Press **`Q`** in either window to quit. Key flags: `--model`, `--camera`,
`--ollama-model` (default `spis-robot`), `--speech-model`, `--microphone`,
`--recognizer {auto,windows,vosk}`, `--tts {piper,windows}`, `--piper-voice`,
`--voice`, `--voice-rate`, `--listen-seconds`.

The `spis-robot` model is a local configuration of `llama3.2:1b` with the
project's rules and examples, created with `config/spis-robot.Modelfile`. It is
**not** a from-scratch weight training; a real weight fine-tune would need a large
set of voluntary, labeled, evaluated dialogues.

## Deploying to Raspberry Pi

1. Copy the project folder to the Pi via USB, Git, or the local network.
2. Install the dependencies and run `python src/setup_assets.py` on the Pi.
3. Connect the official camera and test `python src/live_demo.py --camera 0`.
4. If the camera uses another index, change `--camera` to `1` or `2`.

The first test should use a USB webcam or the Raspberry camera configured as a
V4L2 device. `Picamera2` integration is added once we have the physical Pi, to
avoid writing code that cannot be tested against real hardware.

## Testing

```bash
python -m pytest
```

The `tests/` suite covers gesture features, the gesture gate and model, camera and
data I/O, the conversation brain, the object game and its training, learning and
review, speech input, push-to-talk, music, the robot face, runtime, and state.
