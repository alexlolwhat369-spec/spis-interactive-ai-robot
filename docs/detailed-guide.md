# SPIS Interactive AI Robot — Detailed Guide

This is the in-depth companion to the [README](../README.md). It covers the
architecture, the data and training pipeline, the conversation brain, the
object-guessing game, the voice stack, privacy, and laptop operation.

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
- [Laptop deployment](#laptop-deployment)
- [Testing](#testing)

---

## Overview

The deployed model recognizes seven hand gestures with a camera and triggers a reaction.
The gesture model is trained on **hand coordinates**, not photos or facial
features. Trainable gestures:

- `thumbs_up`
- `peace`
- `stop`
- `heart` (two hands)
- `middle_finger`
- `ok`
- `mohan` (two peace-shaped hands joined to form an M)

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

Training and the full demonstration both run on the laptop. The live robot loads
`models/hand_landmarker.task` and `model/gesture_knn.npz`; it does not need the
original capture samples during a demonstration.

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

Linux / macOS:

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
dependencies and the model beforehand and run the laptop **offline** during
the demo.

## Gesture pipeline

### Capturing samples

Repeat capture for each gesture. Use good lighting and vary the angle and
distance of the hand a little. `heart` needs both hands fully visible.

```bash
python src/collect_samples.py --label thumbs_up --samples 180
python src/collect_samples.py --label peace --samples 180
python src/collect_samples.py --label stop --samples 180
python src/collect_samples.py --label heart --samples 220
python src/collect_samples.py --label middle_finger --samples 180 --auto
python src/collect_samples.py --label ok --samples 180 --auto
python src/collect_samples.py --label mohan --samples 220 --auto
```

For `mohan`, show both hands with the index and middle fingers extended. Touch
the two inner fingertips to form the center of the M, as in the reference pose,
and vary distance, wrist rotation, and height without changing that geometry.
It gets 220 samples because it can be confused with two separate peace signs.
For `middle_finger`, collect only with volunteers who agree to perform the gesture.

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
`stop`, `hand_heart -> heart`, `middle_finger`, and `ok`. The custom `mohan`
pose still needs your own camera samples. It stores no images, user IDs, or
personal metadata.

```bash
python src/import_hagrid.py --annotations-dir PATH_TO_ANNOTATIONS --max-per-class 200
python src/train.py --dataset data/hagrid_landmarks.csv
```

The result is written to `data/hagrid_landmarks.csv`; keep it separate from
`data/landmarks.csv`, which holds your voluntary captures. For the final demo,
also use samples from the same laptop camera and environment used for the event.

### Training and evaluation

```bash
python src/train.py
```

Flags: `--dataset` (default `data/landmarks.csv`), `--model`
(default `model/gesture_knn.npz`), `--test-fraction` (0.2), `--seed` (42),
`--k` (5). It creates:

- `model/gesture_knn.npz` — the model used by the laptop demo.
- `reports/evaluation.json` — accuracy and per-gesture metrics.
- `reports/confusion_matrix.csv` — the confusion matrix.

Do not present a single accuracy number. Also show the confusion matrix, how many
samples exist per class, and error examples. The current local evaluation uses
1,550 samples and reports **93.87%** accuracy over 310 held-out samples. `stop`
has the lowest recall at 83.9%, so it remains the first gesture to improve.

### The live demo gate

```bash
python src/live_demo.py                 # --camera 0, --model model/gesture_knn.npz
```

`heart` draws a blue heart and fires the `heart` reaction. `middle_finger`
shows the annoyed face, `ok` shows the proud face, and `mohan` shows the supplied
Mohan portrait and name. The animated face
opens a second **800×480** window, suitable for a small HDMI screen. Voice is
enabled only if the system has `espeak-ng`/`espeak`; the mic + speech recognition
integration is validated with the hardware. Press **`Q`** to quit.

## Faces and reactions

The robot has eleven reactions (`robot_state.py`):
`idle`, `listening`, `thinking`, `speaking`, `happy`, `proud`, `confused`,
`heart`, `annoyed`, `curious`, `mohan`, plus actions `start_game`, `stop`,
`play_music`.

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

Add `--speak` to have the robot read its replies with the machine's local voice.
Windows uses Piper first and the built-in synthesizer as fallback. This test still
takes typed text; the mic and transcription are connected in the full robot.

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

The live game is hybrid, but the two parts have strict boundaries. The
probability engine owns the object list, question order, scores, exact deductions,
question budget, and the rule that the robot always guesses the visitor's object.
Ollama may interpret a relevant natural sentence such as `I use it to move the
cursor` and may rephrase the next catalog question. It cannot add candidates,
change scores, skip confirmation, or select the final guess. Invalid, unrelated,
low-confidence, or malformed model output is ignored and the canonical question
is used instead.

Question order is hierarchical. Category probes establish technology, school,
food/drink, or play/mobility first. Once one category owns at least 65% of the
probability, only details from that branch and genuinely cross-category details
may be asked. The robot announces the active branch and repeats the normalized
answer it understood, making a speech-recognition mistake visible immediately.

Simple answers are normalized locally without waiting for Ollama. When semantic
game input is available, Vosk uses its full vocabulary instead of the old closed
five-answer grammar. Ollama output must match one of the five answers with at
least 70% confidence and pass a relevance check. An exact answer to an attribute
owned by only one catalog object becomes an immediate, explainable deduction.

All five answers remain valid when the robot presents its final guess. `Probably`
finishes as a close, unconfirmed match, `maybe` asks one more question, and
`probably not` moves to correction instead of being rejected as invalid input.

Start it by saying `play Akinator`, `play Alkinator`, `start the object game`,
`twenty questions`, or `guess what I am thinking`. If a spoken answer does not
look like yes/no/maybe, the robot asks you to repeat it without consuming the
question. Asking to play again starts a clean round.

If the game misses a guess, the robot asks for the correct object. If that object
is already in the catalog, it anonymously stores the round's answers in
`data/game_trials.jsonl` to improve questions after review. If it is a new object,
Ollama may propose one feature question that distinguishes it from the wrong
object. The visitor must approve that question or provide a better one. The result
is stored in `data/pending_object_suggestions.jsonl` for human review and is never
added directly to the trusted catalog.

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

The laptop uses `llama3.2:1b`. Ollama downloads it once and then serves
conversations on `localhost`; this is prompt configuration, not training from scratch.

## Voice conversation

The spoken demo uses **Vosk** locally on the laptop. For guided game turns it runs a
free transcription and a short-answer transcription over the same temporary audio
stream. A clear guided answer repairs phrases such as `probably not`, while a richer
sentence remains available to the semantic game interpreter. The default voice is **Piper** with
the local neural voice `en_US-lessac-medium`, softer and more natural than the
classic Windows voice. The voice model downloads once, then audio does not go to
any API. Installed music pauses during a microphone turn and resumes after an
unrelated reply so the recognizer does not compete with the speakers.

The Ollama provider always keeps the system rules plus the four most recent
conversation turns, so later questions can refer to recent details. It also
guards physical actions: only an explicit game phrase can start Akinator, while
`play`, `ok play`, `play it`, `song`, and `music` route to music. Messages with
multiple questions are explicitly answered as multiple parts.

Before Ollama sees a turn, the runtime gives it exactly one route: conversation,
game start, game answer, music request, music category, or music control. Playback
commands `pause`, `resume`, `next song`, and `stop music` are local and deterministic.

During the full demo, the terminal prints `Heard (not saved): ...` and
`Robot says: ...`. This transcript is only diagnostic console output and is not
written to a file. It separates microphone transcription errors from language
model or TTS errors during rehearsal.

```bash
python src/voice_demo.py --ollama-model llama3.2:1b
```

Fallback to the classic Windows voice:

```bash
python src/voice_demo.py --tts windows --voice auto
```

The `auto` option checks the installed voices and prefers names that indicate a
natural or neural voice before falling back to Microsoft Zira. Piper and the
Windows fallback both clean Markdown, links, code markers, and URLs before
speaking so the robot does not read formatting symbols aloud. This lightweight
selection adapts the TTS approach from the MIT-licensed
[`tsaristov/business_suite`](https://github.com/tsaristov/business_suite); it
does not copy a voice model or require that web application.

In the full robot, Piper is the primary voice and the Windows synthesizer is an
automatic fallback. A failure in either voice leaves the subtitle available and
does not freeze the listening loop.

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
hold a fresh stable sign before it activates. One uncertain camera frame no longer
restarts an otherwise stable pose. Outside the
conversation, `heart` stays visible for 1.5 s and `mohan` for 2 s so they do not
vanish as soon as your hands leave the frame.

Music playback is local and only uses files listed in `assets/music/playlist.json`.
The playlist supports MP3 and WAV files. Pygame keeps the current song paused while
the visitor speaks and resumes from the same position after unrelated conversation.
The platform player remains a compatibility fallback if Pygame is unavailable.

```bash
python src/interactive_robot.py --ollama-model spis-robot --recognizer vosk --microphone 1
```

The face is the only window shown by default. Press **`D`** to show or hide camera
diagnostics and **`Q`** to quit. Diagnostics show what Vosk heard, the selected route,
the resulting action, microphone peak level, and gesture scores. Nothing is written
unless `--diagnostic-log data/turn_diagnostics.jsonl` is supplied; even then only text
and numeric levels are saved, never microphone audio or camera images. Use
`--fullscreen` for presentation mode. Key flags:
`--model`, `--camera`, `--debug-camera`, `--fullscreen`,
`--ollama-model` (default `spis-robot`), `--speech-model`, `--microphone`,
`--recognizer {auto,windows,vosk}`, `--tts {piper,windows}`, `--piper-voice`,
`--voice`, `--voice-rate`, `--listen-seconds`, `--diagnostic-log`.

The `spis-robot` model is a local configuration of `llama3.2:1b` with the
project's rules and examples, created with `config/spis-robot.Modelfile`. It is
**not** a from-scratch weight training; a real weight fine-tune would need a large
set of voluntary, labeled, evaluated dialogues.

## Laptop deployment

This version targets the project laptop: webcam, microphone, speakers, Vosk,
Piper, and Ollama all run locally. No Raspberry Pi, GPIO, motor, or separate
display setup is required for the current demonstration scope.

## Testing

```bash
python -m unittest discover -s tests -v
```

The `tests/` suite covers gesture features, the gesture gate and model, camera and
data I/O, the conversation brain, the object game and its training, learning and
review, speech input, push-to-talk, music, the robot face, runtime, and state.
