# SPIS Interactive AI Robot

An interactive robot that recognizes hand gestures from a camera, shows an
animated face, and holds a short spoken conversation. It can play a 20-questions
style object-guessing game and play music. Everything runs **on-device**: gesture
recognition, speech-to-text, the language model (via Ollama), and text-to-speech
all run locally, with no cloud API during the demo.

> Privacy: the gesture model is trained on **hand-coordinate numbers only** - no
> photos, no faces are ever stored. See [detailed guide](docs/detailed-guide.md#privacy).
> The reproducible training dataset is included at `data/landmarks.csv`; each row
> contains one gesture label and 126 numeric hand-landmark values, never an image.

For the full explanation of the architecture, training, game, and laptop demo,
read the **[detailed guide](docs/detailed-guide.md)**.

---

## Requirements

- Python **3.11+**
- A webcam (USB or built-in)
- Optional for voice: [Ollama](https://ollama.com) with a small local model
  (e.g. `llama3.2:1b`), a microphone, and speakers/headphones

## Install

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Download the hand-detection model once (saved locally, no internet after)
python src/setup_assets.py
```

> Only `opencv-contrib-python` should be installed. Do **not** also install
> `opencv-python` - both provide the `cv2` module and will clash.

## Quick start

The repo ships with a trained gesture model, so you can run demos immediately.

```bash
# See the trained gesture model react to your hand (camera window)
python src/live_demo.py

# Preview every face reaction without a camera
python src/face_demo.py

# Chat with the robot by typing (no mic / no Ollama needed)
python src/chat_console.py
```

Full experience (camera + microphone + voice), needs Ollama running:

```bash
python src/interactive_robot.py --ollama-model spis-robot --recognizer vosk --microphone 1
```

## How to use

- **Gestures:** `thumbs_up`, `peace`, `stop`, `heart` (two hands),
  `middle_finger`, `ok`, and the two-hand `mohan` M sign. No hand = `none`; an
  unclear hand shows `unknown` instead of guessing.
- **Talking:** in `interactive_robot.py` / `voice_demo.py`, **hold SPACE** while you
  speak, release to let the robot process the sentence.
- **Face states:** idle, listening, thinking, speaking, happy, proud, confused,
  heart, annoyed, curious, Mohan portrait.
- **The game:** say `play Akinator`, `play Alkinator`, `twenty questions`, or ask
  it to guess your object. It always tries to guess the object you are thinking of,
  and accepts `yes`, `probably`, `maybe`, `probably not`, and `no` throughout.
  The probability engine owns candidates and guesses; Ollama interprets relevant
  natural answers and safely rephrases questions, with deterministic fallbacks.
- **Music:** say `play music`, `play a song`, or the short follow-up `ok play`.
  The robot asks for a category, pauses music while listening, and resumes it after
  unrelated conversation. Say `pause`, `resume`, `next song`, or `stop music` for
  deterministic playback control. Add playable MP3 or WAV tracks to `assets/music/`.
- **Display:** the face is the main window. Press **`D`** to show or hide camera
  diagnostics. The panel shows the latest transcript, selected route, action, microphone
  peak, gesture confidence, and gesture distance. Use `--fullscreen` for a presentation display.

## How to stop

- In the robot window: press **`Q`**.
- In a **terminal-only** program (`chat_console.py`): type `quit`/`exit` or press
  **Ctrl+C**.
- To leave the Python environment: `deactivate`.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Project layout

```text
src/     application + library code (entry points below)
tests/   pytest suite
docs/    detailed guide + design notes
config/  Ollama Modelfile for the custom "spis-robot" model
data/    object catalog + generated data (gitignored)
assets/  music playlist folder
```

Main entry points: `live_demo.py`, `face_demo.py`, `chat_console.py`,
`voice_demo.py`, `interactive_robot.py`, plus tools `collect_samples.py`,
`train.py`, `setup_assets.py`. Each supports `--help`.
