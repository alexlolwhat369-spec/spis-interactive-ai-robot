# SPIS Interactive AI Robot

An interactive robot that recognizes hand gestures from a camera, shows an
animated face, and holds a short spoken conversation. It can play a 20-questions
style object-guessing game and play music. Everything runs **on-device**: gesture
recognition, speech-to-text, the language model (via Ollama), and text-to-speech
all run locally, with no cloud API during the demo.

> Privacy: the gesture model is trained on **hand-coordinate numbers only** - no
> photos, no faces are ever stored. See [detailed guide](docs/detailed-guide.md#privacy).

For the full explanation (architecture, training, the game, Raspberry Pi
deployment), read the **[detailed guide](docs/detailed-guide.md)**.

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

## Web UI (camera + robot in the browser)

Three scripts wrap the whole flow. The web app **requires a trained gesture
model**, so capture it once with `./train-gestures` before the first `./start`.

```bash
./setup           # .venv (python3.11+), deps, hand model + voice models (~100 MB)
./train-gestures  # 5 short webcam capture sessions, then trains model/gesture_knn.npz
./start           # serve the web UI, then opens http://127.0.0.1:8000
```

The page shows the **live camera** (with hand-landmark overlay) beside the
**animated robot face**; perform a gesture and the robot reacts in real time.
`./start` accepts `--camera <index>` and `--port <n>` (e.g. `./start --camera 1`).
It refuses to boot until the gesture model exists and tells you which script to run.

### Talk to it (hold-to-talk voice)

Hold the **🎤 button** (or the **Spacebar**), speak, and release. Speech is
transcribed locally with Vosk, answered by the robot, and spoken back with Piper
— all on-device. The face shows *listening → thinking → the reply's reaction*.

For real conversation, install [Ollama](https://ollama.com) and create the model:

```bash
ollama create spis-robot -f config/spis-robot.Modelfile
```

Without Ollama the robot automatically falls back to built-in rule replies (still
plays the object-guessing game and reacts to compliments/insults). Voice controls:
`--no-ollama`, `--ollama-model <name>`. If the voice models are missing the page
still runs camera + gestures and simply hides the mic button.

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

- **Gestures:** `wave`, `thumbs_up`, `peace`, `stop`, `heart` (two hands). No hand
  = `none`; an unclear hand shows `unknown` instead of guessing.
- **Talking:** in `interactive_robot.py` / `voice_demo.py`, **hold SPACE** while you
  speak, release to let the robot process the sentence.
- **Face states:** idle, listening, thinking, speaking, happy, proud, confused,
  heart, annoyed, curious.
- **The game:** say `play Akinator`, `play Alkinator`, `twenty questions`, or ask
  it to guess your object. It always tries to guess the object you are thinking of.
- **Music:** ask for music (drop your own tracks in `assets/music/`).

## How to stop

- In any **camera / face window**: press **`Q`**.
- In a **terminal-only** program (`chat_console.py`): type `quit`/`exit` or press
  **Ctrl+C**.
- To leave the Python environment: `deactivate`.

## Run the tests

```bash
python -m pytest
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
