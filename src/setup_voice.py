"""Download the local voice models the web UI needs: Vosk STT + Piper TTS.

Kept separate from the tiny hand model in ``setup_assets.py`` because these are
larger (~100 MB together). Everything stays on-device after this runs once.
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

VOSK_DIR = MODELS / "vosk-model-small-en-us-0.15"
VOSK_ZIP_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

PIPER_DIR = MODELS / "voices"
PIPER_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
)
PIPER_FILES = ("en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def get_vosk(force: bool) -> None:
    if VOSK_DIR.is_dir() and not force:
        print(f"Vosk model already exists: {VOSK_DIR}")
        return
    MODELS.mkdir(parents=True, exist_ok=True)
    archive = MODELS / "vosk-model-small-en-us-0.15.zip"
    _download(VOSK_ZIP_URL, archive)
    print("Extracting Vosk model")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(MODELS)
    archive.unlink(missing_ok=True)
    if not VOSK_DIR.is_dir():
        raise RuntimeError(f"Vosk archive did not contain {VOSK_DIR.name}.")
    print(f"Saved Vosk model: {VOSK_DIR}")


def get_piper(force: bool) -> None:
    for name in PIPER_FILES:
        destination = PIPER_DIR / name
        if destination.is_file() and not force:
            print(f"Piper file already exists: {destination}")
            continue
        _download(PIPER_BASE_URL + name, destination)
    print(f"Saved Piper voice to: {PIPER_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Vosk and Piper voice models.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    args = parser.parse_args()
    get_vosk(args.force)
    get_piper(args.force)
    print("Voice models ready.")


if __name__ == "__main__":
    main()
