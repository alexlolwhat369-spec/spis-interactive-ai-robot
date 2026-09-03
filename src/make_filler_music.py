"""Generate short, royalty-free filler music clips — one per playlist category.

The real playlist references copyrighted tracks that are not shipped. This writes
gentle, synthesized WAV chords into ``assets/music/`` so the demo has playable,
license-free audio out of the box. Regenerate anytime; the files are disposable.

    python src/make_filler_music.py
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "music"
RATE = 44100

# Note name -> frequency (Hz), a small table covering the chords below.
NOTES = {
    "A2": 110.00, "C3": 130.81, "E3": 164.81, "G3": 196.00, "A3": 220.00,
    "B3": 246.94, "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23,
    "G4": 392.00, "A4": 440.00, "B4": 493.88, "C5": 523.25, "E5": 659.25,
}

# name: (chord notes, seconds, brightness 0..1, tempo pulses/sec)
CLIPS: dict[str, tuple[list[str], float, float, float]] = {
    "filler_calm": (["A2", "E3", "A3", "C4"], 9.0, 0.15, 0.0),
    "filler_warm": (["C3", "G3", "C4", "E4"], 9.0, 0.25, 0.0),
    "filler_happy": (["C4", "E4", "G4", "C5"], 8.0, 0.5, 2.0),
    "filler_energetic": (["A3", "E4", "A4", "C5"], 8.0, 0.75, 4.0),
    "filler_celebration": (["C4", "G4", "C5", "E5"], 8.0, 0.9, 3.0),
}

# playlist.json category for each clip.
CATEGORY = {
    "filler_calm": "calm",
    "filler_warm": "warm",
    "filler_happy": "happy",
    "filler_energetic": "energetic",
    "filler_celebration": "celebration",
}


def _tone(freqs: list[float], seconds: float, brightness: float, tempo: float) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(RATE * seconds), endpoint=False)
    wave_sum = np.zeros_like(t)
    for i, f in enumerate(freqs):
        # Fundamental plus a soft second harmonic scaled by brightness.
        vibrato = 1.0 + 0.002 * np.sin(2 * np.pi * 5.0 * t)
        partial = np.sin(2 * np.pi * f * t * vibrato)
        partial += brightness * 0.35 * np.sin(2 * np.pi * 2 * f * t)
        wave_sum += partial / (i + 1.5)  # upper voices quieter
    # Optional gentle rhythmic pulse (tremolo) for the livelier categories.
    if tempo > 0:
        pulse = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * tempo * t))
        wave_sum *= pulse
    # Fade in/out to avoid clicks.
    fade = int(RATE * 0.4)
    env = np.ones_like(wave_sum)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    wave_sum *= env
    peak = float(np.max(np.abs(wave_sum))) or 1.0
    return (wave_sum / peak) * 0.28  # keep it gentle


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm16.tobytes())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (notes, seconds, brightness, tempo) in CLIPS.items():
        freqs = [NOTES[n] for n in notes]
        samples = _tone(freqs, seconds, brightness, tempo)
        path = OUT_DIR / f"{name}.wav"
        _write_wav(path, samples)
        print(f"wrote {path.relative_to(ROOT)}  ({CATEGORY[name]}, {seconds:.0f}s)")


if __name__ == "__main__":
    main()
