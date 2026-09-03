"""Generate original robot sound effects using lightweight audio synthesis."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44_100
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "assets" / "sounds" / "candidates"
RNG = np.random.default_rng(20260902)


def canvas(seconds: float) -> np.ndarray:
    return np.zeros((int(seconds * SAMPLE_RATE), 2), dtype=np.float64)


def envelope(length: int, attack: float = 0.02, release: float = 0.12) -> np.ndarray:
    env = np.ones(length, dtype=np.float64)
    attack_n = min(length, max(1, int(attack * SAMPLE_RATE)))
    release_n = min(length, max(1, int(release * SAMPLE_RATE)))
    env[:attack_n] = np.sin(np.linspace(0.0, math.pi / 2.0, attack_n)) ** 2
    env[-release_n:] *= np.cos(np.linspace(0.0, math.pi / 2.0, release_n)) ** 2
    return env


def oscillator(
    seconds: float,
    frequency: float,
    end_frequency: float | None = None,
    *,
    waveform: str = "sine",
    vibrato_hz: float = 0.0,
    vibrato_depth: float = 0.0,
) -> np.ndarray:
    length = int(seconds * SAMPLE_RATE)
    t = np.arange(length, dtype=np.float64) / SAMPLE_RATE
    end_frequency = frequency if end_frequency is None else end_frequency
    frequencies = np.linspace(frequency, end_frequency, length)
    if vibrato_hz:
        frequencies *= 1.0 + vibrato_depth * np.sin(2.0 * math.pi * vibrato_hz * t)
    phase = 2.0 * math.pi * np.cumsum(frequencies) / SAMPLE_RATE
    if waveform == "triangle":
        return (2.0 / math.pi) * np.arcsin(np.sin(phase))
    if waveform == "soft_square":
        return np.tanh(2.2 * np.sin(phase))
    if waveform == "sparkle":
        return np.sin(phase) + 0.24 * np.sin(2.01 * phase) + 0.12 * np.sin(3.02 * phase)
    return np.sin(phase)


def add_tone(
    target: np.ndarray,
    start: float,
    seconds: float,
    frequency: float,
    end_frequency: float | None = None,
    *,
    amplitude: float = 0.3,
    waveform: str = "sine",
    pan: float = 0.0,
    attack: float = 0.02,
    release: float = 0.12,
    vibrato_hz: float = 0.0,
    vibrato_depth: float = 0.0,
) -> None:
    begin = int(start * SAMPLE_RATE)
    available = max(0, len(target) - begin)
    length = min(int(seconds * SAMPLE_RATE), available)
    if length <= 0:
        return
    mono = oscillator(
        length / SAMPLE_RATE,
        frequency,
        end_frequency,
        waveform=waveform,
        vibrato_hz=vibrato_hz,
        vibrato_depth=vibrato_depth,
    )[:length]
    mono *= envelope(length, attack, release) * amplitude
    angle = (max(-1.0, min(1.0, pan)) + 1.0) * math.pi / 4.0
    target[begin : begin + length, 0] += mono * math.cos(angle)
    target[begin : begin + length, 1] += mono * math.sin(angle)


def add_noise(
    target: np.ndarray,
    start: float,
    seconds: float,
    *,
    amplitude: float = 0.08,
    smooth: int = 18,
    pan: float = 0.0,
) -> None:
    begin = int(start * SAMPLE_RATE)
    length = min(int(seconds * SAMPLE_RATE), max(0, len(target) - begin))
    if length <= 0:
        return
    noise = RNG.normal(0.0, 1.0, length + smooth - 1)
    kernel = np.ones(smooth) / smooth
    mono = np.convolve(noise, kernel, mode="valid")
    mono *= envelope(length, 0.01, max(0.05, seconds * 0.65)) * amplitude
    angle = (max(-1.0, min(1.0, pan)) + 1.0) * math.pi / 4.0
    target[begin : begin + length, 0] += mono * math.cos(angle)
    target[begin : begin + length, 1] += mono * math.sin(angle)


def add_reverb(audio: np.ndarray, taps: tuple[tuple[float, float], ...]) -> np.ndarray:
    wet = audio.copy()
    for delay_seconds, gain in taps:
        delay = int(delay_seconds * SAMPLE_RATE)
        if delay < len(audio):
            wet[delay:] += audio[:-delay] * gain
    return wet


def finish(audio: np.ndarray, *, reverb: bool = True) -> np.ndarray:
    if reverb:
        audio = add_reverb(audio, ((0.075, 0.18), (0.145, 0.10), (0.225, 0.05)))
    audio = np.tanh(audio * 1.15)
    peak = float(np.max(np.abs(audio)))
    return audio * (0.88 / peak) if peak else audio


def ok_confirm() -> np.ndarray:
    audio = canvas(0.85)
    for start, note, pan in ((0.00, 659.25, -0.25), (0.16, 783.99, 0.20), (0.32, 1046.50, 0.0)):
        add_tone(audio, start, 0.42, note, amplitude=0.27, waveform="sparkle", pan=pan, release=0.25)
    return finish(audio)


def angry_alert() -> np.ndarray:
    audio = canvas(1.0)
    add_tone(audio, 0.00, 0.55, 185.0, 112.0, amplitude=0.34, waveform="soft_square", pan=-0.12)
    add_tone(audio, 0.08, 0.58, 247.0, 139.0, amplitude=0.25, waveform="triangle", pan=0.12)
    for start in (0.05, 0.23, 0.41):
        add_noise(audio, start, 0.12, amplitude=0.11, smooth=5, pan=(-0.25 if start == 0.23 else 0.25))
    add_tone(audio, 0.62, 0.28, 123.5, amplitude=0.30, waveform="soft_square", release=0.18)
    return finish(audio, reverb=False)


def heart_magic() -> np.ndarray:
    audio = canvas(1.55)
    notes = (440.0, 554.37, 659.25, 880.0)
    for index, note in enumerate(notes):
        add_tone(
            audio,
            index * 0.17,
            0.72,
            note,
            amplitude=0.22,
            waveform="sparkle",
            pan=(-0.35 + index * 0.23),
            release=0.48,
        )
    add_tone(audio, 0.72, 0.65, 1320.0, 1760.0, amplitude=0.09, waveform="sine", release=0.40)
    return finish(audio)


def proud_success() -> np.ndarray:
    audio = canvas(1.65)
    for start, notes in (
        (0.00, (392.0, 493.88)),
        (0.28, (493.88, 587.33)),
        (0.56, (587.33, 783.99)),
    ):
        for index, note in enumerate(notes):
            add_tone(audio, start, 0.45, note, amplitude=0.18, waveform="triangle", pan=(-0.2 + index * 0.4))
    for note, pan in ((523.25, -0.35), (659.25, 0.0), (783.99, 0.35)):
        add_tone(audio, 0.86, 0.70, note, amplitude=0.20, waveform="sparkle", pan=pan, release=0.42)
    return finish(audio)


def curious_question() -> np.ndarray:
    audio = canvas(1.05)
    add_tone(audio, 0.00, 0.34, 392.0, 440.0, amplitude=0.24, waveform="sine", pan=-0.20)
    add_tone(audio, 0.30, 0.58, 493.88, 659.25, amplitude=0.29, waveform="sine", pan=0.20, vibrato_hz=6.0, vibrato_depth=0.008)
    add_tone(audio, 0.58, 0.35, 987.77, amplitude=0.10, waveform="sparkle", release=0.24)
    return finish(audio)


def game_start() -> np.ndarray:
    audio = canvas(1.55)
    for start, note, pan in ((0.00, 261.63, -0.30), (0.25, 329.63, 0.30), (0.50, 392.0, -0.12)):
        add_tone(audio, start, 0.24, note, amplitude=0.26, waveform="soft_square", pan=pan, release=0.11)
    add_tone(audio, 0.72, 0.62, 440.0, 1174.66, amplitude=0.25, waveform="sparkle", release=0.25)
    add_noise(audio, 0.70, 0.50, amplitude=0.08, smooth=40)
    return finish(audio)


def incorrect_soft() -> np.ndarray:
    audio = canvas(0.95)
    for start, note, pan in ((0.00, 659.25, -0.18), (0.20, 523.25, 0.18), (0.40, 392.0, 0.0)):
        add_tone(audio, start, 0.40, note, amplitude=0.20, waveform="triangle", pan=pan, release=0.24)
    return finish(audio)


def mohan_special() -> np.ndarray:
    audio = canvas(1.55)
    melody = ((0.00, 783.99), (0.20, 987.77), (0.40, 1174.66), (0.66, 1567.98))
    for index, (start, note) in enumerate(melody):
        add_tone(
            audio,
            start,
            0.55,
            note,
            note * 1.035,
            amplitude=0.25,
            waveform="sine",
            pan=(-0.25 if index % 2 == 0 else 0.25),
            release=0.28,
            vibrato_hz=7.2,
            vibrato_depth=0.012,
        )
    add_tone(audio, 0.90, 0.42, 3135.96, 2350.0, amplitude=0.07, waveform="sparkle", release=0.30)
    return finish(audio)


def robot_transition() -> np.ndarray:
    audio = canvas(0.78)
    add_noise(audio, 0.00, 0.58, amplitude=0.14, smooth=60, pan=-0.20)
    add_tone(audio, 0.00, 0.60, 210.0, 1380.0, amplitude=0.20, waveform="sine", pan=0.15, release=0.20)
    add_tone(audio, 0.30, 0.32, 1760.0, 880.0, amplitude=0.09, waveform="sparkle", pan=-0.15, release=0.20)
    return finish(audio)


EFFECTS = {
    "01_ok_confirm.wav": ok_confirm,
    "02_angry_alert.wav": angry_alert,
    "03_heart_magic.wav": heart_magic,
    "04_proud_success.wav": proud_success,
    "05_curious_question.wav": curious_question,
    "06_game_start.wav": game_start,
    "07_incorrect_soft.wav": incorrect_soft,
    "08_mohan_special.wav": mohan_special,
    "09_robot_transition.wav": robot_transition,
}


def write_wav(path: Path, audio: np.ndarray) -> None:
    samples = np.asarray(np.clip(audio, -1.0, 1.0) * 32767.0, dtype="<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(samples.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate original robot sound-effect candidates.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = []
    for filename, builder in EFFECTS.items():
        audio = builder()
        path = args.output / filename
        write_wav(path, audio)
        manifest.append(
            {
                "file": filename,
                "seconds": round(len(audio) / SAMPLE_RATE, 2),
                "sample_rate": SAMPLE_RATE,
                "channels": 2,
            }
        )
        print(f"Generated {path} ({manifest[-1]['seconds']:.2f}s)")

    (args.output / "manifest.json").write_text(
        json.dumps({"effects": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
