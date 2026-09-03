"""Download curated CC0 references and build reaction-sound auditions."""

from __future__ import annotations

import argparse
import html
import json
import os
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame


ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "assets" / "sounds" / "reaction_lab"
SOURCE_DIR = LAB_DIR / "_sources"
SAMPLE_RATE = 44_100
SOURCE_REPOSITORY = "https://github.com/lavenderdotpet/CC0-Public-Domain-Sounds"
RAW_ROOT = "https://raw.githubusercontent.com/lavenderdotpet/CC0-Public-Domain-Sounds/main"

I = "kenney_interfacesounds/Audio"
D = "kenney_digitalaudio/Audio"
C = "80-CC0-creature-SFX"
R = "80-CC0-RPG-SFX"
S = "100-CC0-SFX"
P = "kenney_impactsounds/Audio"

# Each ordered set is three references for A, three for B, and four for C.
REFERENCE_SETS: dict[str, tuple[str, ...]] = {
    "idle": (
        f"{C}/breath.ogg", f"{C}/burble_01.ogg", f"{C}/burble_02.ogg",
        f"{D}/lowRandom.ogg", f"{D}/tone1.ogg", f"{D}/lowThreeTone.ogg",
        f"{I}/select_003.ogg", f"{I}/switch_001.ogg", f"{S}/machine_01.ogg", f"{C}/alien_01.ogg",
    ),
    "listening": (
        f"{I}/question_001.ogg", f"{I}/question_002.ogg", f"{I}/question_003.ogg",
        f"{I}/question_004.ogg", f"{I}/select_001.ogg", f"{I}/select_002.ogg",
        f"{I}/select_003.ogg", f"{I}/open_001.ogg", f"{I}/open_002.ogg", f"{I}/tick_001.ogg",
    ),
    "thinking": (
        f"{I}/scratch_001.ogg", f"{I}/scratch_002.ogg", f"{I}/scratch_003.ogg",
        f"{I}/scratch_004.ogg", f"{I}/scratch_005.ogg", f"{I}/tick_001.ogg",
        f"{I}/tick_002.ogg", f"{I}/tick_004.ogg", f"{S}/machine_01.ogg", f"{D}/lowRandom.ogg",
    ),
    "speaking": (
        f"{C}/alien_01.ogg", f"{C}/alien_02.ogg", f"{C}/alien_03.ogg",
        f"{C}/alien_04.ogg", f"{C}/alien_05.ogg", f"{C}/alien_06.ogg",
        f"{C}/burble_01.ogg", f"{C}/burble_02.ogg", f"{C}/ooh.ogg", f"{C}/cute_02.ogg",
    ),
    "happy": tuple(f"{C}/cute_{number:02}.ogg" for number in range(1, 11)),
    "proud": (
        f"{I}/confirmation_001.ogg", f"{I}/confirmation_002.ogg", f"{I}/confirmation_003.ogg",
        f"{I}/confirmation_004.ogg", f"{D}/powerUp1.ogg", f"{D}/powerUp2.ogg",
        f"{D}/powerUp3.ogg", f"{D}/powerUp4.ogg", f"{D}/powerUp5.ogg", f"{D}/threeTone1.ogg",
    ),
    "confused": (
        f"{I}/question_001.ogg", f"{I}/question_002.ogg", f"{I}/question_003.ogg",
        f"{I}/question_004.ogg", f"{C}/weird_01.ogg", f"{C}/weird_02.ogg",
        f"{C}/weird_03.ogg", f"{C}/weird_04.ogg", f"{C}/weird_05.ogg", f"{C}/ooh.ogg",
    ),
    "heart": (
        f"{P}/impactSoft_heavy_000.ogg", f"{P}/impactSoft_heavy_001.ogg", f"{P}/impactSoft_medium_000.ogg",
        f"{C}/cute_01.ogg", f"{C}/cute_03.ogg", f"{C}/cute_05.ogg",
        f"{C}/ooh.ogg", f"{R}/item_gem_01.ogg", f"{R}/item_gem_03.ogg", f"{R}/spell_01.ogg",
    ),
    "annoyed": (
        f"{I}/error_001.ogg", f"{I}/error_002.ogg", f"{I}/error_003.ogg",
        f"{I}/error_004.ogg", f"{I}/error_005.ogg", f"{C}/grunt_01.ogg",
        f"{C}/grunt_02.ogg", f"{C}/grunt_03.ogg", f"{C}/grunt_04.ogg", f"{C}/grunt_05.ogg",
    ),
    "curious": (
        f"{I}/question_001.ogg", f"{I}/question_002.ogg", f"{I}/question_003.ogg",
        f"{I}/question_004.ogg", f"{C}/cute_04.ogg", f"{C}/cute_06.ogg",
        f"{C}/cute_08.ogg", f"{R}/spell_02.ogg", f"{R}/item_gem_02.ogg", f"{S}/spring_01.ogg",
    ),
    "ok": (
        f"{I}/confirmation_001.ogg", f"{I}/confirmation_002.ogg", f"{I}/confirmation_003.ogg",
        f"{I}/confirmation_004.ogg", f"{I}/select_001.ogg", f"{I}/select_002.ogg",
        f"{I}/select_003.ogg", f"{I}/select_004.ogg", f"{I}/select_005.ogg", f"{I}/select_006.ogg",
    ),
}

VARIANT_TITLES = {
    "idle": ("Soft breath", "Quiet computer", "Gentle motion"),
    "listening": ("Attentive question", "Soft prompt", "Friendly cue"),
    "thinking": ("Pondering", "Working it out", "Tiny processor"),
    "speaking": ("Robot syllables", "Animated chatter", "Friendly mumble"),
    "happy": ("Bright delight", "Warm giggle", "Playful joy"),
    "proud": ("Achievement", "Confident rise", "Small victory"),
    "confused": ("Unsure question", "Puzzled mumble", "What was that?"),
    "heart": ("Heartbeat", "Affectionate chirp", "Love sparkle"),
    "annoyed": ("Firm warning", "Short protest", "Grumpy robot"),
    "curious": ("Interested question", "Surprised interest", "Discovery"),
    "ok": ("Clear confirmation", "Gentle approval", "Quick accepted"),
}

DURATIONS = {
    "idle": 1.20, "listening": 0.95, "thinking": 1.10, "speaking": 1.05,
    "happy": 1.05, "proud": 1.15, "confused": 1.05, "heart": 1.30,
    "annoyed": 0.90, "curious": 1.10, "ok": 0.80,
}

PITCH_BIAS = {
    "idle": 0.94, "listening": 1.04, "thinking": 0.92, "speaking": 1.02,
    "happy": 1.10, "proud": 1.03, "confused": 0.94, "heart": 1.04,
    "annoyed": 0.84, "curious": 1.10, "ok": 1.07,
}


def validate_catalog() -> None:
    expected = {
        "idle", "listening", "thinking", "speaking", "happy", "proud",
        "confused", "heart", "annoyed", "curious", "ok",
    }
    if set(REFERENCE_SETS) != expected:
        raise ValueError("Reaction catalog does not match the robot states excluding Mohan.")
    for reaction, references in REFERENCE_SETS.items():
        if len(references) != 10:
            raise ValueError(f"{reaction} must have exactly ten references.")
        if len(set(references)) != 10:
            raise ValueError(f"{reaction} contains duplicate references.")


def source_path(reference: str) -> Path:
    return SOURCE_DIR / reference.replace("/", "__")


def source_url(reference: str) -> str:
    encoded = "/".join(urllib.parse.quote(part) for part in reference.split("/"))
    return f"{RAW_ROOT}/{encoded}"


def download_sources() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    references = sorted({item for values in REFERENCE_SETS.values() for item in values})
    for reference in references:
        destination = source_path(reference)
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        partial = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {reference}")
        urllib.request.urlretrieve(source_url(reference), partial)
        partial.replace(destination)


def load_audio(path: Path) -> np.ndarray:
    sound = pygame.mixer.Sound(str(path))
    audio = pygame.sndarray.array(sound).astype(np.float32) / 32768.0
    if audio.ndim == 1:
        audio = np.column_stack((audio, audio))
    return audio


def pitch(audio: np.ndarray, ratio: float) -> np.ndarray:
    output_frames = max(1, int(len(audio) / ratio))
    positions = np.linspace(0, len(audio) - 1, output_frames)
    source = np.arange(len(audio))
    return np.column_stack(
        [np.interp(positions, source, audio[:, channel]) for channel in range(2)]
    ).astype(np.float32)


def fade(audio: np.ndarray, fade_in_ms: int = 7, fade_out_ms: int = 90) -> np.ndarray:
    result = audio.copy()
    fade_in_frames = min(len(result), int(SAMPLE_RATE * fade_in_ms / 1000))
    fade_out_frames = min(len(result), int(SAMPLE_RATE * fade_out_ms / 1000))
    if fade_in_frames:
        result[:fade_in_frames] *= np.linspace(0.0, 1.0, fade_in_frames)[:, None]
    if fade_out_frames:
        result[-fade_out_frames:] *= np.linspace(1.0, 0.0, fade_out_frames)[:, None]
    return result


def pan(audio: np.ndarray, position: float) -> np.ndarray:
    angle = (np.clip(position, -1.0, 1.0) + 1.0) * np.pi / 4.0
    result = audio.copy()
    result[:, 0] *= np.cos(angle)
    result[:, 1] *= np.sin(angle)
    return result


def overlay(canvas: np.ndarray, audio: np.ndarray, start: float, gain: float) -> None:
    first = int(start * SAMPLE_RATE)
    last = min(len(canvas), first + len(audio))
    if last > first:
        canvas[first:last] += audio[: last - first] * gain


def add_echo(audio: np.ndarray, delay_ms: int, decay: float) -> np.ndarray:
    dry = audio.copy()
    result = audio.copy()
    delay = int(SAMPLE_RATE * delay_ms / 1000)
    for repeat in (1, 2):
        offset = delay * repeat
        if offset < len(result):
            result[offset:] += dry[:-offset] * (decay**repeat)
    return result


def finish(audio: np.ndarray) -> np.ndarray:
    audio = fade(audio, fade_in_ms=5, fade_out_ms=130)
    audio = np.tanh(audio * 1.10)
    rms = float(np.sqrt(np.mean(audio * audio)))
    if rms:
        audio *= min(1.0, 0.10 / rms)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.88:
        audio *= 0.88 / peak
    return audio


def mix_group(samples: list[np.ndarray], duration: float, pitch_bias: float, variant: int) -> np.ndarray:
    canvas = np.zeros((int(duration * SAMPLE_RATE), 2), dtype=np.float32)
    starts = ((0.00, 0.17, 0.35), (0.00, 0.21, 0.43), (0.00, 0.13, 0.32, 0.52))[variant]
    gains = ((0.62, 0.40, 0.25), (0.58, 0.39, 0.27), (0.50, 0.36, 0.27, 0.20))[variant]
    ratios = ((1.00, 1.08, 0.96), (0.94, 1.03, 1.10), (1.12, 0.93, 1.05, 0.99))[variant]
    pans = ((-0.10, 0.15, -0.25), (0.12, -0.18, 0.25), (-0.22, 0.22, -0.08, 0.12))[variant]
    for audio, start, gain, ratio, position in zip(samples, starts, gains, ratios, pans):
        layer = pan(fade(pitch(audio, ratio * pitch_bias)), position)
        overlay(canvas, layer, start, gain)
    return finish(add_echo(canvas, (115, 145, 95)[variant], (0.15, 0.20, 0.12)[variant]))


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def build_reaction(reaction: str) -> list[Path]:
    loaded = [load_audio(source_path(reference)) for reference in REFERENCE_SETS[reaction]]
    groups = (loaded[:3], loaded[3:6], loaded[6:])
    output_dir = LAB_DIR / reaction / "variants"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, group in enumerate(groups):
        letter = "ABC"[index]
        slug = VARIANT_TITLES[reaction][index].lower().replace(" ", "_").replace("?", "")
        path = output_dir / f"{reaction}_{letter}_{slug}.wav"
        write_wav(path, mix_group(group, DURATIONS[reaction], PITCH_BIAS[reaction], index))
        outputs.append(path)
    return outputs


def write_manifest(outputs: dict[str, list[Path]]) -> None:
    payload: dict[str, object] = {
        "status": "audition_only",
        "source_repository": SOURCE_REPOSITORY,
        "license": "CC0-1.0",
        "mohan": "excluded_existing_sound_preserved",
        "reactions": {},
    }
    reactions = payload["reactions"]
    assert isinstance(reactions, dict)
    for reaction, references in REFERENCE_SETS.items():
        reactions[reaction] = {
            "references": [{"path": item, "url": source_url(item)} for item in references],
            "variants": [
                {"label": letter, "title": title, "file": str(path.relative_to(LAB_DIR)).replace("\\", "/")}
                for letter, title, path in zip("ABC", VARIANT_TITLES[reaction], outputs[reaction])
            ],
        }
    (LAB_DIR / "catalog.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_soundboard(outputs: dict[str, list[Path]]) -> None:
    sections = []
    for reaction, paths in outputs.items():
        candidates = []
        for letter, title, path in zip("ABC", VARIANT_TITLES[reaction], paths):
            relative = path.relative_to(LAB_DIR).as_posix()
            candidates.append(
                f'<label class="candidate"><span class="choice"><input type="radio" name="{reaction}" value="{letter}">'
                f'<strong>{letter}</strong><span>{html.escape(title)}</span></span>'
                f'<audio controls preload="metadata" src="{html.escape(relative)}"></audio></label>'
            )
        sections.append(
            f'<section><div class="reaction-head"><h2>{reaction}</h2><span>10 references / 3 candidates</span></div>'
            f'<div class="candidate-grid">{"".join(candidates)}</div></section>'
        )
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPIS Reaction Sound Lab</title><style>
:root{{--bg:#111416;--panel:#191e21;--line:#344047;--text:#f5f7f8;--muted:#a9b4b9;--cyan:#5ee6e6;--yellow:#ffd166}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 Segoe UI,Arial,sans-serif;letter-spacing:0}}
header{{padding:28px max(20px,5vw);border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:20px}}
h1{{margin:0;font-size:clamp(24px,4vw,38px)}} .status{{color:var(--cyan);font-weight:700}} main{{max-width:1180px;margin:auto;padding:0 24px 48px}}
section{{padding:26px 0 30px;border-bottom:1px solid var(--line)}} .reaction-head{{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:14px}}
h2{{margin:0;text-transform:capitalize;font-size:21px}} .reaction-head span{{color:var(--muted);font-size:13px}} .candidate-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
.candidate{{display:grid;gap:14px;padding:15px;background:var(--panel);border:1px solid var(--line);border-radius:6px;cursor:pointer;min-width:0}}
.candidate:has(input:checked){{border-color:var(--cyan);box-shadow:inset 0 0 0 1px var(--cyan)}} .choice{{display:flex;align-items:center;gap:9px;min-width:0}}
.choice strong{{color:var(--yellow)}} .choice span{{overflow-wrap:anywhere}} input{{accent-color:var(--cyan)}} audio{{width:100%;height:36px}}
footer{{position:sticky;bottom:0;background:#111416ee;border-top:1px solid var(--line);padding:14px max(20px,5vw);display:flex;justify-content:flex-end;gap:10px}}
button{{border:1px solid var(--line);background:var(--panel);color:var(--text);padding:10px 15px;border-radius:5px;font-weight:700;cursor:pointer}} button.primary{{background:var(--cyan);color:#081112;border-color:var(--cyan)}}
@media(max-width:760px){{header{{align-items:flex-start;flex-direction:column}}.candidate-grid{{grid-template-columns:1fr}}.reaction-head{{align-items:flex-start;flex-direction:column}}}}
</style></head><body><header><h1>SPIS Reaction Sound Lab</h1><div class="status" id="status">0 / {len(outputs)} selected</div></header>
<main>{''.join(sections)}</main><footer><button id="clear">Clear</button><button class="primary" id="download">Download selections</button></footer>
<script>
const reactions={json.dumps(list(outputs))}; const key='spis-sound-selections';
function values(){{return Object.fromEntries(reactions.map(r=>[r,document.querySelector(`input[name="${{r}}"]:checked`)?.value||null]))}}
function refresh(){{const v=values();document.getElementById('status').textContent=`${{Object.values(v).filter(Boolean).length}} / ${{reactions.length}} selected`;localStorage.setItem(key,JSON.stringify(v))}}
document.querySelectorAll('input').forEach(i=>i.addEventListener('change',refresh));
try{{const saved=JSON.parse(localStorage.getItem(key)||'{{}}');Object.entries(saved).forEach(([r,v])=>{{const i=document.querySelector(`input[name="${{r}}"][value="${{v}}"]`);if(i)i.checked=true}})}}catch{{}}
document.getElementById('clear').onclick=()=>{{document.querySelectorAll('input').forEach(i=>i.checked=false);refresh()}};
document.getElementById('download').onclick=()=>{{const blob=new Blob([JSON.stringify(values(),null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='reaction-sound-selections.json';a.click();URL.revokeObjectURL(a.href)}};
refresh();
</script></body></html>'''
    (LAB_DIR / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full reaction sound audition lab.")
    parser.add_argument("--download", action="store_true", help="Download missing CC0 references.")
    parser.add_argument("--reaction", choices=tuple(REFERENCE_SETS), help="Build only one reaction.")
    args = parser.parse_args()
    validate_catalog()
    if args.download:
        download_sources()
    missing = [item for values in REFERENCE_SETS.values() for item in values if not source_path(item).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(set(missing))} source files are missing; run with --download.")
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
    reactions = (args.reaction,) if args.reaction else tuple(REFERENCE_SETS)
    outputs = {reaction: build_reaction(reaction) for reaction in reactions}
    if not args.reaction:
        write_manifest(outputs)
        write_soundboard(outputs)
    for paths in outputs.values():
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
