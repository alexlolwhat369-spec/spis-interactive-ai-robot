// Global audio settings shared by every <audio> element the app creates —
// music, gesture effects, reaction effects, and voice replies. A single volume
// + mute governs them all, while each channel keeps a base gain so the relative
// mix (e.g. gesture effects quieter than music) is preserved.
import { useSyncExternalStore } from "react";

const STORAGE_KEY = "spis.audio";
const clamp = (v: number) => Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));

let volume = 1;
let muted = false;

// Live elements → their base gain, so a settings change re-applies to each.
const elements = new Map<HTMLAudioElement, number>();
const listeners = new Set<() => void>();
let snapshot = { volume, muted };

(function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw) as { volume?: number; muted?: boolean };
    if (typeof saved.volume === "number") volume = clamp(saved.volume);
    if (typeof saved.muted === "boolean") muted = saved.muted;
    snapshot = { volume, muted };
  } catch {
    /* corrupt or unavailable storage — fall back to defaults */
  }
})();

const effective = () => (muted ? 0 : volume);

function apply() {
  const v = effective();
  for (const [audio, base] of elements) audio.volume = v * base;
}

function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ volume, muted }));
  } catch {
    /* best effort */
  }
}

function emit() {
  snapshot = { volume, muted };
  for (const l of listeners) l();
}

/** Register a live <audio> so global volume/mute drive it; returns an unregister fn. */
export function registerAudio(audio: HTMLAudioElement, base = 1): () => void {
  elements.set(audio, base);
  audio.volume = effective() * base;
  return () => {
    elements.delete(audio);
  };
}

export function setVolume(v: number): void {
  volume = clamp(v);
  apply();
  persist();
  emit();
}

export function setMuted(m: boolean): void {
  muted = m;
  apply();
  persist();
  emit();
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

function getSnapshot() {
  return snapshot;
}

/** Reactive view of the audio settings for UI controls. */
export function useAudioSettings() {
  const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  return { volume: state.volume, muted: state.muted, setVolume, setMuted };
}
