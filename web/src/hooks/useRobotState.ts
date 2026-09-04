import { useCallback, useEffect, useRef, useState } from "react";
import { getGestures, getState, type GestureInfo, type RobotState } from "@/lib/api";
import { registerAudio } from "@/lib/audio";
import type { MusicApi } from "@/hooks/useMusic";

export function useRobotState(music: MusicApi) {
  const [state, setState] = useState<RobotState | null>(null);
  const [gestures, setGestures] = useState<GestureInfo[]>([]);
  const [soundError, setSoundError] = useState("");
  const soundMap = useRef<Record<string, string>>({});
  const lastGesture = useRef<string | null>(null);
  const lastGestureEvent = useRef<number | undefined>(undefined);
  const lastReaction = useRef<string | null>(null);
  const sfxRef = useRef<HTMLAudioElement | null>(null);
  const sfxUrl = useRef<string | null>(null);
  const sfxCleanup = useRef<(() => void) | null>(null);
  const reactionRef = useRef<HTMLAudioElement | null>(null);
  const reactionCleanup = useRef<(() => void) | null>(null);
  const applyServerState = music.applyServerState;

  const stopGestureSound = useCallback(() => {
    const audio = sfxRef.current;
    sfxRef.current = null;
    sfxUrl.current = null;
    sfxCleanup.current?.();
    sfxCleanup.current = null;
    if (audio) {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    }
  }, []);

  const playGestureSound = useCallback((url: string) => {
    const active = sfxRef.current;
    if (active && !active.paused && !active.ended && sfxUrl.current === url) return;
    stopGestureSound();
    setSoundError("");
    const audio = new Audio(url);
    audio.preload = "auto";
    sfxRef.current = audio;
    sfxUrl.current = url;
    sfxCleanup.current = registerAudio(audio, 0.8);
    const retire = () => {
      if (sfxRef.current === audio) stopGestureSound();
    };
    audio.addEventListener("ended", retire, { once: true });
    audio.addEventListener("error", () => {
      if (sfxRef.current !== audio) return;
      setSoundError("Gesture sound could not be loaded.");
      retire();
    }, { once: true });
    void audio.play().catch((error: unknown) => {
      if (sfxRef.current !== audio) return;
      setSoundError(error instanceof DOMException && error.name === "NotAllowedError"
        ? "Gesture audio blocked by browser."
        : "Gesture sound playback failed.");
      retire();
    });
  }, [stopGestureSound]);

  const stopReactionSound = useCallback(() => {
    const audio = reactionRef.current;
    reactionRef.current = null;
    reactionCleanup.current?.();
    reactionCleanup.current = null;
    if (audio) {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    }
  }, []);

  const playReactionSound = useCallback((reaction: string) => {
    stopReactionSound();
    // A random variant served by the backend; 404s (reaction with no files)
    // resolve through the error handler below and simply play nothing.
    const audio = new Audio(`/sound/reactions/${reaction}`);
    audio.preload = "auto";
    reactionRef.current = audio;
    reactionCleanup.current = registerAudio(audio, 0.7);
    const retire = () => {
      if (reactionRef.current === audio) stopReactionSound();
    };
    audio.addEventListener("ended", retire, { once: true });
    audio.addEventListener("error", retire, { once: true });
    void audio.play().catch(retire);
  }, [stopReactionSound]);

  useEffect(() => {
    let cancelled = false;
    getGestures().then((list) => {
      if (cancelled) return;
      setGestures(list);
      const map: Record<string, string> = {};
      for (const g of list) if (g.sound) map[g.label] = g.sound;
      soundMap.current = map;
    });
    return () => {
      cancelled = true;
      stopGestureSound();
      stopReactionSound();
    };
  }, [stopGestureSound, stopReactionSound]);

  useEffect(() => {
    let alive = true;
    let pending = false;
    const poll = async () => {
      if (pending) return;
      pending = true;
      const s = await getState().finally(() => { pending = false; });
      if (!alive || !s) return;
      const label = s.gesture && s.gesture !== "none" ? s.gesture : null;
      const hasEvent = Number.isInteger(s.gesture_event);
      const isNew = hasEvent
        ? s.gesture_event !== lastGestureEvent.current
        : label !== lastGesture.current;
      if (label && isNew && soundMap.current[label]) {
        playGestureSound(soundMap.current[label]);
      }
      if (hasEvent) lastGestureEvent.current = s.gesture_event;
      lastGesture.current = label;

      // Emotion cue: play a reaction sound when the reaction changes, skipping
      // idle (ambient) and any moment a gesture effect is already sounding so
      // the two channels never double up.
      const reactionChanged = s.reaction !== lastReaction.current;
      const gestureActive = !!sfxRef.current && !sfxRef.current.ended;
      if (reactionChanged && s.reaction && s.reaction !== "idle" && !gestureActive) {
        playReactionSound(s.reaction);
      }
      lastReaction.current = s.reaction;

      setState(s);
      applyServerState(s.music);
    };
    void poll();
    const id = window.setInterval(poll, 500);
    return () => {
      alive = false;
      window.clearInterval(id);
      stopReactionSound();
    };
  }, [applyServerState, playGestureSound, playReactionSound, stopReactionSound]);

  const previewGestureSound = useCallback((label: string) => {
    const url = soundMap.current[label];
    if (url) playGestureSound(url);
    else setSoundError(`Sound file not installed for ${label.replace(/_/g, " ")}.`);
  }, [playGestureSound]);

  return { state, gestures, stopGestureSound, previewGestureSound, soundError };
}
