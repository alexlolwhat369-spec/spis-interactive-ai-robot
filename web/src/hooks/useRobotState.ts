import { useCallback, useEffect, useRef, useState } from "react";
import { getGestures, getState, type GestureInfo, type RobotState } from "@/lib/api";
import type { MusicApi } from "@/hooks/useMusic";

export function useRobotState(music: MusicApi) {
  const [state, setState] = useState<RobotState | null>(null);
  const [gestures, setGestures] = useState<GestureInfo[]>([]);
  const [soundError, setSoundError] = useState("");
  const soundMap = useRef<Record<string, string>>({});
  const lastGesture = useRef<string | null>(null);
  const lastGestureEvent = useRef<number | undefined>(undefined);
  const sfxRef = useRef<HTMLAudioElement | null>(null);
  const sfxUrl = useRef<string | null>(null);
  const applyServerState = music.applyServerState;

  const stopGestureSound = useCallback(() => {
    const audio = sfxRef.current;
    sfxRef.current = null;
    sfxUrl.current = null;
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
    audio.volume = 0.8;
    sfxRef.current = audio;
    sfxUrl.current = url;
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
    };
  }, [stopGestureSound]);

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
      setState(s);
      applyServerState(s.music);
    };
    void poll();
    const id = window.setInterval(poll, 500);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [applyServerState, playGestureSound]);

  const previewGestureSound = useCallback((label: string) => {
    const url = soundMap.current[label];
    if (url) playGestureSound(url);
    else setSoundError(`Sound file not installed for ${label.replace(/_/g, " ")}.`);
  }, [playGestureSound]);

  return { state, gestures, stopGestureSound, previewGestureSound, soundError };
}
