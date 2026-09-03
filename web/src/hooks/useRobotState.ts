import { useEffect, useRef, useState } from "react";
import { getGestures, getState, type GestureInfo, type RobotState } from "@/lib/api";
import type { MusicApi } from "@/hooks/useMusic";

export function useRobotState(music: MusicApi) {
  const [state, setState] = useState<RobotState | null>(null);
  const [gestures, setGestures] = useState<GestureInfo[]>([]);
  const soundMap = useRef<Record<string, string>>({});
  const lastGesture = useRef<string | null>(null);
  const sfxRef = useRef<HTMLAudioElement | null>(null);
  const applyServerState = music.applyServerState;

  useEffect(() => {
    sfxRef.current = new Audio();
    sfxRef.current.preload = "auto";
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
    };
  }, []);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      const s = await getState();
      if (!alive || !s) return;
      const label = s.gesture && s.gesture !== "none" ? s.gesture : null;
      // Edge-trigger a mapped gesture sound once when it first appears.
      if (label && label !== lastGesture.current && soundMap.current[label] && sfxRef.current) {
        sfxRef.current.src = soundMap.current[label];
        sfxRef.current.currentTime = 0;
        void sfxRef.current.play().catch(() => {});
      }
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
  }, [applyServerState]);

  return { state, gestures };
}
