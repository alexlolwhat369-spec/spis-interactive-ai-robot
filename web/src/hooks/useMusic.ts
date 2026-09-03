import { useCallback, useEffect, useRef, useState } from "react";
import { postMusic, type MusicState, type VoiceResult } from "@/lib/api";
import { registerAudio } from "@/lib/audio";

export interface MusicUi {
  available: boolean;
  categories: string[];
  title: string | null;
  category: string | null;
  playing: boolean;
  paused: boolean;
  note: string;
}

export interface MusicApi {
  ui: MusicUi;
  play: (category: string) => Promise<void>;
  resume: () => void;
  pause: () => void;
  next: () => Promise<void>;
  stop: () => Promise<void>;
  /** Fold the /state music snapshot in without fighting local playback. */
  applyServerState: (m: MusicState | undefined) => void;
  /** Pause a playing track for a mic turn (restored by handleVoiceResult). */
  duckForTurn: () => void;
  /** React to music actions a voice turn produced, after the reply plays. */
  handleVoiceResult: (data: VoiceResult) => void;
}

const EMPTY: MusicUi = {
  available: false,
  categories: [],
  title: null,
  category: null,
  playing: false,
  paused: false,
  note: "",
};

export function useMusic(): MusicApi {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const titleRef = useRef<string | null>(null);
  const categoryRef = useRef<string | null>(null);
  const duckedRef = useRef(false);
  const [ui, setUi] = useState<MusicUi>(EMPTY);

  const hasTrack = useCallback(() => {
    const a = audioRef.current;
    return !!a && !!a.currentSrc && !a.ended;
  }, []);

  const sync = useCallback((patch?: Partial<MusicUi>) => {
    const a = audioRef.current;
    const live = !!a && !!a.currentSrc && !a.ended;
    setUi((prev) => ({
      ...prev,
      title: live ? titleRef.current : (patch?.title ?? (live ? prev.title : null)),
      category: categoryRef.current,
      playing: live && !!a && !a.paused,
      paused: live && !!a && a.paused,
      ...patch,
    }));
  }, []);

  useEffect(() => {
    const audio = new Audio();
    audio.preload = "none";
    audioRef.current = audio;
    const unregister = registerAudio(audio, 1);
    const onPlay = () => sync();
    const onPause = () => sync();
    const onEnded = () => {
      titleRef.current = null;
      audio.removeAttribute("src");
      sync();
      void fetch("/music/stop", { method: "POST" }).catch(() => {});
    };
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);
    return () => {
      audio.pause();
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
      unregister();
      audioRef.current = null;
    };
  }, [sync]);

  const startTrack = useCallback(
    (m: MusicState | null) => {
      const a = audioRef.current;
      if (!m || !a) return;
      if (m.ok === false) {
        sync({ note: m.reason || "Could not play that track." });
        return;
      }
      if (!m.url) return;
      titleRef.current = m.title;
      categoryRef.current = m.category;
      duckedRef.current = false;
      a.src = m.url;
      a.play().catch(() => sync({ note: "Browser blocked audio — press play." }));
      sync({ note: "" });
    },
    [sync],
  );

  const play = useCallback(
    async (category: string) => {
      try {
        startTrack(await postMusic("/music/play", { category }));
      } catch {
        sync({ note: "Music request failed." });
      }
    },
    [startTrack, sync],
  );

  const next = useCallback(async () => {
    try {
      startTrack(await postMusic("/music/next"));
    } catch {
      sync({ note: "Music request failed." });
    }
  }, [startTrack, sync]);

  const resume = useCallback(() => {
    audioRef.current?.play().catch(() => {});
  }, []);
  const pause = useCallback(() => {
    audioRef.current?.pause();
  }, []);

  const stop = useCallback(async () => {
    const a = audioRef.current;
    if (a) {
      a.pause();
      a.removeAttribute("src");
      a.load();
    }
    titleRef.current = null;
    categoryRef.current = null;
    sync({ title: null, category: null });
    try {
      await postMusic("/music/stop");
    } catch {
      /* best effort */
    }
  }, [sync]);

  const applyServerState = useCallback(
    (m: MusicState | undefined) => {
      if (!m) return;
      if (!hasTrack()) {
        titleRef.current = null;
        categoryRef.current = m.category;
      }
      setUi((prev) => ({
        ...prev,
        available: m.available,
        categories: m.categories ?? prev.categories,
        category: categoryRef.current ?? m.category,
        note: m.available ? prev.note : "Music unavailable — playlist not configured.",
        title: hasTrack() ? prev.title : m.title,
      }));
    },
    [hasTrack],
  );

  const duckForTurn = useCallback(() => {
    const a = audioRef.current;
    if (a && !a.paused && hasTrack()) {
      duckedRef.current = true;
      a.pause();
    }
  }, [hasTrack]);

  const handleVoiceResult = useCallback(
    (data: VoiceResult) => {
      switch (data.action) {
        case "play_music":
        case "next_music":
          startTrack(data.now_playing);
          break;
        case "pause_music":
          duckedRef.current = false;
          audioRef.current?.pause();
          break;
        case "resume_music":
          duckedRef.current = false;
          audioRef.current?.play().catch(() => {});
          break;
        case "stop_music":
          duckedRef.current = false;
          void stop();
          break;
        default:
          if (duckedRef.current) {
            duckedRef.current = false;
            audioRef.current?.play().catch(() => {});
          }
      }
    },
    [startTrack, stop],
  );

  return { ui, play, resume, pause, next, stop, applyServerState, duckForTurn, handleVoiceResult };
}
