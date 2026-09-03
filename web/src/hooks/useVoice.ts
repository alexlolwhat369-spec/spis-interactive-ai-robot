import { useCallback, useEffect, useRef, useState } from "react";
import { postVoice, voiceAvailable, voiceDone, voiceListening } from "@/lib/api";
import type { MusicApi } from "@/hooks/useMusic";

const TARGET_RATE = 16000;
const clampPct = (v: number) => Math.max(0, Math.min(1, Number(v) || 0));

export type VoiceStatus = "idle" | "recording" | "thinking";

export interface MicDevice {
  id: string;
  label: string;
}

function floatTo16kPCM(buffer: Float32Array, inputRate: number): Int16Array {
  const ratio = inputRate / TARGET_RATE;
  const outLength = Math.floor(buffer.length / ratio);
  const out = new Int16Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const s = (buffer[idx] || 0) * (1 - frac) + (buffer[idx + 1] || 0) * frac;
    const clamped = Math.max(-1, Math.min(1, s));
    out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }
  return out;
}

function playReply(b64: string): Promise<void> {
  return new Promise((resolve) => {
    if (!b64) return resolve();
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
    const audio = new Audio(url);
    audio.onended = () => {
      URL.revokeObjectURL(url);
      resolve();
    };
    audio.onerror = () => {
      URL.revokeObjectURL(url);
      resolve();
    };
    audio.play().catch(() => resolve());
  });
}

export function useVoice(music: MusicApi, stopGestureSound: () => void) {
  const [available, setAvailable] = useState(false);
  const [status, setStatus] = useState<VoiceStatus>("idle");
  const [heard, setHeard] = useState("");
  const [reply, setReply] = useState("");
  const [note, setNote] = useState("");
  const [inputLevel, setInputLevel] = useState(0);
  const [mics, setMics] = useState<MicDevice[]>([]);
  const [selectedMic, setSelectedMic] = useState("");

  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const peakRef = useRef(0);
  const recordingRef = useRef(false);
  const busyRef = useRef(false);

  const refreshMics = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    setMics(
      devices
        .filter((d) => d.kind === "audioinput")
        .map((d, i) => ({ id: d.deviceId, label: d.label || `Microphone ${i + 1}` })),
    );
  }, []);

  useEffect(() => {
    voiceAvailable().then((ok) => {
      setAvailable(ok);
      if (ok) void refreshMics();
    });
  }, [refreshMics]);

  const teardown = useCallback(() => {
    processorRef.current?.disconnect();
    if (processorRef.current) processorRef.current.onaudioprocess = null;
    sourceRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    void ctxRef.current?.close();
    processorRef.current = null;
    sourceRef.current = null;
    streamRef.current = null;
    ctxRef.current = null;
  }, []);

  const start = useCallback(async () => {
    if (recordingRef.current || busyRef.current) return;
    stopGestureSound();
    recordingRef.current = true;
    chunksRef.current = [];
    peakRef.current = 0;
    setInputLevel(0);
    setStatus("recording");
    setNote("");
    music.duckForTurn();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: selectedMic ? { deviceId: { exact: selectedMic } } : true,
      });
      streamRef.current = stream;
      await refreshMics();
      const ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      ctxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      processor.onaudioprocess = (e) => {
        const samples = new Float32Array(e.inputBuffer.getChannelData(0));
        chunksRef.current.push(samples);
        let peak = 0;
        for (const s of samples) peak = Math.max(peak, Math.abs(s));
        peakRef.current = Math.max(peakRef.current, peak);
        setInputLevel(clampPct(peak * 3));
      };
      source.connect(processor);
      processor.connect(ctx.destination);
      voiceListening();
    } catch {
      recordingRef.current = false;
      setStatus("idle");
      setNote("Microphone blocked. Allow mic access and retry.");
    }
  }, [music, refreshMics, selectedMic, stopGestureSound]);

  const stop = useCallback(async () => {
    if (!recordingRef.current) return;
    recordingRef.current = false;
    const inputRate = ctxRef.current?.sampleRate ?? 48000;
    const total = chunksRef.current.reduce((n, c) => n + c.length, 0);
    const merged = new Float32Array(total);
    let offset = 0;
    for (const c of chunksRef.current) {
      merged.set(c, offset);
      offset += c.length;
    }
    teardown();
    setInputLevel(0);

    if (total === 0) {
      setStatus("idle");
      return;
    }
    const pcm = floatTo16kPCM(merged, inputRate);
    busyRef.current = true;
    setStatus("thinking");
    try {
      const data = await postVoice(pcm.buffer as ArrayBuffer);
      setHeard(data.heard ? `You: ${data.heard}` : "(didn't catch that)");
      setReply(data.reply || "");
      if (peakRef.current < 0.01) {
        setNote("Microphone input was very low. Pick another mic or speak closer.");
      } else {
        setNote(data.provider_error ? "Ollama offline — using local replies." : "");
      }
      if (data.now_playing) music.applyServerState(data.now_playing);
      await playReply(data.audio_b64);
      music.handleVoiceResult(data);
    } catch {
      setNote("Voice request failed.");
    } finally {
      voiceDone();
      busyRef.current = false;
      setStatus("idle");
    }
  }, [music, teardown]);

  // Spacebar hold-to-talk (ignore when typing in a field).
  useEffect(() => {
    if (!available) return;
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat && e.target === document.body) {
        e.preventDefault();
        void start();
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space" && recordingRef.current) {
        e.preventDefault();
        void stop();
      }
    };
    document.addEventListener("keydown", down);
    document.addEventListener("keyup", up);
    return () => {
      document.removeEventListener("keydown", down);
      document.removeEventListener("keyup", up);
    };
  }, [available, start, stop]);

  return {
    available,
    status,
    heard,
    reply,
    note,
    inputLevel,
    mics,
    selectedMic,
    setSelectedMic,
    start,
    stop,
  };
}
