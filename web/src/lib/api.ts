export interface MusicState {
  available: boolean;
  categories: string[];
  title: string | null;
  category: string | null;
  url: string | null;
  ok?: boolean;
  reason?: string;
}

export interface Diagnostics {
  sequence: number;
  heard: string;
  route: string;
  action: string;
  reaction: string;
  reply: string;
  provider_error: string | null;
  mic_peak: number;
  mic_average: number;
  transcript_source: string;
}

export interface RobotState {
  gesture: string;
  gesture_event?: number;
  reaction: string;
  confidence: number;
  camera_backend?: string;
  music: MusicState;
  diagnostics: Diagnostics;
}

export interface GestureInfo {
  label: string;
  reply: string;
  reaction: string;
  sound: string | null;
  sound_name?: string | null;
}

export interface VoiceResult {
  heard: string;
  reply: string;
  reaction: string;
  route: string;
  action: string;
  provider_error: string | null;
  now_playing: MusicState | null;
  audio_b64: string;
  audio_seconds: number;
}

const json = { "Content-Type": "application/json" };

export async function getState(): Promise<RobotState | null> {
  try {
    const res = await fetch("/state", { cache: "no-store" });
    return res.ok ? ((await res.json()) as RobotState) : null;
  } catch {
    return null;
  }
}

export async function getGestures(): Promise<GestureInfo[]> {
  try {
    const res = await fetch("/gestures", { cache: "no-store" });
    if (!res.ok) return [];
    return ((await res.json()) as { gestures: GestureInfo[] }).gestures;
  } catch {
    return [];
  }
}

export async function voiceAvailable(): Promise<boolean> {
  try {
    const res = await fetch("/voice/available", { cache: "no-store" });
    return (await res.json()).available === true;
  } catch {
    return false;
  }
}

export async function postMusic(path: string, body?: unknown): Promise<MusicState> {
  const res = await fetch(path, {
    method: "POST",
    headers: json,
    body: body ? JSON.stringify(body) : undefined,
  });
  return (await res.json()) as MusicState;
}

export function voiceListening(): void {
  void fetch("/voice/listening", { method: "POST" }).catch(() => {});
}
export function voiceDone(): void {
  void fetch("/voice/done", { method: "POST" }).catch(() => {});
}

export async function postVoice(pcm: ArrayBuffer): Promise<VoiceResult> {
  const res = await fetch("/voice", {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: pcm,
  });
  return (await res.json()) as VoiceResult;
}
