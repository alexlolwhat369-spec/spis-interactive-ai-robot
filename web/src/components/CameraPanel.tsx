import { Mic } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { RobotState } from "@/lib/api";
import type { useVoice } from "@/hooks/useVoice";

type Voice = ReturnType<typeof useVoice>;

const MIC_LABEL: Record<string, string> = {
  idle: "Hold to talk",
  recording: "Listening…",
  thinking: "Thinking…",
};

export function CameraPanel({ state, voice }: { state: RobotState | null; voice: Voice }) {
  const label = state && state.gesture && state.gesture !== "none" ? state.gesture.replace(/_/g, " ") : "—";
  const reaction = state?.reaction ?? "—";
  const conf = Math.max(0, Math.min(1, state?.confidence ?? 0));

  const press = voice.status !== "thinking";
  const onDown = () => press && void voice.start();
  const onUp = () => void voice.stop();

  return (
    <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl border border-border bg-black">
      <img
        src="/camera.mjpg"
        alt="Live camera with hand-landmark overlays"
        className="h-full w-full object-cover"
      />

      {/* Live signal overlay (top) */}
      <div className="pointer-events-none absolute inset-x-0 top-0 flex flex-col gap-1.5 p-3">
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge variant="secondary" className="bg-background/70 backdrop-blur-sm capitalize">
            Gesture: <span className="ml-1 font-semibold text-foreground">{label}</span>
          </Badge>
          <Badge variant="secondary" className="bg-background/70 backdrop-blur-sm capitalize">
            Reaction: <span className="ml-1 font-semibold text-foreground">{reaction}</span>
          </Badge>
          <Badge variant="secondary" className="ml-auto bg-background/70 backdrop-blur-sm tabular-nums">
            {conf ? Math.round(conf * 100) + "%" : "—"}
          </Badge>
        </div>
        <div className="h-1 w-full overflow-hidden rounded-full bg-background/60">
          <div
            className="h-full rounded-full bg-primary transition-[width] duration-200"
            style={{ width: `${conf * 100}%` }}
          />
        </div>
      </div>

      {/* Hold-to-talk (bottom-left) */}
      {voice.available && (
        <div className="absolute bottom-3 left-3 flex flex-col gap-1.5">
          <Button
            variant={voice.status === "recording" ? "destructive" : "secondary"}
            className={cn(
              "bg-background/80 backdrop-blur-sm",
              voice.status === "recording" && "bg-destructive text-white",
            )}
            disabled={voice.status === "thinking"}
            onMouseDown={onDown}
            onMouseUp={onUp}
            onMouseLeave={() => voice.status === "recording" && onUp()}
            onTouchStart={(e) => {
              e.preventDefault();
              onDown();
            }}
            onTouchEnd={(e) => {
              e.preventDefault();
              onUp();
            }}
          >
            <Mic />
            {MIC_LABEL[voice.status]}
          </Button>
          {voice.status === "recording" && (
            <div className="h-1 w-40 overflow-hidden rounded-full bg-background/60">
              <div
                className="h-full rounded-full bg-success transition-[width] duration-100"
                style={{ width: `${voice.inputLevel * 100}%` }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
