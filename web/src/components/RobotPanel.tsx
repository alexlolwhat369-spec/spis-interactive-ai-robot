import { Music, Pause, Play, Square, SkipForward } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { MusicApi } from "@/hooks/useMusic";

export function RobotPanel({ music }: { music: MusicApi }) {
  const { ui } = music;
  const nowPlaying = ui.title ? `${ui.title}${ui.paused ? " · paused" : ""}` : "Nothing playing";

  return (
    <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl border border-border bg-black">
      <img
        src="/face.mjpg"
        alt="Animated robot face reacting in real time"
        className="h-full w-full object-cover"
      />

      {/* Music overlay (bottom of the robot screen) */}
      <div className="absolute inset-x-0 bottom-0 flex flex-col gap-2 border-t border-border/60 bg-background/75 p-3 backdrop-blur-sm">
        {!ui.available ? (
          <p className="text-sm text-muted-foreground">Music unavailable — playlist not configured.</p>
        ) : (
          <>
            <div className="flex items-center gap-2">
              <Music className="size-4 shrink-0 text-muted-foreground" />
              <span className="min-w-0 flex-1 truncate text-sm text-foreground">{nowPlaying}</span>
              <div className="flex items-center gap-1">
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label={ui.playing ? "Pause" : "Play"}
                  onClick={() => (ui.playing ? music.pause() : music.resume())}
                >
                  {ui.playing ? <Pause /> : <Play />}
                </Button>
                <Button size="icon" variant="ghost" aria-label="Next track" onClick={() => void music.next()}>
                  <SkipForward />
                </Button>
                <Button size="icon" variant="ghost" aria-label="Stop" onClick={() => void music.stop()}>
                  <Square />
                </Button>
              </div>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {ui.categories.map((c) => (
                <Button
                  key={c}
                  size="sm"
                  variant={ui.category === c ? "default" : "outline"}
                  className={cn("h-7 px-2.5 text-xs capitalize", ui.category !== c && "bg-transparent")}
                  onClick={() => void music.play(c)}
                >
                  {c}
                </Button>
              ))}
            </div>
            {ui.note && <p className="text-xs text-muted-foreground">{ui.note}</p>}
          </>
        )}
      </div>
    </div>
  );
}
