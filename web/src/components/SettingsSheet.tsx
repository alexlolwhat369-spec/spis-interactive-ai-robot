import { Settings, Volume2, VolumeX } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { GestureReference } from "@/components/GestureReference";
import { Diagnostics } from "@/components/Diagnostics";
import { useAudioSettings } from "@/lib/audio";
import type { GestureInfo, RobotState } from "@/lib/api";

function AudioControls() {
  const { volume, muted, setVolume, setMuted } = useAudioSettings();
  const percent = Math.round((muted ? 0 : volume) * 100);
  return (
    <div className="flex items-center gap-3">
      <Button
        variant="outline"
        size="icon"
        aria-label={muted ? "Unmute" : "Mute"}
        aria-pressed={muted}
        onClick={() => setMuted(!muted)}
      >
        {muted ? <VolumeX /> : <Volume2 />}
      </Button>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={muted ? 0 : volume}
        aria-label="Volume"
        onChange={(e) => {
          setVolume(Number(e.target.value));
          if (muted) setMuted(false);
        }}
        className="h-2 flex-1 cursor-pointer accent-primary"
      />
      <span className="w-9 shrink-0 text-right text-sm tabular-nums text-muted-foreground">
        {percent}%
      </span>
    </div>
  );
}

export function SettingsSheet({
  gestures,
  state,
  onPreviewSound,
}: {
  gestures: GestureInfo[];
  state: RobotState | null;
  onPreviewSound: (label: string) => void;
}) {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="outline" size="icon" aria-label="Settings">
          <Settings />
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="max-w-md">
        <SheetHeader>
          <SheetTitle>Settings</SheetTitle>
          <SheetDescription>Gesture reference and live diagnostics.</SheetDescription>
        </SheetHeader>
        <ScrollArea className="-mr-4 flex-1 pr-4">
          <div className="flex flex-col gap-6 pb-6">
            <section>
              <h3 className="mb-2 text-sm font-medium text-muted-foreground">Audio</h3>
              <AudioControls />
              <p className="mt-2 text-xs text-muted-foreground">
                Volume for music, gesture effects, reactions, and voice replies.
              </p>
            </section>
            <Separator />
            <section>
              <h3 className="mb-2 text-sm font-medium text-muted-foreground">Gesture reference</h3>
              <GestureReference gestures={gestures} onPreviewSound={onPreviewSound} />
            </section>
            <Separator />
            <section>
              <h3 className="mb-2 text-sm font-medium text-muted-foreground">Diagnostics</h3>
              <Diagnostics state={state} />
            </section>
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
