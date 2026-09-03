import { Card } from "@/components/ui/card";
import type { useVoice } from "@/hooks/useVoice";

type Voice = ReturnType<typeof useVoice>;

export function Transcript({ voice }: { voice: Voice }) {
  return (
    <Card className="flex flex-col gap-2 p-3 sm:flex-row sm:items-center">
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm text-muted-foreground">{voice.heard}</div>
        {voice.reply && <div className="truncate text-sm text-foreground">{voice.reply}</div>}
        {voice.note && <div className="text-xs text-muted-foreground">{voice.note}</div>}
        {!voice.heard && !voice.reply && (
          <div className="text-sm text-muted-foreground">
            Hold the mic button or Spacebar, speak, then release.
          </div>
        )}
      </div>
      {voice.mics.length > 0 && (
        <select
          className="h-9 rounded-md border border-border bg-background px-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          value={voice.selectedMic}
          onChange={(e) => voice.setSelectedMic(e.target.value)}
          aria-label="Microphone"
        >
          <option value="">Default microphone</option>
          {voice.mics.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      )}
    </Card>
  );
}
