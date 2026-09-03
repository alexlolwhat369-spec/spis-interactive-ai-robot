import { useMusic } from "@/hooks/useMusic";
import { useRobotState } from "@/hooks/useRobotState";
import { useVoice } from "@/hooks/useVoice";
import { CameraPanel } from "@/components/CameraPanel";
import { RobotPanel } from "@/components/RobotPanel";
import { SettingsSheet } from "@/components/SettingsSheet";
import { Transcript } from "@/components/Transcript";

export default function App() {
  const music = useMusic();
  const { state, gestures, stopGestureSound } = useRobotState(music);
  const voice = useVoice(music, stopGestureSound);

  return (
    <div className="mx-auto flex min-h-full max-w-6xl flex-col gap-4 p-4 sm:p-6">
      <header className="flex items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">SPIS Interactive AI Robot</h1>
          <p className="text-sm text-muted-foreground">Gestures · voice · music — running on-device</p>
        </div>
        <SettingsSheet gestures={gestures} state={state} />
      </header>

      <main className="grid gap-4 lg:grid-cols-2">
        <CameraPanel state={state} voice={voice} />
        <RobotPanel music={music} />
      </main>

      {voice.available && <Transcript voice={voice} />}

      <footer className="mt-auto pt-2 text-center text-xs text-muted-foreground">
        On-device pipeline — camera, speech, and music stay on this machine. No images or audio are
        stored.
      </footer>
    </div>
  );
}
