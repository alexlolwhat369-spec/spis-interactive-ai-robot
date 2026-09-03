import { Settings } from "lucide-react";
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
import type { GestureInfo, RobotState } from "@/lib/api";

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
