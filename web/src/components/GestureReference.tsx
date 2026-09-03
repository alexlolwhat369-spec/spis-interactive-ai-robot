import {
  CircleCheck,
  Hand,
  Heart,
  Sparkles,
  ThumbsUp,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { GestureInfo } from "@/lib/api";

const ICON: Record<string, LucideIcon> = {
  thumbs_up: ThumbsUp,
  peace: Hand,
  stop: Hand,
  heart: Heart,
  ok: CircleCheck,
  middle_finger: TriangleAlert,
  mohan: Sparkles,
};

export function GestureReference({ gestures }: { gestures: GestureInfo[] }) {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
      {gestures.map((g) => {
        const Icon = ICON[g.label] ?? Hand;
        return (
          <div
            key={g.label}
            className="flex items-start gap-3 rounded-lg border border-border bg-card p-3"
          >
            <div className="flex size-9 shrink-0 items-center justify-center rounded-md bg-muted text-foreground">
              <Icon className="size-5" />
            </div>
            <div className="min-w-0">
              <div className="text-sm font-medium capitalize">{g.label.replace(/_/g, " ")}</div>
              <div className="truncate text-sm text-muted-foreground">{g.reply || "—"}</div>
              <Badge variant="outline" className="mt-1.5 capitalize">
                {g.reaction}
              </Badge>
            </div>
          </div>
        );
      })}
    </div>
  );
}
