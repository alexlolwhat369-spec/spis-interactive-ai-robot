import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import type { RobotState } from "@/lib/api";

function Meter({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value || 0));
  return (
    <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-muted">
      <div
        className="h-full rounded-full bg-primary transition-[width] duration-200"
        style={{ width: `${pct * 100}%` }}
      />
    </div>
  );
}

function Row({ label, children, wide }: { label: string; children: ReactNode; wide?: boolean }) {
  return (
    <div className={cn("rounded-lg border border-border bg-card p-3", wide && "sm:col-span-2")}>
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 break-words font-mono text-sm text-foreground tabular-nums">{children}</div>
    </div>
  );
}

export function Diagnostics({ state }: { state: RobotState | null }) {
  const d = state?.diagnostics ?? null;
  const gesture = state && state.gesture && state.gesture !== "none" ? state.gesture.replace(/_/g, " ") : "—";
  const conf = Math.max(0, Math.min(1, state?.confidence ?? 0));
  const peak = Math.max(0, Math.min(1, d?.mic_peak ?? 0));
  const avg = Math.max(0, Math.min(1, d?.mic_average ?? 0));

  return (
    <div className="flex flex-col gap-4">
      {/* Live signal — updates every poll (~500ms). */}
      <div>
        <div className="mb-1.5 text-[11px] uppercase tracking-wide text-muted-foreground">Live</div>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <Row label="Gesture">
            <span className="capitalize">{gesture}</span>
          </Row>
          <Row label="Reaction">
            <span className="capitalize">{state?.reaction ?? "—"}</span>
          </Row>
          <Row label={`Confidence · ${Math.round(conf * 100)}%`}>
            <Meter value={conf} />
          </Row>
          <Row label="Camera">{state?.camera_backend || "—"}</Row>
        </div>
      </div>

      {/* Last voice turn — updates when you speak. */}
      <div>
        <div className="mb-1.5 text-[11px] uppercase tracking-wide text-muted-foreground">
          Last voice turn
        </div>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <Row label="Route">{d?.route || "idle"}</Row>
          <Row label="Action">{d?.action || "none"}</Row>
          <Row label="Transcript source">{d?.transcript_source || "none"}</Row>
          <Row label="Turn">{d?.sequence ?? 0}</Row>
          <Row label={`Mic peak · ${Math.round(peak * 100)}%`}>
            <Meter value={peak} />
          </Row>
          <Row label={`Mic average · ${Math.round(avg * 100)}%`}>
            <Meter value={avg} />
          </Row>
          <Row label="Heard" wide>
            {d?.heard || "—"}
          </Row>
          <Row label="Provider error" wide>
            <span className={cn(d?.provider_error && "text-destructive")}>
              {d?.provider_error || "none"}
            </span>
          </Row>
        </div>
      </div>
    </div>
  );
}
