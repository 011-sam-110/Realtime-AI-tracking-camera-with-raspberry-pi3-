import { cn } from "@/lib/utils";
import type { LiveStatus } from "@/api/live";
import type { ActivityStatus } from "@/api/types";

/** Connection dot for the live feed. */
export function ConnDot({ status }: { status: LiveStatus }) {
  const map: Record<LiveStatus, { color: string; label: string; pulse: boolean }> = {
    open: { color: "bg-emerald-400", label: "Live", pulse: true },
    connecting: { color: "bg-accent-amber", label: "Connecting", pulse: true },
    closed: { color: "bg-accent-rose", label: "Offline", pulse: false },
  };
  const s = map[status];
  return (
    <span className="inline-flex items-center gap-2 text-xs text-ink-muted">
      <span
        className={cn(
          "h-2 w-2 rounded-full",
          s.color,
          s.pulse && "animate-pulse-dot",
        )}
      />
      {s.label}
    </span>
  );
}

/** Perception/activity worker health badge. */
export function ActivityStatusPill({ status }: { status: ActivityStatus }) {
  const map: Record<
    ActivityStatus["state"],
    { color: string; label: string }
  > = {
    ok: { color: "bg-emerald-400", label: "Perception OK" },
    disabled: { color: "bg-ink-faint", label: "Perception off" },
    no_vision_model: { color: "bg-accent-amber", label: "No vision model" },
    error: { color: "bg-accent-rose", label: "Perception error" },
  };
  const s = map[status.state] ?? map.disabled;
  return (
    <span
      className="pill"
      title={status.message || s.label}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", s.color)} />
      {s.label}
    </span>
  );
}
