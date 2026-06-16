import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Tailwind-aware className combiner (shadcn convention). */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Format a unix ts (seconds) as a short local time, e.g. "14:03". */
export function fmtTime(ts: number): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Format a unix ts (seconds) as a date + time, e.g. "Jun 10, 14:03". */
export function fmtDateTime(ts: number): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Format a unix ts as a calendar date, e.g. "Jun 10". */
export function fmtDate(ts: number): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleDateString([], {
    month: "short",
    day: "numeric",
  });
}

/** Relative "time ago", e.g. "3m ago", "just now", "2h ago". */
export function timeAgo(ts: number): string {
  if (!ts) return "never";
  const diff = Date.now() / 1000 - ts;
  if (diff < 5) return "just now";
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/** Humanise a duration in seconds, e.g. "1h 12m", "4m", "38s". */
export function fmtDuration(seconds: number): string {
  if (!seconds || seconds < 1) return "0s";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/** Deterministic accent color for a person id (for avatars/series). */
const PALETTE = [
  "#5eead4",
  "#a78bfa",
  "#fbbf24",
  "#fb7185",
  "#60a5fa",
  "#34d399",
  "#f472b6",
  "#facc15",
];
export function colorForId(id: number): string {
  return PALETTE[((id % PALETTE.length) + PALETTE.length) % PALETTE.length];
}

/** Initials from a label, e.g. "Sam" -> "S", "Unknown #3" -> "?". */
export function initials(label: string): string {
  if (!label) return "?";
  if (label.toLowerCase().startsWith("unknown")) return "?";
  const parts = label.trim().split(/\s+/);
  return (parts[0]?.[0] ?? "?").toUpperCase() + (parts[1]?.[0]?.toUpperCase() ?? "");
}

/** Whether a person row is an unnamed/unknown identity. */
export function isUnknown(p: { name: string | null; kind: string }): boolean {
  return !p.name || p.kind !== "known";
}
