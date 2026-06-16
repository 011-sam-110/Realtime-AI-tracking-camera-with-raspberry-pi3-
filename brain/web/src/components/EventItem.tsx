import { useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  Box,
  LogOut,
  Sparkles,
  Trash2 as Mess,
} from "lucide-react";
import { thumbUrl } from "@/api/client";
import type { EventRow } from "@/api/types";
import { Badge } from "@/components/ui/Badge";
import { cn, fmtDateTime } from "@/lib/utils";

const TYPE_META: Record<
  string,
  { icon: typeof Activity; variant: "accent" | "violet" | "amber" | "rose" | "default"; label: string }
> = {
  activity: { icon: Activity, variant: "accent", label: "activity" },
  object: { icon: Box, variant: "violet", label: "object" },
  chore: { icon: Mess, variant: "amber", label: "chore" },
  presence: { icon: LogOut, variant: "default", label: "presence" },
};

function typeMeta(type: string) {
  return TYPE_META[type] ?? { icon: Sparkles, variant: "default" as const, label: type };
}

/** A single event row: thumbnail (if any), caption, type/source badges, time. */
export function EventItem({
  event,
  showPerson = true,
}: {
  event: EventRow;
  showPerson?: boolean;
}) {
  const meta = typeMeta(event.type);
  const Icon = meta.icon;
  const [thumbFailed, setThumbFailed] = useState(false);
  const hasThumb = !!event.thumb_ref && !thumbFailed;

  const caption =
    event.text ||
    [event.action, event.object, event.location ? `@ ${event.location}` : null]
      .filter(Boolean)
      .join(" ") ||
    "(event)";

  return (
    <div className="flex items-start gap-3 rounded-xl border border-line bg-white/[0.02] p-3 transition-colors hover:bg-white/[0.04]">
      <div className="relative h-14 w-14 shrink-0 overflow-hidden rounded-lg bg-base-800">
        {hasThumb ? (
          <img
            src={thumbUrl(event.thumb_ref as string)}
            alt=""
            loading="lazy"
            className="h-full w-full object-cover"
            onError={() => setThumbFailed(true)}
          />
        ) : (
          <div
            className={cn(
              "flex h-full w-full items-center justify-center",
              "text-ink-faint",
            )}
          >
            <Icon className="h-5 w-5" />
          </div>
        )}
      </div>

      <div className="min-w-0 flex-1">
        <p className="text-sm text-ink">{caption}</p>
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
          <Badge variant={meta.variant}>
            <Icon className="h-3 w-3" />
            {meta.label}
          </Badge>
          {event.source && event.source !== "vlm" && (
            <span className="pill">{event.source}</span>
          )}
          {showPerson && event.person_id != null && (
            <Link
              to={`/people/${event.person_id}`}
              className="text-[11px] text-accent hover:underline"
            >
              person #{event.person_id}
            </Link>
          )}
          {event.confidence > 0 && event.confidence < 1 && (
            <span className="text-[11px] text-ink-faint">
              {(event.confidence * 100).toFixed(0)}%
            </span>
          )}
        </div>
      </div>

      <time className="shrink-0 text-[11px] text-ink-faint" title={fmtDateTime(event.ts)}>
        {fmtDateTime(event.ts)}
      </time>
    </div>
  );
}
