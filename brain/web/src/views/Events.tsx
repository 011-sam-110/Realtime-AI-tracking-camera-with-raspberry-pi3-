import { useCallback, useMemo, useState } from "react";
import { Activity, Filter, RefreshCw } from "lucide-react";
import { getEvents, getPeople, type EventQuery } from "@/api/client";
import { useAsync } from "@/hooks/useAsync";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Select } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import {
  CenteredSpinner,
  EmptyState,
  ErrorState,
} from "@/components/ui/States";
import { EventItem } from "@/components/EventItem";
import { fmtDate } from "@/lib/utils";

const TYPE_OPTIONS = ["all", "activity", "object", "chore", "presence"] as const;
const SINCE_OPTIONS = [
  { label: "All time", hours: 0 },
  { label: "Last hour", hours: 1 },
  { label: "Last 24h", hours: 24 },
  { label: "Last 7 days", hours: 24 * 7 },
];

export default function Events() {
  const [typeFilter, setTypeFilter] =
    useState<(typeof TYPE_OPTIONS)[number]>("all");
  const [personFilter, setPersonFilter] = useState<number | "">("");
  const [sinceHours, setSinceHours] = useState(0);

  const peopleAsync = useAsync(getPeople, []);

  const since = useMemo(
    () => (sinceHours > 0 ? Date.now() / 1000 - sinceHours * 3600 : undefined),
    [sinceHours],
  );

  const eventsFn = useCallback(() => {
    const q: EventQuery = { limit: 300 };
    if (personFilter !== "") q.person_id = Number(personFilter);
    if (since != null) q.since = since;
    return getEvents(q);
  }, [personFilter, since]);

  const events = useAsync(eventsFn, [personFilter, since]);

  // Type filter is client-side (the API filters by person/since/limit per §11).
  const filtered = (events.data ?? []).filter(
    (e) => typeFilter === "all" || e.type === typeFilter,
  );

  // Group by calendar day for a tidy feed.
  const groups = useMemo(() => {
    const map = new Map<string, typeof filtered>();
    for (const e of filtered) {
      const key = fmtDate(e.ts);
      const arr = map.get(key) ?? [];
      arr.push(e);
      map.set(key, arr);
    }
    return Array.from(map.entries());
  }, [filtered]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">Events</h1>
          <p className="text-sm text-ink-muted">
            Everything the brain has logged — activities, objects, chores.
          </p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={events.refetch}
          disabled={events.loading}
        >
          <RefreshCw
            className={`h-4 w-4 ${events.loading ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </div>

      <Card className="p-4">
        <div className="flex flex-wrap items-center gap-3">
          <span className="flex items-center gap-1.5 text-xs text-ink-faint">
            <Filter className="h-3.5 w-3.5" /> Filters
          </span>
          <Select
            value={typeFilter}
            onChange={(e) =>
              setTypeFilter(e.target.value as (typeof TYPE_OPTIONS)[number])
            }
            className="w-36"
          >
            {TYPE_OPTIONS.map((t) => (
              <option key={t} value={t}>
                {t === "all" ? "All types" : t}
              </option>
            ))}
          </Select>
          <Select
            value={personFilter}
            onChange={(e) =>
              setPersonFilter(
                e.target.value === "" ? "" : Number(e.target.value),
              )
            }
            className="w-44"
          >
            <option value="">All people</option>
            {(peopleAsync.data ?? []).map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </Select>
          <Select
            value={sinceHours}
            onChange={(e) => setSinceHours(Number(e.target.value))}
            className="w-36"
          >
            {SINCE_OPTIONS.map((o) => (
              <option key={o.hours} value={o.hours}>
                {o.label}
              </option>
            ))}
          </Select>
          <div className="ml-auto">
            <Badge variant="accent">{filtered.length} shown</Badge>
          </div>
        </div>
      </Card>

      {events.loading && !events.data ? (
        <CenteredSpinner label="Loading events…" />
      ) : events.error ? (
        <ErrorState message={events.error} onRetry={events.refetch} />
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={<Activity className="h-6 w-6" />}
          title="No events"
          hint="Nothing matches these filters yet. As the perception worker watches the room, events will stream in here."
        />
      ) : (
        <div className="space-y-6">
          {groups.map(([day, items]) => (
            <Card key={day}>
              <CardHeader>
                <CardTitle>{day}</CardTitle>
                <Badge>{items.length}</Badge>
              </CardHeader>
              <div className="space-y-2">
                {items.map((ev) => (
                  <EventItem key={ev.id} event={ev} />
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
