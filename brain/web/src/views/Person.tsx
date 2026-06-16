import { useCallback } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Clock, History } from "lucide-react";
import { getPerson, getTimeline } from "@/api/client";
import { useAsync } from "@/hooks/useAsync";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import {
  CenteredSpinner,
  EmptyState,
  ErrorState,
} from "@/components/ui/States";
import { PersonAvatar } from "@/components/PersonAvatar";
import { EventItem } from "@/components/EventItem";
import { fmtDateTime, isUnknown, timeAgo } from "@/lib/utils";

export default function Person() {
  const { id } = useParams<{ id: string }>();
  const personId = Number(id);

  const personFn = useCallback(() => getPerson(personId), [personId]);
  const timelineFn = useCallback(() => getTimeline(personId), [personId]);

  const person = useAsync(personFn, [personId]);
  const timeline = useAsync(timelineFn, [personId]);

  const p = person.data;

  return (
    <div className="space-y-6">
      <Link
        to="/people"
        className="inline-flex items-center gap-1.5 text-sm text-ink-muted hover:text-ink"
      >
        <ArrowLeft className="h-4 w-4" /> All people
      </Link>

      {person.loading && !p ? (
        <CenteredSpinner label="Loading person…" />
      ) : person.error ? (
        <ErrorState message={person.error} onRetry={person.refetch} />
      ) : !p ? (
        <EmptyState title="Person not found" hint="They may have been deleted or merged." />
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Profile */}
          <div className="space-y-4">
            <Card className="flex flex-col items-center gap-4 text-center">
              <div className="h-28 w-28 overflow-hidden rounded-2xl">
                <PersonAvatar id={p.id} label={p.label} fill />
              </div>
              <div>
                <h1 className="text-lg font-semibold text-ink">{p.label}</h1>
                <div className="mt-2 flex flex-wrap items-center justify-center gap-1.5">
                  {isUnknown(p) ? (
                    <Badge variant="amber">unknown</Badge>
                  ) : (
                    <Badge variant="success">known</Badge>
                  )}
                  {p.age != null && (
                    <Badge>~{Math.round(p.age)}y</Badge>
                  )}
                  {p.sex && <Badge>{p.sex}</Badge>}
                </div>
              </div>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Details</CardTitle>
              </CardHeader>
              <dl className="space-y-2 text-sm">
                <Detail label="Person ID" value={`#${p.id}`} />
                <Detail label="Kind" value={p.kind} />
                <Detail
                  label="First seen"
                  value={fmtDateTime(p.created_ts)}
                />
                <Detail
                  label="Last seen"
                  value={`${timeAgo(p.last_seen_ts)}`}
                  title={fmtDateTime(p.last_seen_ts)}
                />
              </dl>
            </Card>
          </div>

          {/* Timeline */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <History className="h-4 w-4 text-accent" />
                  Timeline
                </CardTitle>
                {timeline.data && (
                  <Badge variant="accent">{timeline.data.length} events</Badge>
                )}
              </CardHeader>

              {timeline.loading && !timeline.data ? (
                <CenteredSpinner />
              ) : timeline.error ? (
                <ErrorState message={timeline.error} onRetry={timeline.refetch} />
              ) : !timeline.data || timeline.data.length === 0 ? (
                <EmptyState
                  icon={<Clock className="h-6 w-6" />}
                  title="No activity yet"
                  hint={`${p.label} hasn't generated any events. They'll show up here as the perception worker captions the scene.`}
                />
              ) : (
                <div className="space-y-2">
                  {timeline.data.map((ev) => (
                    <EventItem key={ev.id} event={ev} showPerson={false} />
                  ))}
                </div>
              )}
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}

function Detail({
  label,
  value,
  title,
}: {
  label: string;
  value: string;
  title?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-line/50 pb-2 last:border-0">
      <dt className="text-xs text-ink-faint">{label}</dt>
      <dd className="text-right text-sm text-ink" title={title}>
        {value}
      </dd>
    </div>
  );
}
