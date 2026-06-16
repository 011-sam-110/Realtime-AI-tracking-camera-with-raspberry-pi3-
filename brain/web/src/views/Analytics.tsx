import type { ReactNode } from "react";
import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BarChart3, Clock, TrendingUp } from "lucide-react";
import { getStats } from "@/api/client";
import type { PersonStats, StatsResponse } from "@/api/types";
import { useAsync } from "@/hooks/useAsync";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import {
  CenteredSpinner,
  EmptyState,
  ErrorState,
} from "@/components/ui/States";
import { colorForId, fmtDuration } from "@/lib/utils";

const TOOLTIP_STYLE = {
  background: "rgba(11,15,26,0.95)",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 12,
  color: "#e6ebf5",
  fontSize: 12,
} as const;

export default function Analytics() {
  const { data, loading, error, refetch } = useAsync(getStats, []);

  if (loading && !data) return <CenteredSpinner label="Crunching analytics…" />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;

  const stats: StatsResponse = data ?? { people: [] };
  const people = stats.people ?? [];

  if (people.length === 0) {
    return (
      <div className="space-y-6">
        <Header window={stats.window_days} />
        <EmptyState
          icon={<BarChart3 className="h-6 w-6" />}
          title="No analytics yet"
          hint="Once the brain logs presence and events for a few people, you'll see presence time and chore/mess trends here."
        />
      </div>
    );
  }

  return <AnalyticsBody stats={stats} />;
}

function Header({ window }: { window?: number }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">
          Analytics
        </h1>
        <p className="text-sm text-ink-muted">
          Presence time and chore/mess activity per person over time.
        </p>
      </div>
      {window != null && <Badge variant="accent">last {window} days</Badge>}
    </div>
  );
}

function AnalyticsBody({ stats }: { stats: StatsResponse }) {
  const people = stats.people;

  // Presence (hours) per person — bar chart.
  const presenceData = useMemo(
    () =>
      people.map((p) => ({
        name: p.label,
        hours: +(p.total_presence_seconds / 3600).toFixed(2),
        id: p.person_id,
      })),
    [people],
  );

  // Chore vs total events per person — grouped bar.
  const eventsData = useMemo(
    () =>
      people.map((p) => ({
        name: p.label,
        chores: p.total_chore_count,
        events: p.total_event_count,
        id: p.person_id,
      })),
    [people],
  );

  // Presence over time — one line per person across the union of buckets.
  const timeSeries = useMemo(() => buildTimeSeries(people), [people]);

  // Chore share — pie.
  const choreShare = useMemo(
    () =>
      people
        .filter((p) => p.total_chore_count > 0)
        .map((p) => ({
          name: p.label,
          value: p.total_chore_count,
          id: p.person_id,
        })),
    [people],
  );

  const totals = useMemo(
    () => ({
      presence: people.reduce((a, p) => a + p.total_presence_seconds, 0),
      events: people.reduce((a, p) => a + p.total_event_count, 0),
      chores: people.reduce((a, p) => a + p.total_chore_count, 0),
    }),
    [people],
  );

  return (
    <div className="space-y-6">
      <Header window={stats.window_days} />

      {/* KPI row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Kpi
          icon={<Clock className="h-4 w-4" />}
          label="Total presence"
          value={fmtDuration(totals.presence)}
        />
        <Kpi
          icon={<TrendingUp className="h-4 w-4" />}
          label="Total events"
          value={String(totals.events)}
        />
        <Kpi
          icon={<BarChart3 className="h-4 w-4" />}
          label="Chore / mess events"
          value={String(totals.chores)}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Presence per person</CardTitle>
            <CardDescription>hours in view</CardDescription>
          </CardHeader>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={presenceData} margin={{ left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="name" tick={{ fill: "#8a93a6", fontSize: 11 }} />
              <YAxis tick={{ fill: "#8a93a6", fontSize: 11 }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="hours" radius={[6, 6, 0, 0]}>
                {presenceData.map((d) => (
                  <Cell key={d.id} fill={colorForId(d.id)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Events vs chores</CardTitle>
            <CardDescription>per person</CardDescription>
          </CardHeader>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={eventsData} margin={{ left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="name" tick={{ fill: "#8a93a6", fontSize: 11 }} />
              <YAxis tick={{ fill: "#8a93a6", fontSize: 11 }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Legend wrapperStyle={{ fontSize: 12, color: "#8a93a6" }} />
              <Bar dataKey="events" name="all events" fill="#a78bfa" radius={[6, 6, 0, 0]} />
              <Bar dataKey="chores" name="chores" fill="#fbbf24" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card className={timeSeries.keys.length ? "" : "lg:col-span-2"}>
          <CardHeader>
            <CardTitle>Presence over time</CardTitle>
            <CardDescription>hours per bucket</CardDescription>
          </CardHeader>
          {timeSeries.rows.length === 0 ? (
            <EmptyState
              title="No time series"
              hint="Per-bucket data isn't available yet."
            />
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={timeSeries.rows} margin={{ left: -16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="bucket" tick={{ fill: "#8a93a6", fontSize: 11 }} />
                <YAxis tick={{ fill: "#8a93a6", fontSize: 11 }} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Legend wrapperStyle={{ fontSize: 12, color: "#8a93a6" }} />
                {timeSeries.keys.map((k) => (
                  <Line
                    key={k.id}
                    type="monotone"
                    dataKey={k.label}
                    stroke={colorForId(k.id)}
                    strokeWidth={2}
                    dot={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        {choreShare.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Chore share</CardTitle>
              <CardDescription>who makes / handles the mess</CardDescription>
            </CardHeader>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={choreShare}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={55}
                  outerRadius={90}
                  paddingAngle={3}
                >
                  {choreShare.map((d) => (
                    <Cell key={d.id} fill={colorForId(d.id)} stroke="transparent" />
                  ))}
                </Pie>
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Legend wrapperStyle={{ fontSize: 12, color: "#8a93a6" }} />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        )}
      </div>
    </div>
  );
}

function Kpi({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <Card className="flex items-center gap-3">
      <span className="rounded-lg bg-accent/10 p-2.5 text-accent">{icon}</span>
      <div>
        <p className="text-[11px] uppercase tracking-wide text-ink-faint">
          {label}
        </p>
        <p className="text-xl font-semibold tabular-nums text-ink">{value}</p>
      </div>
    </Card>
  );
}

/**
 * Pivot per-person bucket series into rows keyed by bucket, one column per person
 * (value = hours). Tolerates people with no/empty buckets.
 */
function buildTimeSeries(people: PersonStats[]): {
  rows: Array<Record<string, string | number>>;
  keys: Array<{ id: number; label: string }>;
} {
  const buckets = new Set<string>();
  for (const p of people) for (const b of p.buckets ?? []) buckets.add(b.bucket);
  const ordered = Array.from(buckets).sort();
  const keys = people
    .filter((p) => (p.buckets ?? []).length > 0)
    .map((p) => ({ id: p.person_id, label: p.label }));

  const rows = ordered.map((bucket) => {
    const row: Record<string, string | number> = { bucket };
    for (const p of people) {
      const b = (p.buckets ?? []).find((x) => x.bucket === bucket);
      row[p.label] = b ? +(b.presence_seconds / 3600).toFixed(2) : 0;
    }
    return row;
  });

  return { rows, keys };
}
