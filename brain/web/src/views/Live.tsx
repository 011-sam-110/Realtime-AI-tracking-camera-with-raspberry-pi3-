import type { ReactNode } from "react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Compass, Gauge, ImageOff, Users } from "lucide-react";
import { useLiveFeed } from "@/hooks/useLiveFeed";
import { videoUrl } from "@/api/client";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { ConnDot, ActivityStatusPill } from "@/components/StatusDot";
import { PersonAvatar } from "@/components/PersonAvatar";
import { EmptyState } from "@/components/ui/States";
import { timeAgo } from "@/lib/utils";

function Stat({
  icon,
  label,
  value,
  hint,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="glass-soft flex items-center gap-3 px-4 py-3">
      <span className="rounded-lg bg-accent/10 p-2 text-accent">{icon}</span>
      <div className="min-w-0">
        <p className="text-[11px] uppercase tracking-wide text-ink-faint">
          {label}
        </p>
        <p className="truncate text-lg font-semibold tabular-nums text-ink">
          {value}
        </p>
        {hint && <p className="text-[11px] text-ink-muted">{hint}</p>}
      </div>
    </div>
  );
}

export default function Live() {
  const { frame, status } = useLiveFeed();
  const [imgFailed, setImgFailed] = useState(false);

  // Stable src so the MJPEG <img> doesn't restart the stream on every frame update.
  const src = useMemo(() => videoUrl(true), []);

  const pan = frame.servo_angles?.pan ?? 0;
  const tilt = frame.servo_angles?.tilt;
  const people = frame.people;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">Live</h1>
          <p className="text-sm text-ink-muted">
            Annotated feed, tracking, and who's in the room right now.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <ActivityStatusPill status={frame.activity_status} />
          <ConnDot status={status} />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Feed */}
        <div className="lg:col-span-2 space-y-4">
          <Card className="overflow-hidden p-0">
            <div className="relative aspect-[4/3] w-full bg-black">
              {!imgFailed ? (
                <img
                  src={src}
                  alt="Annotated live feed"
                  className="h-full w-full object-contain"
                  onError={() => setImgFailed(true)}
                />
              ) : (
                <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-ink-faint">
                  <ImageOff className="h-8 w-8" />
                  <p className="text-sm">Feed unavailable</p>
                  <p className="max-w-xs text-center text-xs text-ink-muted">
                    The brain isn't streaming /video. Start the pipeline to see
                    the annotated feed.
                  </p>
                </div>
              )}

              {/* Top-left overlay chips */}
              <div className="pointer-events-none absolute left-3 top-3 flex flex-wrap gap-2">
                <span className="pill bg-black/40 backdrop-blur">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      status === "open" ? "bg-emerald-400 animate-pulse-dot" : "bg-accent-rose"
                    }`}
                  />
                  {status === "open" ? "LIVE" : "no signal"}
                </span>
                <span className="pill bg-black/40 backdrop-blur tabular-nums">
                  {frame.fps.toFixed(1)} fps
                </span>
              </div>
            </div>
          </Card>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Stat
              icon={<Gauge className="h-4 w-4" />}
              label="FPS"
              value={frame.fps.toFixed(1)}
            />
            <Stat
              icon={<Compass className="h-4 w-4" />}
              label="Pan angle"
              value={`${pan.toFixed(0)}°`}
              hint={tilt != null ? `tilt ${tilt.toFixed(0)}°` : "pan axis"}
            />
            <Stat
              icon={<Users className="h-4 w-4" />}
              label="People"
              value={String(people.length)}
              hint={people.length === 1 ? "in frame" : "in frame"}
            />
          </div>
        </div>

        {/* Live people panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>In the room</CardTitle>
              <Badge variant={people.length ? "accent" : "default"}>
                {people.length} present
              </Badge>
            </CardHeader>

            {people.length === 0 ? (
              <EmptyState
                icon={<Users className="h-6 w-6" />}
                title="Nobody detected"
                hint="When someone steps into view they'll appear here with their live activity."
              />
            ) : (
              <ul className="space-y-2">
                {people.map((p) => (
                  <li key={p.track_id + ":" + p.person_id}>
                    <Link
                      to={`/people/${p.person_id}`}
                      className="flex items-center gap-3 rounded-xl border border-line bg-white/[0.02] p-2.5 transition-colors hover:bg-white/[0.05]"
                    >
                      <PersonAvatar id={p.person_id} label={p.label} size={42} />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <p className="truncate text-sm font-medium text-ink">
                            {p.label}
                          </p>
                          {p.kind === "known" ? (
                            <Badge variant="success">known</Badge>
                          ) : (
                            <Badge variant="amber">unknown</Badge>
                          )}
                        </div>
                        <p className="truncate text-xs text-ink-muted">
                          {p.current_activity || "—"}
                        </p>
                        <p className="text-[11px] text-ink-faint">
                          {[
                            p.age != null ? `~${Math.round(p.age)}y` : null,
                            p.sex || null,
                            `seen ${timeAgo(p.last_seen_ts)}`,
                          ]
                            .filter(Boolean)
                            .join(" · ")}
                        </p>
                      </div>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Perception</CardTitle>
            </CardHeader>
            <div className="space-y-2 text-sm">
              <Row label="Status">
                <ActivityStatusPill status={frame.activity_status} />
              </Row>
              {frame.activity_status.message && (
                <Row label="Detail">
                  <span className="text-right text-xs text-ink-muted">
                    {frame.activity_status.message}
                  </span>
                </Row>
              )}
              <Row label="Servo pan">
                <span className="tabular-nums text-ink">{pan.toFixed(1)}°</span>
              </Row>
              <Row label="Throughput">
                <span className="tabular-nums text-ink">
                  {frame.fps.toFixed(1)} fps
                </span>
              </Row>
              <Row label="Uptime feel">
                <span className="text-xs text-ink-muted">
                  {status === "open"
                    ? "streaming"
                    : status === "connecting"
                      ? "reconnecting…"
                      : "offline"}
                </span>
              </Row>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function Row({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-line/60 py-1.5 last:border-0">
      <span className="text-xs text-ink-faint">{label}</span>
      {children}
    </div>
  );
}
