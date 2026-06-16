import type { ReactNode } from "react";
import { Cloud, Cpu, Database, Info, SlidersHorizontal } from "lucide-react";
import { getConfig } from "@/api/client";
import { useAsync } from "@/hooks/useAsync";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { CenteredSpinner } from "@/components/ui/States";

export default function Settings() {
  const { data, loading } = useAsync(getConfig, []);

  // §11 lists no settings endpoint. getConfig() returns null on 404/offline —
  // in that case we show the documented defaults read-only with a clear note.
  const hasEndpoint = data != null;

  const recogThreshold = data?.recognition?.recog_threshold ?? 0.45;
  const vlmBackend = data?.vlm?.backend ?? "local";
  const retention = data?.retention_days ?? 30;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Settings</h1>
        <p className="text-sm text-ink-muted">
          Recognition, perception and retention configuration.
        </p>
      </div>

      {!hasEndpoint && (
        <Card className="border-accent-amber/30 bg-accent-amber/[0.04]">
          <div className="flex items-start gap-3">
            <Info className="mt-0.5 h-5 w-5 shrink-0 text-accent-amber" />
            <div className="text-sm">
              <p className="font-medium text-ink">Read-only — no settings API yet</p>
              <p className="mt-1 text-ink-muted">
                The brain doesn't expose a settings endpoint (INTERFACES §11), so
                these are the documented defaults from{" "}
                <code className="rounded bg-white/10 px-1 py-0.5 text-xs">config.json</code>.
                Edit that file on the brain host and restart to change them. Values
                here will go live automatically if a{" "}
                <code className="rounded bg-white/10 px-1 py-0.5 text-xs">
                  GET /api/config
                </code>{" "}
                endpoint is added.
              </p>
            </div>
          </div>
        </Card>
      )}

      {loading && !data ? (
        <CenteredSpinner />
      ) : (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <SlidersHorizontal className="h-4 w-4 text-accent" />
                Recognition
              </CardTitle>
              <CardDescription>face matching</CardDescription>
            </CardHeader>
            <SettingRow
              label="Recognition threshold"
              hint="cosine similarity to count as a match (higher = stricter)"
              value={recogThreshold.toFixed(2)}
            />
            <ThresholdBar value={recogThreshold} />
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {vlmBackend === "cloud" ? (
                  <Cloud className="h-4 w-4 text-accent-violet" />
                ) : (
                  <Cpu className="h-4 w-4 text-accent" />
                )}
                Vision model
              </CardTitle>
              <CardDescription>perception backend</CardDescription>
            </CardHeader>
            <SettingRow
              label="Backend"
              hint="local runs on the GPU; cloud uses the FreeLLMAPI gateway"
              value={
                <Badge variant={vlmBackend === "cloud" ? "violet" : "accent"}>
                  {vlmBackend}
                </Badge>
              }
            />
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-4 w-4 text-accent" />
                Retention
              </CardTitle>
              <CardDescription>rolling window</CardDescription>
            </CardHeader>
            <SettingRow
              label="Retention"
              hint="sightings + events older than this are pruned; people & face vectors are kept forever"
              value={`${retention} days`}
            />
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-4 w-4 text-accent" />
                About
              </CardTitle>
            </CardHeader>
            <div className="space-y-2 text-sm text-ink-muted">
              <p>
                Kitchen Vision is local-first: biometric data stays on the brain
                machine, the cloud VLM is opt-in, and deleting a person erases
                everything about them.
              </p>
              <p className="text-xs text-ink-faint">
                Dashboard served by the FastAPI brain (single origin).
              </p>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

function SettingRow({
  label,
  hint,
  value,
}: {
  label: string;
  hint?: string;
  value: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 py-1.5">
      <div className="min-w-0">
        <p className="text-sm text-ink">{label}</p>
        {hint && <p className="mt-0.5 text-xs text-ink-faint">{hint}</p>}
      </div>
      <div className="shrink-0 text-sm font-medium tabular-nums text-ink">
        {value}
      </div>
    </div>
  );
}

function ThresholdBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-white/5">
      <div
        className="h-full rounded-full bg-accent"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
