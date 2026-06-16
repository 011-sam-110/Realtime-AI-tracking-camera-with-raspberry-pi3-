import * as React from "react";
import { AlertTriangle, Inbox, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "./Button";

/** Skeleton block for loading states. */
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("shimmer rounded-lg", className)} />;
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn("h-5 w-5 animate-spin text-accent", className)} />;
}

export function EmptyState({
  icon,
  title,
  hint,
  action,
}: {
  icon?: React.ReactNode;
  title: string;
  hint?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-line bg-white/[0.015] px-6 py-14 text-center">
      <div className="rounded-full bg-white/5 p-3 text-ink-faint">
        {icon ?? <Inbox className="h-6 w-6" />}
      </div>
      <div>
        <p className="text-sm font-medium text-ink">{title}</p>
        {hint && <p className="mt-1 max-w-sm text-xs text-ink-muted">{hint}</p>}
      </div>
      {action}
    </div>
  );
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  const offline = /network error/i.test(message);
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-accent-amber/30 bg-accent-amber/[0.04] px-6 py-12 text-center">
      <div className="rounded-full bg-accent-amber/10 p-3 text-accent-amber">
        <AlertTriangle className="h-6 w-6" />
      </div>
      <div>
        <p className="text-sm font-medium text-ink">
          {offline ? "Brain not reachable" : "Something went wrong"}
        </p>
        <p className="mt-1 max-w-sm text-xs text-ink-muted">
          {offline
            ? "The Kitchen Vision backend isn't running or can't be reached. Start it and retry."
            : message}
        </p>
      </div>
      {onRetry && (
        <Button variant="secondary" size="sm" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

export function CenteredSpinner({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-ink-muted">
      <Spinner />
      {label && <p className="text-xs">{label}</p>}
    </div>
  );
}
