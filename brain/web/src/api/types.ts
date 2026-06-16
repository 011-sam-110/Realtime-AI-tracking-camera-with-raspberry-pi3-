/**
 * Typed shapes for the Kitchen Vision brain HTTP/WS API (INTERFACES.md §11).
 *
 * These mirror the Python return shapes EXACTLY:
 *   - store.db.list_people()      -> Person (+ crop_urls added by the API layer)
 *   - store.db.get_events()       -> EventRow
 *   - store.db.get_timeline()     -> EventRow[]
 *   - STATE.people item           -> LivePerson
 *   - GET /events frame           -> LiveFrame
 *
 * The integrator must reconcile /api/stats against the api agent — see api/client.ts
 * notes and the subagent report. Everything else is pinned by §3/§4/§11.
 */

/** A person row as returned by store.db.list_people() / get_person() (INTERFACES §4). */
export interface Person {
  id: number;
  name: string | null;
  kind: string; // "known" | "unknown"
  age: number | null;
  sex: string | null; // "M" | "F" | null
  created_ts: number;
  last_seen_ts: number;
  label: string; // name, else "Unknown #N"
  /** Added by the API layer in GET /api/people: ["/api/faces/{id}/0.jpg", ...] (§11). */
  crop_urls?: string[];
}

/** An event row (store.db get_events/get_timeline → row dict). Mirrors Event (§3) + id. */
export interface EventRow {
  id: number;
  ts: number;
  person_id: number | null;
  type: string; // "activity" | "object" | "chore" | "presence"
  action: string | null;
  object: string | null;
  location: string | null;
  text: string;
  confidence: number;
  source: string; // "vlm" | "cv" | ...
  thumb_ref: string | null;
  payload: Record<string, unknown> | null;
}

/** A live person as published in STATE.people (INTERFACES §2). */
export interface LivePerson {
  person_id: number;
  label: string;
  kind: string;
  box: [number, number, number, number]; // [x, y, w, h]
  track_id: number;
  age: number | null;
  sex: string | null;
  current_activity: string;
  last_seen_ts: number;
}

/** Health of the perception worker (INTERFACES §2). */
export interface ActivityStatus {
  state: "ok" | "disabled" | "no_vision_model" | "error";
  message: string;
}

/** The ~1–4 Hz frame pushed over GET /events (WS) / streamed via SSE (INTERFACES §11). */
export interface LiveFrame {
  people: LivePerson[];
  activity_status: ActivityStatus;
  servo_angles: { pan: number; tilt?: number };
  fps: number;
}

// --- /api/stats --------------------------------------------------------------
// NOTE: §11 only says "per-person aggregates (presence time, event/chore counts)
// over time". The exact JSON shape is NOT pinned in INTERFACES, so this is the
// shape the SPA CONSUMES; the integrator must confirm the api agent emits it (or
// adapt the api agent / this file). See integration_notes in the subagent report.

/** A single time-bucketed sample for a person (e.g. one per day). */
export interface StatBucket {
  /** ISO date or unix-ts bucket label, e.g. "2026-06-10". */
  bucket: string;
  /** Seconds of presence in this bucket. */
  presence_seconds: number;
  /** Count of chore/mess events in this bucket. */
  chore_count: number;
  /** Count of all events in this bucket. */
  event_count: number;
}

export interface PersonStats {
  person_id: number;
  label: string;
  /** Total presence seconds across the window. */
  total_presence_seconds: number;
  /** Total chore/mess events across the window. */
  total_chore_count: number;
  /** Total events across the window. */
  total_event_count: number;
  /** Time series (oldest → newest). May be empty. */
  buckets: StatBucket[];
}

export interface StatsResponse {
  /** Per-person aggregates. */
  people: PersonStats[];
  /** Window the stats cover, in days (mirrors retention_days). Optional. */
  window_days?: number;
}

// --- Settings (read-only; may not have an endpoint yet) ----------------------
// §11 lists no settings endpoint. The Settings view shows config values if an
// endpoint exists, else a read-only note. If the api agent adds GET /api/config
// it should return (a subset of) core.config DEFAULTS; this is the consumed shape.
export interface ConfigResponse {
  retention_days?: number;
  recognition?: { recog_threshold?: number; [k: string]: unknown };
  vlm?: { backend?: "local" | "cloud"; [k: string]: unknown };
  [k: string]: unknown;
}
