/**
 * Typed HTTP client for the Kitchen Vision brain API (INTERFACES.md §11).
 *
 * Covers EXACTLY the §11 endpoints the SPA consumes. Every call returns the typed
 * shape from ./types. All paths are relative so they work behind the Vite dev proxy
 * AND from the FastAPI-served prod build (single origin).
 *
 * Endpoints consumed (path · method · request · response):
 *   GET    /api/people                         -> { people: Person[] }   (each +crop_urls)
 *   POST   /api/people/{id}/name   {name}      -> any (success ack)
 *   POST   /api/people/{id}/merge  {into}      -> any (success ack)
 *   DELETE /api/people/{id}                    -> any (success ack)
 *   GET    /api/people/{id}/timeline           -> { timeline: EventRow[] }
 *   GET    /api/events?since=&limit=&person_id= -> { events: EventRow[] }
 *   GET    /api/stats                          -> StatsResponse        (shape NOT pinned — see types.ts)
 *   GET    /api/config                         -> ConfigResponse       (optional — may 404)
 *   GET    /api/faces/{id}/{n}.jpg             (img src, not fetched here)
 *   GET    /api/thumbs/{ref}.jpg               (img src, not fetched here)
 *   GET    /video                              (img src — annotated MJPEG)
 *   GET    /events                             (WS preferred, SSE fallback — see ./live)
 */
import type {
  ConfigResponse,
  EventRow,
  Person,
  StatsResponse,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      headers: { Accept: "application/json", ...(init?.headers || {}) },
      ...init,
    });
  } catch (e) {
    throw new ApiError(0, `Network error: ${(e as Error).message}`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = String((body as { detail: unknown }).detail);
      }
    } catch {
      /* non-JSON error body — keep statusText */
    }
    throw new ApiError(res.status, detail || `HTTP ${res.status}`);
  }
  // 204 / empty body tolerance.
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

function jsonBody(data: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  };
}

// --- People ------------------------------------------------------------------

export async function getPeople(): Promise<Person[]> {
  const data = await req<{ people: Person[] }>("/api/people");
  return data.people ?? [];
}

export async function getPerson(id: number): Promise<Person | undefined> {
  // No dedicated GET /api/people/{id} in §11 — resolve from the list.
  const people = await getPeople();
  return people.find((p) => p.id === id);
}

export async function renamePerson(id: number, name: string): Promise<void> {
  await req(`/api/people/${id}/name`, jsonBody({ name }));
}

export async function mergePerson(id: number, into: number): Promise<void> {
  await req(`/api/people/${id}/merge`, jsonBody({ into }));
}

export async function deletePerson(id: number): Promise<void> {
  await req(`/api/people/${id}`, { method: "DELETE" });
}

export async function getTimeline(id: number): Promise<EventRow[]> {
  const data = await req<{ timeline: EventRow[] }>(
    `/api/people/${id}/timeline`,
  );
  return data.timeline ?? [];
}

// --- Events ------------------------------------------------------------------

export interface EventQuery {
  since?: number;
  limit?: number;
  person_id?: number;
}

export async function getEvents(q: EventQuery = {}): Promise<EventRow[]> {
  const params = new URLSearchParams();
  if (q.since != null) params.set("since", String(q.since));
  if (q.limit != null) params.set("limit", String(q.limit));
  if (q.person_id != null) params.set("person_id", String(q.person_id));
  const qs = params.toString();
  const data = await req<{ events: EventRow[] }>(
    `/api/events${qs ? `?${qs}` : ""}`,
  );
  return data.events ?? [];
}

// --- Analytics ---------------------------------------------------------------

export async function getStats(): Promise<StatsResponse> {
  return req<StatsResponse>("/api/stats");
}

// --- Settings / config (optional endpoint) -----------------------------------

export async function getConfig(): Promise<ConfigResponse | null> {
  try {
    return await req<ConfigResponse>("/api/config");
  } catch (e) {
    if (e instanceof ApiError && (e.status === 404 || e.status === 0)) {
      return null; // no settings endpoint yet — Settings view shows a note.
    }
    throw e;
  }
}

// --- Static media URL builders (used as <img src>) ----------------------------

/** Annotated MJPEG feed. Add a cache-buster to force a fresh stream on remount. */
export function videoUrl(cacheBust = false): string {
  return cacheBust ? `/video?t=${Date.now()}` : "/video";
}

/** Face crop n for a person (newest = 0). INTERFACES §11 GET /api/faces/{id}/{n}.jpg. */
export function faceUrl(personId: number, n = 0): string {
  return `/api/faces/${personId}/${n}.jpg`;
}

/** Event thumbnail by ref. INTERFACES §11 GET /api/thumbs/{ref}.jpg. */
export function thumbUrl(ref: string): string {
  return `/api/thumbs/${ref}.jpg`;
}
