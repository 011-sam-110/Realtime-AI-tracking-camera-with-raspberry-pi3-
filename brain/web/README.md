# Kitchen Vision — Dashboard SPA (`web/`)

The React dashboard (INTERFACES.md §12). Vite + React + TS + Tailwind + Recharts.
Hand-rolled shadcn-style primitives (cva + tailwind-merge) under `src/components/ui/`.

## Develop
```
npm install
npm run dev        # http://localhost:5173, proxies /api /video /events -> :8090
```
Override the brain target: `VITE_API_TARGET=http://otherhost:8090 npm run dev`.

## Build (prod)
```
npm run build      # type-checks (tsc -b) then emits web/dist
```
FastAPI serves `web/dist` at `/` with an SPA fallback to `index.html` (INTERFACES §11).
All API paths in `src/api/` are **relative**, so the built app works from the same
origin with no proxy.

## What it consumes (all in `src/api/`)
- `src/api/types.ts` — typed shapes mirroring the Python return values (§3/§4).
- `src/api/client.ts` — the §11 REST endpoints (one function each).
- `src/api/live.ts` — `GET /events` live feed: WebSocket preferred, SSE fallback,
  auto-reconnect with backoff.

See the subagent report `integration_notes` for the exact endpoint/shape list the
integrator must reconcile against the api agent — notably `/api/stats` and the
optional `/api/config` (whose shapes are not pinned in INTERFACES).

## Views
Live · People · Person · Events · Analytics · Settings. Every view has explicit
loading (skeleton/spinner), empty, and error (incl. "brain not reachable") states,
so the UI is fully functional with no backend running.
