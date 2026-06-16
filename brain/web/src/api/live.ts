/**
 * Live connection to GET /events (INTERFACES.md §11): WebSocket preferred, SSE
 * fallback, with auto-reconnect/backoff. Yields LiveFrame objects to a callback.
 *
 * §11: "GET /events (WS preferred; SSE fallback) — ~1–4 Hz
 *       {people, activity_status, servo_angles, fps}".
 *
 * The same path serves both transports; we try a WebSocket upgrade first and, if
 * it never opens (or closes immediately), fall back to EventSource. The hook
 * useLiveFeed() in ../hooks wraps this for React.
 */
import type { LiveFrame } from "./types";

export type LiveStatus = "connecting" | "open" | "closed";

export interface LiveHandlers {
  onFrame: (frame: LiveFrame) => void;
  onStatus?: (status: LiveStatus) => void;
}

function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/events`;
}

function parseFrame(raw: string): LiveFrame | null {
  try {
    const data = JSON.parse(raw) as Partial<LiveFrame>;
    if (!data || typeof data !== "object") return null;
    return {
      people: Array.isArray(data.people) ? data.people : [],
      activity_status:
        data.activity_status ?? { state: "disabled", message: "" },
      servo_angles: data.servo_angles ?? { pan: 0 },
      fps: typeof data.fps === "number" ? data.fps : 0,
    };
  } catch {
    return null;
  }
}

/**
 * Connect to the live feed. Returns a disposer that tears everything down.
 * Reconnects with capped exponential backoff; transparently downgrades WS→SSE.
 */
export function connectLive(handlers: LiveHandlers): () => void {
  let disposed = false;
  let ws: WebSocket | null = null;
  let es: EventSource | null = null;
  let retry = 0;
  let timer: number | undefined;
  let openedOnce = false;

  const setStatus = (s: LiveStatus) => handlers.onStatus?.(s);

  const cleanup = () => {
    if (ws) {
      ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null;
      try {
        ws.close();
      } catch {
        /* ignore */
      }
      ws = null;
    }
    if (es) {
      es.onmessage = es.onerror = null;
      try {
        es.close();
      } catch {
        /* ignore */
      }
      es = null;
    }
  };

  const scheduleReconnect = () => {
    if (disposed) return;
    setStatus("closed");
    cleanup();
    const delay = Math.min(1000 * 2 ** retry, 15000);
    retry += 1;
    timer = window.setTimeout(connectSse /* prefer the stable transport on retry */, delay);
  };

  const onMessage = (data: string) => {
    const frame = parseFrame(data);
    if (frame) {
      retry = 0;
      handlers.onFrame(frame);
    }
  };

  function connectSse() {
    if (disposed) return;
    setStatus("connecting");
    cleanup();
    try {
      es = new EventSource("/events");
    } catch {
      scheduleReconnect();
      return;
    }
    es.onopen = () => {
      openedOnce = true;
      retry = 0;
      setStatus("open");
    };
    es.onmessage = (ev) => onMessage(ev.data);
    es.onerror = () => {
      // EventSource auto-retries, but if it never opened, back off ourselves.
      if (!openedOnce) scheduleReconnect();
      else setStatus("connecting");
    };
  }

  function connectWs() {
    if (disposed) return;
    setStatus("connecting");
    cleanup();
    let opened = false;
    try {
      ws = new WebSocket(wsUrl());
    } catch {
      connectSse();
      return;
    }
    ws.onopen = () => {
      opened = true;
      openedOnce = true;
      retry = 0;
      setStatus("open");
    };
    ws.onmessage = (ev) => onMessage(String(ev.data));
    ws.onerror = () => {
      if (!opened) {
        // WS not supported / route is SSE-only → fall back immediately.
        connectSse();
      }
    };
    ws.onclose = () => {
      if (!opened) {
        connectSse();
      } else {
        scheduleReconnect();
      }
    };
  }

  connectWs();

  return () => {
    disposed = true;
    if (timer) window.clearTimeout(timer);
    cleanup();
  };
}
