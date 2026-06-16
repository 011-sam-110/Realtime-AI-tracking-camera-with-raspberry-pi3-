/** React hook over the GET /events live feed (WS→SSE), with status (INTERFACES §11). */
import { useEffect, useRef, useState } from "react";
import { connectLive, type LiveStatus } from "@/api/live";
import type { LiveFrame } from "@/api/types";

const EMPTY: LiveFrame = {
  people: [],
  activity_status: { state: "disabled", message: "" },
  servo_angles: { pan: 0 },
  fps: 0,
};

export function useLiveFeed(): { frame: LiveFrame; status: LiveStatus } {
  const [frame, setFrame] = useState<LiveFrame>(EMPTY);
  const [status, setStatus] = useState<LiveStatus>("connecting");
  // Avoid re-subscribing on every render.
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    const dispose = connectLive({
      onFrame: (f) => {
        if (mounted.current) setFrame(f);
      },
      onStatus: (s) => {
        if (mounted.current) setStatus(s);
      },
    });
    return () => {
      mounted.current = false;
      dispose();
    };
  }, []);

  return { frame, status };
}
