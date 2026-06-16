/** Tiny async-data hook with loading/error/empty states + manual refetch. */
import { useCallback, useEffect, useRef, useState } from "react";

export interface AsyncState<T> {
  data: T | undefined;
  loading: boolean;
  error: string | undefined;
  refetch: () => void;
}

export function useAsync<T>(
  fn: () => Promise<T>,
  deps: ReadonlyArray<unknown> = [],
): AsyncState<T> {
  const [data, setData] = useState<T | undefined>(undefined);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | undefined>(undefined);
  const [nonce, setNonce] = useState(0);
  const alive = useRef(true);

  const run = useCallback(async () => {
    setLoading(true);
    setError(undefined);
    try {
      const result = await fn();
      if (alive.current) setData(result);
    } catch (e) {
      if (alive.current) setError((e as Error).message || "Request failed");
    } finally {
      if (alive.current) setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    alive.current = true;
    void run();
    return () => {
      alive.current = false;
    };
  }, [run, nonce]);

  const refetch = useCallback(() => setNonce((n) => n + 1), []);
  return { data, loading, error, refetch };
}
