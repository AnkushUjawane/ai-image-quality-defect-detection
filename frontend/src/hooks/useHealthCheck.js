import { useEffect, useState } from "react";
import { checkHealth } from "../api/client";

/**
 * Polls /api/health on an interval and exposes connection status.
 * Returns { online, modelLoaded, checking }.
 */
export function useHealthCheck(intervalMs = 15000) {
  const [state, setState] = useState({ online: false, modelLoaded: false, checking: true });

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const data = await checkHealth();
        if (!cancelled) {
          setState({ online: true, modelLoaded: !!data.model_loaded, checking: false });
        }
      } catch {
        if (!cancelled) {
          setState({ online: false, modelLoaded: false, checking: false });
        }
      }
    }

    poll();
    const id = setInterval(poll, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  return state;
}