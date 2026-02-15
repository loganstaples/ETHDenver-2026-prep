import { useState, useEffect, useRef, useCallback } from "react";
import type { ActivityEvent } from "../lib/types";
import { getActivityLog } from "../lib/tauri";

const MAX_ENTRIES = 50;
const POLL_INTERVAL = 2000;

export function useActivityLog(running: boolean) {
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const lastIdRef = useRef<number>(0);

  const poll = useCallback(async () => {
    if (!running) return;
    try {
      const newEvents = await getActivityLog(lastIdRef.current);
      if (newEvents.length > 0) {
        lastIdRef.current = Math.max(...newEvents.map((e) => e.id));
        setEvents((prev) => {
          const merged = [...newEvents, ...prev];
          return merged.slice(0, MAX_ENTRIES);
        });
      }
    } catch {
      // Backend not ready
    }
  }, [running]);

  useEffect(() => {
    if (!running) {
      setEvents([]);
      lastIdRef.current = 0;
      return;
    }
    poll();
    const interval = setInterval(poll, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [running, poll]);

  return events;
}
