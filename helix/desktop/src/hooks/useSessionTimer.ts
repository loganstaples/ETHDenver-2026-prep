import { useState, useEffect, useRef } from "react";

export function useSessionTimer(running: boolean, backendUptime: number) {
  const [seconds, setSeconds] = useState(backendUptime);
  const intervalRef = useRef<ReturnType<typeof setInterval>>();

  // Sync with backend on each poll update
  useEffect(() => {
    if (running) {
      setSeconds(backendUptime);
    } else {
      setSeconds(0);
    }
  }, [backendUptime, running]);

  // Client-side 1s tick between polls
  useEffect(() => {
    if (!running) {
      setSeconds(0);
      return;
    }
    intervalRef.current = setInterval(() => {
      setSeconds((s) => s + 1);
    }, 1000);
    return () => clearInterval(intervalRef.current);
  }, [running]);

  return seconds;
}
