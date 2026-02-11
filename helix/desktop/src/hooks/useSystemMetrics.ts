import { useState, useEffect } from "react";
import type { SystemMetrics } from "../lib/types";
import { getSystemMetrics } from "../lib/tauri";

const POLL_INTERVAL = 2000;

export function useSystemMetrics() {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    cpu_usage_percent: 0,
    memory_used_mb: 0,
    memory_total_mb: 0,
    gpu_usage_percent: null,
    gpu_memory_used_mb: null,
    gpu_memory_total_mb: null,
  });

  useEffect(() => {
    const fetch = async () => {
      try {
        const m = await getSystemMetrics();
        setMetrics(m);
      } catch {
        // Backend not ready
      }
    };

    fetch();
    const interval = setInterval(fetch, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, []);

  return metrics;
}
