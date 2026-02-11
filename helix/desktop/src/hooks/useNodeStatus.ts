import { useState, useEffect, useCallback } from "react";
import type { NodeStatus } from "../lib/types";
import { getNodeStatus, startNode, stopNode } from "../lib/tauri";

const POLL_INTERVAL = 2000;

export function useNodeStatus() {
  const [status, setStatus] = useState<NodeStatus>({
    running: false,
    role: "Compute",
    uptime_secs: 0,
    peer_id: "0x0000...0000",
    connected_peers: 0,
  });
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const s = await getNodeStatus();
      setStatus(s);
    } catch {
      // Backend not ready yet
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [refresh]);

  const start = useCallback(async () => {
    await startNode();
    await refresh();
  }, [refresh]);

  const stop = useCallback(async () => {
    await stopNode();
    await refresh();
  }, [refresh]);

  return { status, loading, start, stop, refresh };
}
