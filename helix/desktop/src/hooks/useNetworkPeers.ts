import { useState, useEffect } from "react";
import type { PeerInfo } from "../lib/types";
import { getPeers } from "../lib/tauri";

const POLL_INTERVAL = 3000;

export function useNetworkPeers() {
  const [peers, setPeers] = useState<PeerInfo[]>([]);

  useEffect(() => {
    const fetch = async () => {
      try {
        const p = await getPeers();
        setPeers(p);
      } catch {
        // Backend not ready
      }
    };

    fetch();
    const interval = setInterval(fetch, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, []);

  return peers;
}
