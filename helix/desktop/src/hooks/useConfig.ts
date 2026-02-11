import { useState, useEffect, useCallback } from "react";
import type { NodeConfig } from "../lib/types";
import { getConfig, setConfig } from "../lib/tauri";

export function useConfig() {
  const [config, setConfigState] = useState<NodeConfig | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const c = await getConfig();
        setConfigState(c);
      } catch {
        // Backend not ready
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  const updateConfig = useCallback(async (newConfig: NodeConfig) => {
    await setConfig(newConfig);
    setConfigState(newConfig);
  }, []);

  return { config, loading, updateConfig };
}
