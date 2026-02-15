import { useState, useEffect, useCallback } from "react";
import type { NodeConfig } from "../lib/types";
import { getConfig, setConfig } from "../lib/tauri";

const DEFAULT_CONFIG: NodeConfig = {
  listen_address: "0.0.0.0:9944",
  aggregator_address: "127.0.0.1:9945",
  rpc_port: 9933,
  cpu_threads: 4,
  gpu_memory_limit_mb: 4096,
  max_concurrent_tasks: 2,
  local_epochs: 1,
  batch_size: 32,
  generate_proofs: true,
  max_error_bound: 0.01,
  use_tls: false,
};

export function useConfig() {
  const [config, setConfigState] = useState<NodeConfig>(DEFAULT_CONFIG);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const c = await getConfig();
        setConfigState(c);
      } catch {
        // Backend not ready — use defaults
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  const updateConfig = useCallback(async (newConfig: NodeConfig) => {
    try {
      await setConfig(newConfig);
    } catch {
      // Backend not available
    }
    setConfigState(newConfig);
  }, []);

  return { config, loading, updateConfig };
}
