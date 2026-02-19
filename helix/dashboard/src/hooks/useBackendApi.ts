'use client';

/**
 * HELIX Backend API Hook
 * Unified polling-based client for the Rust backend REST endpoints.
 * Provides typed data for health, workers, round status, events,
 * plus on-demand helpers for inference events, worker reputation, and losses.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

// ============================================================================
// Types (matching Rust backend response shapes)
// ============================================================================

export interface BackendWorker {
  id: string;
  status: string;
  rounds_completed: number;
  rounds_participated: number;
  cpu_load: number;
  memory_mb: number;
  reputation_score: number; // 0.0-1.0
  success_rate: number;
  earnings_wei: number;
  last_heartbeat: number;
  capabilities: {
    can_train: boolean;
    can_prove: boolean;
    can_aggregate: boolean;
    gpu_model: string | null;
    max_batch_size: number;
  };
}

export interface BackendHealth {
  status: string;
  role: string;
  workers: number;
  current_round: number | null;
  completed_rounds: number;
  connected_peers: number;
  uptime_secs: number;
  mpc: {
    enabled: boolean;
    active_session: boolean;
    session_id: string | null;
    num_parties: number;
    party_index: number | null;
  };
  memory_bytes: number;
  fault_tolerance: {
    healthy_workers: number;
    degraded_workers: number;
    failed_workers: number;
    system_healthy: boolean;
  };
}

export interface BackendPeer {
  id: string;
  address: string;
  last_seen: number;
  reputation: number;
}

export interface BackendRoundStatus {
  worker_count: number;
  available_workers: number;
  computing_workers: number;
  current_round: {
    round_id: number;
    phase: string;
    gradients_received: number;
    workers_assigned: number;
    commitment_hash: string | null;
  } | null;
  completed_rounds: number;
  workers: BackendWorker[];
}

export interface BackendEvent {
  type: string;
  timestamp?: number;
  session_id?: string;
  model_owner?: string;
  model_token_id?: number;
  requester?: string;
  prediction?: number;
  confidence?: number;
  fee?: number;
  latency?: number;
  workers?: number;
  worker_ids?: string[];
  [key: string]: unknown;
}

export interface WorkerReputation {
  worker_id: string;
  overall_score: number;
  success_rate: number;
  responsiveness: number;
  validity: number;
  bandwidth: number;
  uptime: number;
  total_interactions: number;
  rounds_participated: number;
  rounds_succeeded: number;
  is_banned: boolean;
}

// ============================================================================
// Hook options & return types
// ============================================================================

export interface UseBackendApiOptions {
  refreshInterval?: number; // default 5000ms
  enabled?: boolean; // default true
}

export interface UseBackendApiReturn {
  health: BackendHealth | null;
  workers: BackendWorker[];
  peers: BackendPeer[];
  roundStatus: BackendRoundStatus | null;
  events: BackendEvent[];
  isLoading: boolean;
  isConnected: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  fetchInferenceEvents: (ownerAddress: string) => Promise<BackendEvent[]>;
  fetchWorkerReputation: (workerId: string) => Promise<WorkerReputation | null>;
  fetchLosses: (sessionId: string) => Promise<{
    losses: number[];
    current_step: number;
    total_steps: number;
  } | null>;
}

// ============================================================================
// Constants
// ============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';
const FETCH_TIMEOUT_MS = 10_000;
const DEFAULT_REFRESH_INTERVAL = 5_000;

// ============================================================================
// Helpers
// ============================================================================

async function safeFetch<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useBackendApi(options?: UseBackendApiOptions): UseBackendApiReturn {
  const {
    refreshInterval = DEFAULT_REFRESH_INTERVAL,
    enabled = true,
  } = options ?? {};

  // State
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [workers, setWorkers] = useState<BackendWorker[]>([]);
  const [peers, setPeers] = useState<BackendPeer[]>([]);
  const [roundStatus, setRoundStatus] = useState<BackendRoundStatus | null>(null);
  const [events, setEvents] = useState<BackendEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Refs for cleanup
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);

  // ===========================================================================
  // Core polling fetch
  // ===========================================================================

  const fetchAll = useCallback(async () => {
    if (!mountedRef.current) return;

    try {
      const [healthData, workersData, roundData, eventsData] = await Promise.all([
        safeFetch<BackendHealth>(`${API_BASE}/health`),
        safeFetch<BackendWorker[] | { workers: BackendWorker[]; peers?: BackendPeer[] }>(
          `${API_BASE}/api/workers`,
        ),
        safeFetch<BackendRoundStatus>(`${API_BASE}/round/status`),
        safeFetch<BackendEvent[] | { events: BackendEvent[] }>(
          `${API_BASE}/api/events`,
        ),
      ]);

      if (!mountedRef.current) return;

      // Health
      if (healthData) {
        setHealth(healthData);
        setIsConnected(true);
        setError(null);
      } else {
        setIsConnected(false);
        setError('Backend unreachable');
      }

      // Workers & peers
      if (workersData) {
        if (Array.isArray(workersData)) {
          setWorkers(workersData);
        } else {
          setWorkers(workersData.workers ?? []);
          if (workersData.peers) {
            setPeers(workersData.peers);
          }
        }
      }

      // Round status
      if (roundData) {
        setRoundStatus(roundData);
      }

      // Events
      if (eventsData) {
        if (Array.isArray(eventsData)) {
          setEvents(eventsData);
        } else {
          setEvents(eventsData.events ?? []);
        }
      }
    } catch (err) {
      if (!mountedRef.current) return;
      setIsConnected(false);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  }, []);

  // ===========================================================================
  // Manual refresh
  // ===========================================================================

  const refresh = useCallback(async () => {
    setIsLoading(true);
    await fetchAll();
  }, [fetchAll]);

  // ===========================================================================
  // On-demand helpers (not polled)
  // ===========================================================================

  const fetchInferenceEvents = useCallback(
    async (ownerAddress: string): Promise<BackendEvent[]> => {
      const data = await safeFetch<BackendEvent[] | { events: BackendEvent[] }>(
        `${API_BASE}/api/events?owner=${encodeURIComponent(ownerAddress)}`,
      );
      if (!data) return [];
      return Array.isArray(data) ? data : (data.events ?? []);
    },
    [],
  );

  const fetchWorkerReputation = useCallback(
    async (workerId: string): Promise<WorkerReputation | null> => {
      return safeFetch<WorkerReputation>(
        `${API_BASE}/api/workers/${encodeURIComponent(workerId)}/reputation`,
      );
    },
    [],
  );

  const fetchLosses = useCallback(
    async (
      sessionId: string,
    ): Promise<{ losses: number[]; current_step: number; total_steps: number } | null> => {
      return safeFetch<{ losses: number[]; current_step: number; total_steps: number }>(
        `${API_BASE}/api/training/sessions/${encodeURIComponent(sessionId)}/losses`,
      );
    },
    [],
  );

  // ===========================================================================
  // Effects
  // ===========================================================================

  useEffect(() => {
    mountedRef.current = true;

    if (!enabled) {
      return () => {
        mountedRef.current = false;
      };
    }

    // Initial fetch
    fetchAll();

    // Polling interval
    intervalRef.current = setInterval(fetchAll, refreshInterval);

    return () => {
      mountedRef.current = false;
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [enabled, refreshInterval, fetchAll]);

  // ===========================================================================
  // Return
  // ===========================================================================

  return {
    health,
    workers,
    peers,
    roundStatus,
    events,
    isLoading,
    isConnected,
    error,
    refresh,
    fetchInferenceEvents,
    fetchWorkerReputation,
    fetchLosses,
  };
}

export default useBackendApi;
