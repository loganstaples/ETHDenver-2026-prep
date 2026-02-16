'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

// ============================================================================
// Types
// ============================================================================

export interface TrainingJobConfig {
  architecture: number[];
  num_workers: number;
  num_steps: number;
  learning_rate: number;
  checkpoint_freq: number;
  mac_interval: number;
  zk_mode: 'off' | 'always' | 'risk';
  zk_checkpoint_freq: number;
  min_workers_for_mpc: number;
  train_size: number;
  test_size: number;
  use_real_mnist: boolean;
  payment_eth: number;
  stake_per_worker_eth: number;
  simulate_cheater: boolean;
  seed: number;
}

export interface CheaterInfo {
  party_index: number;
  step: number;
}

export interface TrainingSessionState {
  session_id: string;
  status: 'starting' | 'running' | 'complete' | 'failed';
  current_step: number;
  total_steps: number;
  current_loss: number;
  losses: number[];
  accuracy: number | null;
  checkpoints_submitted: number;
  mac_checks_passed: number;
  cheater_detected: CheaterInfo | null;
  zk_proofs_generated: number;
  zk_activated_by_risk: boolean;
  phase: number;
  phase_description: string;
  coordinator_address: string;
  job_id: number;
  elapsed_secs: number;
  started_at: number;
}

export interface TrainingEvent {
  session_id: string;
  event: {
    type: string;
    [key: string]: unknown;
  };
}

export type LossDataPoint = {
  step: number;
  loss: number;
};

export interface UseMpcTrainingReturn {
  startTraining: (config: TrainingJobConfig) => Promise<void>;
  session: TrainingSessionState | null;
  losses: LossDataPoint[];
  events: TrainingEvent[];
  isConnected: boolean;
  isStarting: boolean;
  error: string | null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001/ws';

// ============================================================================
// Hook Implementation
// ============================================================================

export function useMpcTraining(): UseMpcTrainingReturn {
  const [session, setSession] = useState<TrainingSessionState | null>(null);
  const [losses, setLosses] = useState<LossDataPoint[]>([]);
  const [events, setEvents] = useState<TrainingEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ========================================================================
  // WebSocket connection
  // ========================================================================

  const connectWebSocket = useCallback((sessionId: string) => {
    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        // Subscribe to this session's events
        ws.send(JSON.stringify({ subscribe: `training:${sessionId}` }));
      };

      ws.onclose = () => {
        setIsConnected(false);
        // Attempt reconnect if session is still active
        if (sessionIdRef.current) {
          reconnectTimeoutRef.current = setTimeout(() => {
            if (sessionIdRef.current) {
              connectWebSocket(sessionIdRef.current);
            }
          }, 2000);
        }
      };

      ws.onerror = () => {
        setIsConnected(false);
      };

      ws.onmessage = (event) => {
        try {
          const data: TrainingEvent = JSON.parse(event.data);
          if (!data.event) return;

          setEvents((prev) => [...prev, data].slice(-200));

          const evt = data.event;

          // Handle training_step events
          if (evt.type === 'training_step') {
            const step = evt.step as number;
            const loss = evt.loss as number;
            const macOk = evt.mac_ok as boolean;

            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                current_step: step,
                current_loss: loss,
                mac_checks_passed: macOk ? prev.mac_checks_passed + 1 : prev.mac_checks_passed,
              };
            });

            setLosses((prev) => [...prev, { step, loss }]);
          }

          // Handle phase events
          if (evt.type === 'phase_started' || evt.type === 'phase_completed') {
            const phase = evt.phase as number | undefined;
            const description = evt.description as string | undefined;

            if (phase !== undefined) {
              setSession((prev) => {
                if (!prev) return prev;
                return {
                  ...prev,
                  phase,
                  phase_description: description || prev.phase_description,
                  status: 'running',
                };
              });
            }
          }

          // Handle checkpoint events
          if (evt.type === 'checkpoint_submitted') {
            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                checkpoints_submitted: prev.checkpoints_submitted + 1,
              };
            });
          }

          // Handle cheater detection
          if (evt.type === 'cheater_detected') {
            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                cheater_detected: {
                  party_index: evt.party_index as number,
                  step: evt.step as number,
                },
              };
            });
          }

          // Handle cheater slashed
          if (evt.type === 'cheater_slashed') {
            // Already captured in cheater_detected
          }

          // Handle training complete
          if (evt.type === 'training_complete') {
            const accuracy = evt.accuracy as number | undefined;
            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                status: 'running',
                accuracy: accuracy ?? prev.accuracy,
              };
            });
          }

          // Handle ZK proof events
          if (evt.type === 'zk_proof_started') {
            setSession((prev) => {
              if (!prev) return prev;
              return { ...prev, zk_activated_by_risk: true };
            });
          }

          if (evt.type === 'zk_proof_generated') {
            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                zk_proofs_generated: prev.zk_proofs_generated + 1,
              };
            });
          }

          // Handle session complete
          if (evt.type === 'session_complete') {
            const accuracy = evt.accuracy as number | undefined;
            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                status: 'complete',
                accuracy: accuracy ?? prev.accuracy,
              };
            });
          }

          // Handle session failure
          if (evt.type === 'session_failed') {
            const reason = evt.reason as string | undefined;
            setSession((prev) => {
              if (!prev) return prev;
              return { ...prev, status: 'failed' };
            });
            setError(reason || 'Training session failed');
          }
        } catch {
          // Ignore malformed messages
        }
      };
    } catch {
      setIsConnected(false);
    }
  }, []);

  // ========================================================================
  // Polling fallback (supplements WebSocket)
  // ========================================================================

  const startPolling = useCallback((sessionId: string) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/training/sessions/${sessionId}`);
        if (res.ok) {
          const data: TrainingSessionState = await res.json();
          setSession(data);

          // Update losses from server state
          if (data.losses && data.losses.length > 0) {
            setLosses(data.losses.map((loss, i) => ({ step: i + 1, loss })));
          }

          // Stop polling if terminal state
          if (data.status === 'complete' || data.status === 'failed') {
            if (pollIntervalRef.current) {
              clearInterval(pollIntervalRef.current);
              pollIntervalRef.current = null;
            }
          }
        }
      } catch {
        // Polling failure is non-fatal — WebSocket is primary
      }
    };

    // Poll every 3 seconds as a fallback
    pollIntervalRef.current = setInterval(poll, 3000);
    // Also poll immediately
    poll();
  }, []);

  // ========================================================================
  // Start training
  // ========================================================================

  const startTraining = useCallback(async (config: TrainingJobConfig) => {
    setIsStarting(true);
    setError(null);
    setLosses([]);
    setEvents([]);
    setSession(null);

    try {
      const res = await fetch(`${API_BASE}/api/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.message || errBody.error || `HTTP ${res.status}`);
      }

      const data: { session_id: string; status: string } = await res.json();
      const sessionId = data.session_id;
      sessionIdRef.current = sessionId;

      // Initialize local session state
      setSession({
        session_id: sessionId,
        status: 'starting',
        current_step: 0,
        total_steps: config.num_steps,
        current_loss: 0,
        losses: [],
        accuracy: null,
        checkpoints_submitted: 0,
        mac_checks_passed: 0,
        cheater_detected: null,
        zk_proofs_generated: 0,
        zk_activated_by_risk: false,
        phase: 1,
        phase_description: 'Starting session...',
        coordinator_address: '',
        job_id: 0,
        elapsed_secs: 0,
        started_at: Date.now() / 1000,
      });

      // Connect WebSocket and start polling
      connectWebSocket(sessionId);
      startPolling(sessionId);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start training';
      setError(message);
    } finally {
      setIsStarting(false);
    }
  }, [connectWebSocket, startPolling]);

  // ========================================================================
  // Cleanup
  // ========================================================================

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      sessionIdRef.current = null;
    };
  }, []);

  return {
    startTraining,
    session,
    losses,
    events,
    isConnected,
    isStarting,
    error,
  };
}

export default useMpcTraining;
