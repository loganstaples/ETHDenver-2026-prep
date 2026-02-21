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
  /** Which worker party should cheat (0-indexed). Only used when simulate_cheater is true. */
  cheater_party?: number;
  /** At which training step the cheater corrupts weights. Only used when simulate_cheater is true. */
  cheater_step?: number;
  seed: number;
  model_name?: string;
  model_slug?: string;
  /** Pre-registered on-chain job ID (user wallet already paid) */
  job_id?: number;
  /** On-chain model token ID for weight caching */
  model_token_id?: number;
  /** Trusted node addresses — only these workers handle plaintext weights */
  trusted_nodes?: string[];
}

export interface CheaterInfo {
  party_index: number;
  step: number;
  slashed: boolean;
  slash_tx_hash: string | null;
  recovered: boolean;
  recovery_workers: number | null;
  /** Step training resumed from after checkpoint rollback */
  resumed_from_step: number | null;
}

export interface TrainingSessionState {
  session_id: string;
  status: 'starting' | 'running' | 'paused' | 'stopped' | 'complete' | 'failed';
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
  sub_step: string | null;
  checkpoint_freq: number;
  coordinator_address: string;
  job_id: number;
  elapsed_secs: number;
  started_at: number;
  workers_active?: number;
  model_name?: string;
  model_slug?: string;
  cost_gas_spent_wei?: number;
  cost_worker_fees_adi?: number;
  cost_deposit_adi?: number;
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
  accuracy?: number;
};

export interface UploadedData {
  samples: number;
  inputDim: number;
  outputDim: number;
}

export interface UploadedWeights {
  totalParams: number;
}

export interface ZeroGStorageResult {
  rootHash: string;
  txHash: string;
  explorerUrl: string;
}

export interface UseMpcTrainingReturn {
  startTraining: (config: TrainingJobConfig) => Promise<void>;
  uploadData: (file: File) => Promise<void>;
  uploadWeights: (file: File) => Promise<void>;
  downloadModel: (sessionId: string) => Promise<void>;
  storeOnZeroG: (sessionId: string, version?: string) => Promise<void>;
  sendCommand: (command: Record<string, unknown>) => void;
  session: TrainingSessionState | null;
  losses: LossDataPoint[];
  events: TrainingEvent[];
  isConnected: boolean;
  isStarting: boolean;
  error: string | null;
  uploadedData: UploadedData | null;
  uploadedWeights: UploadedWeights | null;
  workersOnline: number;
  zeroGResult: ZeroGStorageResult | null;
  isStoringOnZeroG: boolean;
  elapsedTime: number;
  history: TrainingSessionState[];
  isLoadingHistory: boolean;
  fetchHistory: () => Promise<void>;
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
  const [uploadedData, setUploadedData] = useState<UploadedData | null>(null);
  const [uploadedWeights, setUploadedWeights] = useState<UploadedWeights | null>(null);
  const [workersOnline, setWorkersOnline] = useState(0);
  const [zeroGResult, setZeroGResult] = useState<ZeroGStorageResult | null>(null);
  const [isStoringOnZeroG, setIsStoringOnZeroG] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [history, setHistory] = useState<TrainingSessionState[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ========================================================================
  // Fetch training history from server
  // ========================================================================

  const fetchHistory = useCallback(async () => {
    setIsLoadingHistory(true);
    try {
      const res = await fetch(`${API_BASE}/api/training/sessions`);
      if (res.ok) {
        const sessions: TrainingSessionState[] = await res.json();
        setHistory(sessions.filter(s => s.status === 'complete' || s.status === 'failed' || s.status === 'stopped'));
      }
    } catch {
      // Silently fail
    } finally {
      setIsLoadingHistory(false);
    }
  }, []);

  // Fetch history on mount
  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  // ========================================================================
  // Live elapsed time counter
  // ========================================================================

  useEffect(() => {
    if (!session || session.status === 'complete' || session.status === 'failed') {
      return;
    }
    const interval = setInterval(() => {
      if (session.started_at > 0) {
        setElapsedTime(Math.floor(Date.now() / 1000 - session.started_at));
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [session?.status, session?.started_at]);

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
          const data = JSON.parse(event.data);

          // Handle full session state snapshot (sent on subscribe/reconnect)
          if (data.type === 'session_state' && data.state) {
            const s = data.state as TrainingSessionState;
            setSession(s);
            if (s.losses && s.losses.length > 0) {
              setLosses(s.losses.map((loss: number, i: number) => ({
                step: i + 1,
                loss,
                // Use real evaluated accuracy on last point if session has it
                accuracy: (i === s.losses.length - 1 && s.accuracy != null)
                  ? s.accuracy
                  : Math.max(0, Math.min(1, Math.exp(-loss))),
              })));
            }
            return;
          }

          if (!data.event) return;

          setEvents((prev) => [...prev, data as TrainingEvent].slice(-200));

          const evt = data.event;

          // Handle training_step events
          if (evt.type === 'training_step') {
            const step = evt.step as number;
            const loss = evt.loss as number;
            const accuracy = evt.accuracy as number | undefined;
            const macOk = evt.mac_ok as boolean;

            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                current_step: step,
                current_loss: loss,
                accuracy: accuracy ?? prev.accuracy,
                phase_description: `Training step ${step}/${prev.total_steps} — loss: ${loss.toFixed(4)}`,
                mac_checks_passed: macOk ? prev.mac_checks_passed + 1 : prev.mac_checks_passed,
              };
            });

            setLosses((prev) => [...prev, { step, loss, accuracy: accuracy ?? undefined }]);
          }

          // Handle sub-step events (real-time operation status)
          if (evt.type === 'sub_step') {
            const operation = evt.operation as string;
            // Detect checkpoint completion events from the MPC training loop
            const isCheckpointComplete = operation.startsWith('Checkpoint ') && operation.includes('verified');
            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                sub_step: operation,
                checkpoints_submitted: isCheckpointComplete
                  ? prev.checkpoints_submitted + 1
                  : prev.checkpoints_submitted,
              };
            });
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

          // Handle cheater detection — set cheater state for CheaterToast overlay.
          // Training continues uninterrupted; the loss curve keeps a red reference line.
          if (evt.type === 'cheater_detected') {
            const cheaterStep = evt.step as number;
            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                cheater_detected: {
                  party_index: evt.party_index as number,
                  step: cheaterStep,
                  slashed: false,
                  slash_tx_hash: null,
                  recovered: false,
                  recovery_workers: null,
                  resumed_from_step: null,
                },
              };
            });
          }

          // Handle cheater slashed on-chain
          if (evt.type === 'cheater_slashed') {
            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                cheater_detected: prev.cheater_detected
                  ? {
                      ...prev.cheater_detected,
                      slashed: true,
                      slash_tx_hash: (evt.tx_hash as string) || null,
                    }
                  : prev.cheater_detected,
              };
            });
          }

          // Handle recovery after cheater removal
          if (evt.type === 'recovery_completed') {
            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                workers_active: (evt.honest_workers as number) ?? prev.workers_active,
                cheater_detected: prev.cheater_detected
                  ? {
                      ...prev.cheater_detected,
                      recovered: true,
                      recovery_workers: (evt.honest_workers as number) ?? null,
                      resumed_from_step: (evt.resumed_from_step as number) ?? null,
                    }
                  : prev.cheater_detected,
              };
            });
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
            // Update last loss entry with real evaluated accuracy so chart/headline match
            if (accuracy !== undefined) {
              setLosses((prev) => {
                if (prev.length === 0) return prev;
                const updated = [...prev];
                updated[updated.length - 1] = { ...updated[updated.length - 1], accuracy };
                return updated;
              });
            }
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
            // Update last loss entry with real evaluated accuracy so chart/headline match
            if (accuracy !== undefined) {
              setLosses((prev) => {
                if (prev.length === 0) return prev;
                const updated = [...prev];
                updated[updated.length - 1] = { ...updated[updated.length - 1], accuracy };
                return updated;
              });
            }
            fetchHistory();
          }

          // Handle worker count updates
          if (evt.type === 'workers_updated') {
            const count = evt.count as number | undefined;
            if (count !== undefined) {
              setWorkersOnline(count);
            }
          }

          // Handle session failure
          if (evt.type === 'session_failed') {
            const reason = (evt.error ?? evt.reason) as string | undefined;
            setSession((prev) => {
              if (!prev) return prev;
              return { ...prev, status: 'failed' };
            });
            setError(reason || 'Training session failed');
            fetchHistory();
          }

          // Handle training paused
          if (evt.type === 'training_paused') {
            const step = evt.step as number | undefined;
            setSession((prev) => {
              if (!prev) return prev;
              return { ...prev, status: 'paused', current_step: step ?? prev.current_step };
            });
          }

          // Handle training stopped
          if (evt.type === 'training_stopped') {
            setSession((prev) => {
              if (!prev) return prev;
              return { ...prev, status: 'stopped' };
            });
            fetchHistory();
          }

          // Handle cost updates
          if (evt.type === 'cost_update') {
            setSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                cost_gas_spent_wei: (evt.gas_spent_wei as number) ?? 0,
                cost_worker_fees_adi: (evt.worker_fees_accrued as number) ?? 0,
                cost_deposit_adi: (evt.deposit_amount as number) ?? 0,
              };
            });
          }
        } catch {
          // Ignore malformed messages
        }
      };
    } catch {
      setIsConnected(false);
    }
  }, [fetchHistory]);

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
          const raw = await res.json();
          // Normalize cheater_detected from REST (which lacks slashed/recovered fields)
          const data: TrainingSessionState = {
            ...raw,
            cheater_detected: raw.cheater_detected
              ? {
                  party_index: raw.cheater_detected.party_index,
                  step: raw.cheater_detected.step,
                  slashed: raw.cheater_detected.slashed ?? false,
                  slash_tx_hash: raw.cheater_detected.slash_tx_hash ?? null,
                  recovered: raw.cheater_detected.recovered ?? false,
                  recovery_workers: raw.cheater_detected.recovery_workers ?? null,
                  resumed_from_step: raw.cheater_detected.resumed_from_step ?? null,
                }
              : null,
          };

          // Merge: only apply poll data if it's newer than what WS set.
          // Preserve richer cheater info from WS events when poll data is stale.
          setSession((prev) => {
            if (!prev) return data;
            if (data.current_step > prev.current_step || data.status === 'complete' || data.status === 'failed') {
              // Keep WS-enriched cheater info if poll doesn't have it
              const cheater = prev.cheater_detected && (!data.cheater_detected?.slashed && prev.cheater_detected.slashed)
                ? prev.cheater_detected
                : data.cheater_detected;
              return { ...data, cheater_detected: cheater, checkpoint_freq: prev.checkpoint_freq || data.checkpoint_freq };
            }
            return {
              ...prev,
              coordinator_address: data.coordinator_address || prev.coordinator_address,
              job_id: data.job_id || prev.job_id,
              workers_active: data.workers_active ?? prev.workers_active,
              status: prev.status,
            };
          });

          // Only update losses if poll has more data points than WS
          setLosses((prev) => {
            if (data.losses && data.losses.length > prev.length) {
              return data.losses.map((loss, i) => ({
                step: i + 1,
                loss,
                // Use real evaluated accuracy on last point if session has it
                accuracy: (i === data.losses.length - 1 && data.accuracy != null)
                  ? data.accuracy
                  : Math.max(0, Math.min(1, Math.exp(-loss))),
              }));
            }
            return prev;
          });

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
        sub_step: null,
        checkpoint_freq: config.checkpoint_freq,
        coordinator_address: '',
        job_id: 0,
        elapsed_secs: 0,
        started_at: Date.now() / 1000,
        workers_active: config.num_workers ?? 3,
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
  // Recover active session on mount (so navigating back to /train reconnects)
  // ========================================================================

  const recoveredRef = useRef(false);

  useEffect(() => {
    if (recoveredRef.current || sessionIdRef.current) return;
    recoveredRef.current = true;

    const recover = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/training/sessions`);
        if (!res.ok) return;
        const sessions: TrainingSessionState[] = await res.json();
        const active = sessions.find(
          (s) => s.status === 'starting' || s.status === 'running' || s.status === 'paused',
        );
        if (!active) return;

        // Restore session state
        sessionIdRef.current = active.session_id;
        setSession(active);
        if (active.losses && active.losses.length > 0) {
          setLosses(
            active.losses.map((loss: number, i: number) => ({
              step: i + 1,
              loss,
              accuracy:
                i === active.losses.length - 1 && active.accuracy != null
                  ? active.accuracy
                  : Math.max(0, Math.min(1, Math.exp(-loss))),
            })),
          );
        }
        if (active.started_at > 0) {
          setElapsedTime(Math.floor(Date.now() / 1000 - active.started_at));
        }

        // Reconnect WebSocket + polling
        connectWebSocket(active.session_id);
        startPolling(active.session_id);
      } catch {
        // Non-fatal — user can still start a new session
      }
    };

    recover();
  }, [connectWebSocket, startPolling]);

  // ========================================================================
  // Data Upload
  // ========================================================================

  const uploadData = useCallback(async (file: File) => {
    try {
      setError(null);
      const text = await file.text();
      const data = JSON.parse(text);

      const res = await fetch(`${API_BASE}/api/training/data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.message || errBody.error || `HTTP ${res.status}`);
      }

      const result = await res.json();
      setUploadedData({
        samples: result.samples ?? (Array.isArray(data) ? data.length : 0),
        inputDim: result.input_dim ?? 0,
        outputDim: result.output_dim ?? 0,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to upload data';
      setError(message);
      throw err;
    }
  }, []);

  // ========================================================================
  // Weights Upload
  // ========================================================================

  const uploadWeights = useCallback(async (file: File) => {
    try {
      setError(null);
      const text = await file.text();
      const data = JSON.parse(text);

      const res = await fetch(`${API_BASE}/api/training/weights`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.message || errBody.error || `HTTP ${res.status}`);
      }

      const result = await res.json();
      const totalParams = result.total_params
        ?? ((result.w1_size ?? 0) + (result.b1_size ?? 0) + (result.w2_size ?? 0) + (result.b2_size ?? 0));
      setUploadedWeights({
        totalParams,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to upload weights';
      setError(message);
      throw err;
    }
  }, []);

  // ========================================================================
  // Model Download
  // ========================================================================

  const downloadModel = useCallback(async (sessionId: string) => {
    try {
      setError(null);
      const res = await fetch(`${API_BASE}/api/training/sessions/${sessionId}/model`);

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.message || errBody.error || `HTTP ${res.status}`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const sessionSlice = sessionId.slice(0, 8);
      a.download = `helix-model-${sessionSlice}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to download model';
      setError(message);
    }
  }, []);

  // ========================================================================
  // Store on 0G Storage
  // ========================================================================

  const storeOnZeroG = useCallback(async (sessionId: string, version?: string) => {
    try {
      setIsStoringOnZeroG(true);
      setError(null);

      // Fetch model weights from the Helix backend
      const modelRes = await fetch(`${API_BASE}/api/training/sessions/${sessionId}/model`);
      if (!modelRes.ok) {
        const errBody = await modelRes.json().catch(() => ({}));
        throw new Error(errBody.message || errBody.error || `HTTP ${modelRes.status}`);
      }
      const modelData = await modelRes.json();

      // Upload to 0G Storage via our Next.js API route
      // Model endpoint returns raw weights {w1, b1, w2, b2}
      const res = await fetch('/api/store-on-0g', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          weights: modelData,
          version,
        }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || `HTTP ${res.status}`);
      }

      const result = await res.json();
      setZeroGResult({
        rootHash: result.root_hash,
        txHash: result.tx_hash,
        explorerUrl: result.explorer_url,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to store on 0G';
      setError(message);
    } finally {
      setIsStoringOnZeroG(false);
    }
  }, []);

  // ========================================================================
  // Worker count polling
  // ========================================================================

  useEffect(() => {
    const fetchWorkers = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/workers`);
        if (res.ok) {
          const data = await res.json();
          const arr: { status?: string }[] = Array.isArray(data) ? data : (data.workers ?? []);
          setWorkersOnline(arr.filter(w => w.status !== 'offline').length);
        }
      } catch {
        // Non-fatal
      }
    };

    fetchWorkers();
    const interval = setInterval(fetchWorkers, 5000);
    return () => clearInterval(interval);
  }, []);

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

  // ========================================================================
  // Send command via WebSocket
  // ========================================================================

  const sendCommand = useCallback((command: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(command));
    }
  }, []);

  return {
    startTraining,
    uploadData,
    uploadWeights,
    downloadModel,
    storeOnZeroG,
    sendCommand,
    session,
    losses,
    events,
    isConnected,
    isStarting,
    error,
    uploadedData,
    uploadedWeights,
    workersOnline,
    zeroGResult,
    isStoringOnZeroG,
    elapsedTime,
    history,
    isLoadingHistory,
    fetchHistory,
  };
}

export default useMpcTraining;
