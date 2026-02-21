'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import type { TrainingSessionState, TrainingEvent, LossDataPoint } from './useMpcTraining';

export type { TrainingSessionState, TrainingEvent, LossDataPoint };

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001/ws';

export interface UseDashboardSessionsReturn {
  sessions: TrainingSessionState[];
  activeSession: TrainingSessionState | null;
  losses: LossDataPoint[];
  events: TrainingEvent[];
  isLoading: boolean;
  selectSession: (id: string) => void;
}

export function useDashboardSessions(): UseDashboardSessionsReturn {
  const [sessions, setSessions] = useState<TrainingSessionState[]>([]);
  const [activeSession, setActiveSession] = useState<TrainingSessionState | null>(null);
  const [losses, setLosses] = useState<LossDataPoint[]>([]);
  const [events, setEvents] = useState<TrainingEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const sessionsPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const activePollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Fetch all sessions
  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/training/sessions`);
      if (!res.ok) return;
      const data: TrainingSessionState[] = await res.json();
      setSessions(data);

      // Auto-select first non-complete session if nothing selected
      if (!selectedId) {
        const active = data.find((s) => s.status !== 'complete' && s.status !== 'failed');
        if (active) {
          setSelectedId(active.session_id);
        } else if (data.length > 0) {
          setSelectedId(data[0].session_id);
        }
      }
    } catch {
      // Non-fatal
    } finally {
      setIsLoading(false);
    }
  }, [selectedId]);

  // Fetch active session detail
  const fetchActiveSession = useCallback(async (sessionId: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/training/sessions/${sessionId}`);
      if (!res.ok) return;
      const data: TrainingSessionState = await res.json();

      setActiveSession((prev) => {
        if (!prev) return data;
        if (data.current_step > prev.current_step || data.status === 'complete' || data.status === 'failed') {
          return data;
        }
        return {
          ...prev,
          coordinator_address: data.coordinator_address || prev.coordinator_address,
          job_id: data.job_id || prev.job_id,
          workers_active: data.workers_active ?? prev.workers_active,
          status: prev.status,
        };
      });

      setLosses((prev) => {
        if (data.losses && data.losses.length > prev.length) {
          return data.losses.map((loss: number, i: number) => ({
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
    } catch {
      // Non-fatal — WebSocket is primary
    }
  }, []);

  // WebSocket connection
  const connectWebSocket = useCallback((sessionId: string) => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ subscribe: `training:${sessionId}` }));
      };

      ws.onclose = () => {
        if (selectedId) {
          reconnectRef.current = setTimeout(() => {
            if (selectedId) connectWebSocket(selectedId);
          }, 2000);
        }
      };

      ws.onerror = () => {};

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Full session snapshot
          if (data.type === 'session_state' && data.state) {
            const s = data.state as TrainingSessionState;
            setActiveSession(s);
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

          const stamped = { ...data, receivedAt: Date.now() } as TrainingEvent & { receivedAt: number };
          setEvents((prev) => [...prev, stamped].slice(-200));

          const evt = data.event;

          if (evt.type === 'training_step') {
            setActiveSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                current_step: evt.step as number,
                current_loss: evt.loss as number,
                mac_checks_passed: (evt.mac_ok as boolean)
                  ? prev.mac_checks_passed + 1
                  : prev.mac_checks_passed,
              };
            });
            const loss = evt.loss as number;
            const accuracy = (evt.accuracy as number | undefined)
              ?? Math.max(0, Math.min(1, Math.exp(-loss)));
            setLosses((prev) => [...prev, { step: evt.step as number, loss, accuracy }]);
          }

          if (evt.type === 'phase_started' || evt.type === 'phase_completed') {
            const phase = evt.phase as number | undefined;
            const description = evt.description as string | undefined;
            if (phase !== undefined) {
              setActiveSession((prev) => {
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

          if (evt.type === 'checkpoint_submitted') {
            setActiveSession((prev) => {
              if (!prev) return prev;
              return { ...prev, checkpoints_submitted: prev.checkpoints_submitted + 1 };
            });
          }

          if (evt.type === 'cheater_detected') {
            setActiveSession((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                cheater_detected: {
                  party_index: evt.party_index as number,
                  step: evt.step as number,
                  slashed: false,
                  slash_tx_hash: null,
                  recovered: false,
                  recovery_workers: null,
                  resumed_from_step: null,
                },
              };
            });
          }

          if (evt.type === 'cheater_slashed') {
            setActiveSession((prev) => {
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

          if (evt.type === 'recovery_completed') {
            setActiveSession((prev) => {
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

          if (evt.type === 'training_complete') {
            const accuracy = evt.accuracy as number | undefined;
            setActiveSession((prev) => {
              if (!prev) return prev;
              return { ...prev, accuracy: accuracy ?? prev.accuracy };
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

          if (evt.type === 'session_complete') {
            const accuracy = evt.accuracy as number | undefined;
            setActiveSession((prev) => {
              if (!prev) return prev;
              return { ...prev, status: 'complete', accuracy: accuracy ?? prev.accuracy };
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

          if (evt.type === 'session_failed') {
            setActiveSession((prev) => {
              if (!prev) return prev;
              return { ...prev, status: 'failed' };
            });
          }
        } catch {
          // Ignore malformed messages
        }
      };
    } catch {
      // WebSocket unavailable
    }
  }, [selectedId]);

  // Select session
  const selectSession = useCallback((id: string) => {
    setSelectedId(id);
    setEvents([]);
    setLosses([]);
    setActiveSession(null);
  }, []);

  // Poll all sessions every 5s
  useEffect(() => {
    fetchSessions();
    sessionsPollRef.current = setInterval(fetchSessions, 5000);
    return () => {
      if (sessionsPollRef.current) clearInterval(sessionsPollRef.current);
    };
  }, [fetchSessions]);

  // When selectedId changes, connect WS + poll active session
  useEffect(() => {
    if (!selectedId) {
      setActiveSession(null);
      setLosses([]);
      setEvents([]);
      return;
    }

    fetchActiveSession(selectedId);
    connectWebSocket(selectedId);
    activePollRef.current = setInterval(() => fetchActiveSession(selectedId), 2000);

    return () => {
      if (activePollRef.current) clearInterval(activePollRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
    };
  }, [selectedId, fetchActiveSession, connectWebSocket]);

  return {
    sessions,
    activeSession,
    losses,
    events,
    isLoading,
    selectSession,
  };
}

export default useDashboardSessions;
