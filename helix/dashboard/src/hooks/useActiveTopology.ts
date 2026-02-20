'use client';

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { getApiClient } from '@/lib/api';

// ============================================================================
// Types
// ============================================================================

export interface ActiveModel {
    modelId: string;
    modelName: string;
    taskType: 'training' | 'inference';
    workerIds: string[];
    startedAt: number; // epoch ms
}

interface ActiveTasksResponse {
    active_models: Array<{
        model_id: string;
        model_name: string;
        task_type: 'training' | 'inference';
        worker_ids: string[];
        started_at: number;
    }>;
}

interface WorkerAssignedData {
    model_id: string;
    model_name?: string;
    task_type?: 'training' | 'inference';
    worker_id: string;
    started_at?: number;
}

interface WorkerRemovedData {
    model_id: string;
    worker_id: string;
}

interface ModelActivityChangedData {
    model_id: string;
    model_name?: string;
    task_type?: 'training' | 'inference';
    worker_ids?: string[];
    started_at?: number;
    // Backend sends "started"/"completed"/"failed"; treat completed/failed as remove
    action?: string;
    active?: boolean;
}

// ============================================================================
// Constants
// ============================================================================

const INFERENCE_MIN_ACTIVE_MS = 5000;
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

// ============================================================================
// Hook Implementation
// ============================================================================

export function useActiveTopology(): {
    activeModels: ActiveModel[];
    isConnected: boolean;
} {
    const [models, setModels] = useState<Map<string, ActiveModel>>(new Map());
    const [isConnected, setIsConnected] = useState(false);
    const wsUnsubscribeRef = useRef<(() => void) | null>(null);
    const apiClient = useMemo(() => getApiClient(), []);

    // ========================================================================
    // Initial Fetch
    // ========================================================================

    const fetchActiveTasks = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/active-tasks`);
            if (!res.ok) return;

            const data: ActiveTasksResponse = await res.json();
            const newModels = new Map<string, ActiveModel>();

            for (const task of data.active_models) {
                newModels.set(task.model_id, {
                    modelId: task.model_id,
                    modelName: task.model_name,
                    taskType: task.task_type,
                    workerIds: [...task.worker_ids],
                    startedAt: task.started_at,
                });
            }

            setModels(newModels);
        } catch {
            // Backend may not be running; keep existing state
        }
    }, []);

    // ========================================================================
    // WebSocket Setup
    // ========================================================================

    const setupWebSocket = useCallback(async () => {
        try {
            await apiClient.connectWebSocket();
            setIsConnected(true);

            apiClient.subscribeToChannel('nodes');

            // worker_assigned: add worker to model's workerIds
            const unsubAssigned = apiClient.onWebSocketMessage<WorkerAssignedData>(
                'worker_assigned',
                (message) => {
                    const d = message.data;
                    if (!d.model_id || !d.worker_id) return;

                    setModels((prev) => {
                        const next = new Map(prev);
                        const existing = next.get(d.model_id);

                        if (existing) {
                            if (!existing.workerIds.includes(d.worker_id)) {
                                next.set(d.model_id, {
                                    ...existing,
                                    workerIds: [...existing.workerIds, d.worker_id],
                                });
                            }
                        } else {
                            next.set(d.model_id, {
                                modelId: d.model_id,
                                modelName: d.model_name ?? `Model ${d.model_id}`,
                                taskType: d.task_type ?? 'training',
                                workerIds: [d.worker_id],
                                startedAt: d.started_at ?? Date.now(),
                            });
                        }

                        return next;
                    });
                },
            );

            // worker_removed: remove worker from model; remove model if 0 workers
            const unsubRemoved = apiClient.onWebSocketMessage<WorkerRemovedData>(
                'worker_removed',
                (message) => {
                    const d = message.data;
                    if (!d.model_id || !d.worker_id) return;

                    setModels((prev) => {
                        const next = new Map(prev);
                        const existing = next.get(d.model_id);
                        if (!existing) return prev;

                        const remaining = existing.workerIds.filter(
                            (wid) => wid !== d.worker_id,
                        );

                        if (remaining.length === 0) {
                            next.delete(d.model_id);
                        } else {
                            next.set(d.model_id, {
                                ...existing,
                                workerIds: remaining,
                            });
                        }

                        return next;
                    });
                },
            );

            // model_activity_changed: add/update/remove model
            const unsubActivity = apiClient.onWebSocketMessage<ModelActivityChangedData>(
                'model_activity_changed',
                (message) => {
                    const d = message.data;
                    if (!d.model_id) return;

                    setModels((prev) => {
                        const next = new Map(prev);

                        // Determine if this is a removal: action=completed/failed/remove or active=false
                        const isRemove =
                            d.active === false ||
                            d.action === 'completed' ||
                            d.action === 'failed' ||
                            d.action === 'remove';

                        if (isRemove) {
                            next.delete(d.model_id);
                            return next;
                        }

                        // Add or update
                        const existing = next.get(d.model_id);
                        next.set(d.model_id, {
                            modelId: d.model_id,
                            modelName: d.model_name ?? existing?.modelName ?? `Model ${d.model_id}`,
                            taskType: d.task_type ?? existing?.taskType ?? 'training',
                            workerIds: d.worker_ids ?? existing?.workerIds ?? [],
                            startedAt: d.started_at ?? existing?.startedAt ?? Date.now(),
                        });

                        return next;
                    });
                },
            );

            wsUnsubscribeRef.current = () => {
                unsubAssigned();
                unsubRemoved();
                unsubActivity();
            };
        } catch (err) {
            console.warn('useActiveTopology: WebSocket connection failed:', err);
            setIsConnected(false);
        }
    }, [apiClient]);

    // ========================================================================
    // Effects
    // ========================================================================

    useEffect(() => {
        fetchActiveTasks();
        setupWebSocket();

        return () => {
            wsUnsubscribeRef.current?.();
        };
    }, [fetchActiveTasks, setupWebSocket]);

    // ========================================================================
    // Re-render timer for inference 5s filter
    // ========================================================================

    const [tick, setTick] = useState(0);

    useEffect(() => {
        // Check if any inference tasks are pending the 5s threshold
        const now = Date.now();
        let hasWaiting = false;
        models.forEach((model) => {
            if (model.taskType === 'inference' && now - model.startedAt < INFERENCE_MIN_ACTIVE_MS) {
                hasWaiting = true;
            }
        });

        if (!hasWaiting) return;

        // Re-trigger filter every second until all pending tasks cross threshold
        const timer = setInterval(() => setTick((t) => t + 1), 1000);
        return () => clearInterval(timer);
    }, [models]);

    // ========================================================================
    // Filter out inference tasks active < 5 seconds
    // ========================================================================

    const activeModels = useMemo(() => {
        const now = Date.now();
        const result: ActiveModel[] = [];

        models.forEach((model) => {
            if (
                model.taskType === 'inference' &&
                now - model.startedAt < INFERENCE_MIN_ACTIVE_MS
            ) {
                return; // skip inference tasks active < 5s
            }
            result.push(model);
        });

        return result;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [models, tick]);

    return { activeModels, isConnected };
}

export default useActiveTopology;
