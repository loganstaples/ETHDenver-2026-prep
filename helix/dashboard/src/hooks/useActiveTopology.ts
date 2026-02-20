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
    tasks: Array<{
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
    action: 'add' | 'update' | 'remove';
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

            for (const task of data.tasks) {
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

            apiClient.subscribeToChannel('topology');

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

                        if (d.action === 'remove') {
                            next.delete(d.model_id);
                            return next;
                        }

                        const existing = next.get(d.model_id);

                        if (d.action === 'add' || !existing) {
                            next.set(d.model_id, {
                                modelId: d.model_id,
                                modelName: d.model_name ?? existing?.modelName ?? `Model ${d.model_id}`,
                                taskType: d.task_type ?? existing?.taskType ?? 'training',
                                workerIds: d.worker_ids ?? existing?.workerIds ?? [],
                                startedAt: d.started_at ?? existing?.startedAt ?? Date.now(),
                            });
                        } else {
                            // update
                            next.set(d.model_id, {
                                ...existing,
                                modelName: d.model_name ?? existing.modelName,
                                taskType: d.task_type ?? existing.taskType,
                                workerIds: d.worker_ids ?? existing.workerIds,
                                startedAt: d.started_at ?? existing.startedAt,
                            });
                        }

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
    }, [models]);

    return { activeModels, isConnected };
}

export default useActiveTopology;
