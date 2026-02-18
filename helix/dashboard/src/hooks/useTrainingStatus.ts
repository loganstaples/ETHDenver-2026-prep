'use client';

/**
 * HELIX Training Status Hook
 * Real-time training state management with WebSocket streaming and contract integration.
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useWebSocket } from '@/lib/websocket';
import { useContractEvents, useModel, useErrorBound } from './useContract';

// ============================================================================
// Types
// ============================================================================

export interface TrainingStatus {
    modelId: bigint;
    sessionId: string;
    status: 'idle' | 'initializing' | 'training' | 'paused' | 'completed' | 'failed';
    phase: 'setup' | 'forward' | 'backward' | 'aggregation' | 'proof_generation' | 'verification';
    progress: number;
    startedAt: number;
    updatedAt: number;
    estimatedCompletion?: number;
}

export interface TrainingMetricsLive {
    epoch: number;
    batch: number;
    totalEpochs: number;
    totalBatches: number;
    loss: number;
    accuracy: number;
    learningRate: number;
    gradientNorm: number;
    throughput: number;
    errorBound: number;
    timestamp: number;
}

export interface TrainingConfig {
    modelId: bigint;
    modelName: string;
    architecture: string;
    parameters: number;
    epochs: number;
    batchSize: number;
    learningRate: number;
    optimizer: string;
    lossFunction: string;
    mpcThreshold: number;
    proofFrequency: number;
    maxErrorBound: number;
}

export interface WorkerContribution {
    address: string;
    proofsSubmitted: number;
    lastContribution: number;
    status: 'active' | 'idle' | 'disconnected';
    currentTask?: string;
}

export interface TrainingAlert {
    id: string;
    type: 'info' | 'warning' | 'error' | 'success';
    title: string;
    message: string;
    timestamp: number;
    acknowledged: boolean;
    data?: Record<string, unknown>;
}

export interface UseTrainingStatusOptions {
    modelId: bigint | number;
    sessionId?: string;
    enableWebSocket?: boolean;
    enablePolling?: boolean;
    pollingInterval?: number;
    onStatusChange?: (status: TrainingStatus) => void;
    onMetricsUpdate?: (metrics: TrainingMetricsLive) => void;
    onAlert?: (alert: TrainingAlert) => void;
}

export interface UseTrainingStatusReturn {
    // Core State
    status: TrainingStatus | null;
    metrics: TrainingMetricsLive | null;
    config: TrainingConfig | null;

    // History
    lossHistory: Array<{ epoch: number; batch: number; loss: number; timestamp: number }>;
    accuracyHistory: Array<{ epoch: number; accuracy: number; timestamp: number }>;
    errorBoundHistory: Array<{ epoch: number; bound: number; timestamp: number }>;
    throughputHistory: Array<{ timestamp: number; value: number }>;

    // Workers
    workers: WorkerContribution[];
    activeWorkerCount: number;

    // Alerts
    alerts: TrainingAlert[];
    unacknowledgedAlerts: TrainingAlert[];
    acknowledgeAlert: (alertId: string) => void;
    clearAlerts: () => void;

    // Connection
    isConnected: boolean;
    isLoading: boolean;
    error: string | null;

    // Actions
    refresh: () => Promise<void>;
    pauseTraining: () => Promise<void>;
    resumeTraining: () => Promise<void>;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useTrainingStatus(options: UseTrainingStatusOptions): UseTrainingStatusReturn {
    const {
        modelId,
        sessionId,
        enableWebSocket = true,
        enablePolling: _enablePolling = true,
        pollingInterval: _pollingInterval = 2000,
        onStatusChange,
        onMetricsUpdate,
        onAlert,
    } = options;

    const modelIdBigInt = typeof modelId === 'number' ? BigInt(modelId) : modelId;

    // Core State
    const [status, setStatus] = useState<TrainingStatus | null>(null);
    const [metrics, setMetrics] = useState<TrainingMetricsLive | null>(null);
    const [config, setConfig] = useState<TrainingConfig | null>(null);

    // History
    const [lossHistory, setLossHistory] = useState<Array<{ epoch: number; batch: number; loss: number; timestamp: number }>>([]);
    const [accuracyHistory, setAccuracyHistory] = useState<Array<{ epoch: number; accuracy: number; timestamp: number }>>([]);
    const [errorBoundHistory, setErrorBoundHistory] = useState<Array<{ epoch: number; bound: number; timestamp: number }>>([]);
    const [throughputHistory, setThroughputHistory] = useState<Array<{ timestamp: number; value: number }>>([]);

    // Workers
    const [workers, setWorkers] = useState<WorkerContribution[]>([]);

    // Alerts
    const [alerts, setAlerts] = useState<TrainingAlert[]>([]);

    // Loading/Error
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Refs
    const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // Contract Integration
    const { model } = useModel(modelIdBigInt);
    const { errorBound: contractErrorBound } = useErrorBound(modelIdBigInt);
    const { proofEvents, slashedEvents } = useContractEvents(modelIdBigInt);

    // WebSocket Connection
    const { isConnected, subscribe: _subscribe, on } = useWebSocket({
        autoConnect: enableWebSocket,
        channels: enableWebSocket ? [`training:${modelIdBigInt.toString()}`] : [],
    });

    // ========================================================================
    // Initialize
    // ========================================================================

    const initializeTrainingData = useCallback(async () => {
        try {
            setIsLoading(true);
            setError(null);

            // Initialize status
            const initialStatus: TrainingStatus = {
                modelId: modelIdBigInt,
                sessionId: sessionId || `session-${Date.now()}`,
                status: 'idle',
                phase: 'setup',
                progress: 0,
                startedAt: Date.now(),
                updatedAt: Date.now(),
            };
            setStatus(initialStatus);

            // Initialize config from model data if available
            const initialConfig: TrainingConfig = {
                modelId: modelIdBigInt,
                modelName: model?.ipfsHash || 'HELIX Model',
                architecture: 'Transformer',
                parameters: 0,
                epochs: 10,
                batchSize: 64,
                learningRate: 0.001,
                optimizer: 'AdamW',
                lossFunction: 'CrossEntropy',
                mpcThreshold: 3,
                proofFrequency: 10,
                maxErrorBound: 0.01,
            };
            setConfig(initialConfig);

            // No initial metrics — wait for real data from WebSocket or API
            setMetrics(null);
            setLossHistory([]);
            setAccuracyHistory([]);
            setErrorBoundHistory([]);

            // Initialize workers from proof events only (no demo workers)
            const workerMap = new Map<string, WorkerContribution>();
            proofEvents.forEach(event => {
                const existing = workerMap.get(event.prover);
                workerMap.set(event.prover, {
                    address: event.prover,
                    proofsSubmitted: (existing?.proofsSubmitted || 0) + 1,
                    lastContribution: event.timestamp,
                    status: 'active',
                });
            });
            setWorkers(Array.from(workerMap.values()));

        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to initialize training data');
        } finally {
            setIsLoading(false);
        }
    }, [modelIdBigInt, sessionId, model, proofEvents]);

    // ========================================================================
    // WebSocket Message Handlers
    // ========================================================================

    useEffect(() => {
        if (!enableWebSocket) return;

        const unsubStatus = on<TrainingStatus>('training_status', (msg) => {
            setStatus(msg.data);
            onStatusChange?.(msg.data);
        });

        const unsubMetrics = on<TrainingMetricsLive>('training_metrics', (msg) => {
            setMetrics(msg.data);
            onMetricsUpdate?.(msg.data);

            // Update history
            setLossHistory(prev => [...prev, {
                epoch: msg.data.epoch,
                batch: msg.data.batch,
                loss: msg.data.loss,
                timestamp: msg.data.timestamp,
            }].slice(-500));

            setAccuracyHistory(prev => [...prev, {
                epoch: msg.data.epoch,
                accuracy: msg.data.accuracy,
                timestamp: msg.data.timestamp,
            }].slice(-500));

            setThroughputHistory(prev => [...prev, {
                timestamp: msg.data.timestamp,
                value: msg.data.throughput,
            }].slice(-100));
        });

        const unsubAlert = on<TrainingAlert>('training_alert', (msg) => {
            const alert = { ...msg.data, acknowledged: false };
            setAlerts(prev => [alert, ...prev].slice(0, 50));
            onAlert?.(alert);
        });

        const unsubWorker = on<WorkerContribution>('worker_update', (msg) => {
            setWorkers(prev => {
                const existing = prev.find(w => w.address === msg.data.address);
                if (existing) {
                    return prev.map(w => w.address === msg.data.address ? msg.data : w);
                }
                return [...prev, msg.data];
            });
        });

        return () => {
            unsubStatus();
            unsubMetrics();
            unsubAlert();
            unsubWorker();
        };
    }, [enableWebSocket, on, onStatusChange, onMetricsUpdate, onAlert]);

    // ========================================================================
    // Effects
    // ========================================================================

    useEffect(() => {
        initializeTrainingData();
    }, [initializeTrainingData]);

    // Update error bound from contract
    useEffect(() => {
        if (contractErrorBound !== null && contractErrorBound !== undefined) {
            setMetrics(prev => prev ? {
                ...prev,
                errorBound: Number(contractErrorBound) / 1e18,
            } : prev);
        }
    }, [contractErrorBound]);

    // Create alerts from slashed events
    useEffect(() => {
        slashedEvents.forEach(event => {
            const alertId = `slash-${event.transactionHash}`;
            setAlerts(prev => {
                if (prev.some(a => a.id === alertId)) return prev;
                const newAlert: TrainingAlert = {
                    id: alertId,
                    type: 'error' as const,
                    title: 'Worker Slashed',
                    message: `Worker ${event.prover.slice(0, 10)}... was slashed: ${event.reason}`,
                    timestamp: event.timestamp,
                    acknowledged: false,
                    data: { prover: event.prover, amount: event.amount.toString() },
                };
                return [newAlert, ...prev].slice(0, 50);
            });
        });
    }, [slashedEvents]);

    // ========================================================================
    // Computed Values
    // ========================================================================

    const activeWorkerCount = useMemo(
        () => workers.filter(w => w.status === 'active').length,
        [workers]
    );

    const unacknowledgedAlerts = useMemo(
        () => alerts.filter(a => !a.acknowledged),
        [alerts]
    );

    // ========================================================================
    // Actions
    // ========================================================================

    const acknowledgeAlert = useCallback((alertId: string) => {
        setAlerts(prev => prev.map(a =>
            a.id === alertId ? { ...a, acknowledged: true } : a
        ));
    }, []);

    const clearAlerts = useCallback(() => {
        setAlerts([]);
    }, []);

    const refresh = useCallback(async () => {
        await initializeTrainingData();
    }, [initializeTrainingData]);

    const pauseTraining = useCallback(async () => {
        setStatus(prev => prev ? { ...prev, status: 'paused', updatedAt: Date.now() } : prev);
        if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
        }
    }, []);

    const resumeTraining = useCallback(async () => {
        setStatus(prev => prev ? { ...prev, status: 'training', updatedAt: Date.now() } : prev);
    }, []);

    // ========================================================================
    // Return
    // ========================================================================

    return {
        status,
        metrics,
        config,
        lossHistory,
        accuracyHistory,
        errorBoundHistory,
        throughputHistory,
        workers,
        activeWorkerCount,
        alerts,
        unacknowledgedAlerts,
        acknowledgeAlert,
        clearAlerts,
        isConnected,
        isLoading,
        error,
        refresh,
        pauseTraining,
        resumeTraining,
    };
}

export default useTrainingStatus;
