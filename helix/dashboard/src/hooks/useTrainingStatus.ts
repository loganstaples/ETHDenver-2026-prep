'use client';

/**
 * HELIX Training Status Hook
 * Real-time training state management with WebSocket streaming and contract integration.
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useWebSocket, type WebSocketMessage } from '@/lib/websocket';
import { useContractEvents, useModel, useErrorBound } from './useContract';
import { getApiClient, generateMockTrainingMetrics } from '@/lib/api';

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
        enablePolling = true,
        pollingInterval = 2000,
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

    // Refs for animation
    const epochRef = useRef(0);
    const batchRef = useRef(0);
    const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // Contract Integration
    const { model } = useModel(modelIdBigInt);
    const { errorBound: contractErrorBound } = useErrorBound(modelIdBigInt);
    const { proofEvents, slashedEvents } = useContractEvents(modelIdBigInt);

    // WebSocket Connection
    const { isConnected, subscribe, on } = useWebSocket({
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
                status: 'training',
                phase: 'forward',
                progress: 0,
                startedAt: Date.now() - 3600000,
                updatedAt: Date.now(),
            };
            setStatus(initialStatus);

            // Initialize config
            const initialConfig: TrainingConfig = {
                modelId: modelIdBigInt,
                modelName: model?.ipfsHash || 'HELIX Transformer',
                architecture: 'Transformer',
                parameters: 1_500_000,
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

            // Initialize metrics with mock data
            const mockMetrics = generateMockTrainingMetrics(0);
            const initialMetrics: TrainingMetricsLive = {
                epoch: 0,
                batch: 0,
                totalEpochs: 10,
                totalBatches: 100,
                loss: mockMetrics.loss,
                accuracy: mockMetrics.accuracy,
                learningRate: 0.001,
                gradientNorm: mockMetrics.gradientNorm,
                throughput: mockMetrics.throughput,
                errorBound: mockMetrics.accumulatedErrorBound,
                timestamp: Date.now(),
            };
            setMetrics(initialMetrics);

            // Initialize history
            setLossHistory(mockMetrics.lossHistory.map(h => ({
                ...h,
                batch: 0,
            })));
            setAccuracyHistory(mockMetrics.accuracyHistory);
            setErrorBoundHistory(mockMetrics.errorBoundHistory);

            // Initialize workers from proof events
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

            // Add some demo workers if none from events
            if (workerMap.size === 0) {
                const demoWorkers = [
                    '0x742d35Cc6634C0532925a3b844Bc9e7595f01231',
                    '0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199',
                    '0xdD2FD4581271e230360230F9337D5c0430Bf44C0',
                ];
                demoWorkers.forEach((addr, i) => {
                    workerMap.set(addr, {
                        address: addr,
                        proofsSubmitted: Math.floor(Math.random() * 50) + 10,
                        lastContribution: Date.now() - Math.random() * 60000,
                        status: i === 0 ? 'active' : i === 1 ? 'idle' : 'active',
                        currentTask: i === 0 ? 'Computing gradients' : undefined,
                    });
                });
            }
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
    // Polling Animation
    // ========================================================================

    useEffect(() => {
        if (!enablePolling) return;

        pollingRef.current = setInterval(() => {
            // Advance batch
            batchRef.current += 1;
            const totalBatches = config?.batchSize ? 100 : 100;

            if (batchRef.current >= totalBatches) {
                batchRef.current = 0;
                epochRef.current = Math.min(epochRef.current + 1, (config?.epochs || 10));
            }

            // Calculate new metrics
            const epoch = epochRef.current;
            const batch = batchRef.current;
            const baseLoss = 2.5 * Math.exp(-0.3 * epoch) + 0.1;
            const baseAccuracy = Math.min(0.99, 1 - Math.exp(-0.2 * epoch) * 0.8);

            const newMetrics: TrainingMetricsLive = {
                epoch,
                batch,
                totalEpochs: config?.epochs || 10,
                totalBatches,
                loss: baseLoss + (Math.random() - 0.5) * 0.05,
                accuracy: baseAccuracy + (Math.random() - 0.5) * 0.02,
                learningRate: 0.001 * Math.pow(0.95, epoch),
                gradientNorm: 0.1 + Math.random() * 0.4,
                throughput: 50 + Math.random() * 30,
                errorBound: 0.0001 * epoch + Math.random() * 0.00005,
                timestamp: Date.now(),
            };

            setMetrics(newMetrics);

            // Update loss history at batch boundaries
            if (batch % 10 === 0) {
                setLossHistory(prev => [...prev, {
                    epoch,
                    batch,
                    loss: newMetrics.loss,
                    timestamp: Date.now(),
                }].slice(-500));
            }

            // Update accuracy/error at epoch boundaries
            if (batch === 0 && epoch > 0) {
                setAccuracyHistory(prev => [...prev, {
                    epoch,
                    accuracy: newMetrics.accuracy,
                    timestamp: Date.now(),
                }].slice(-100));

                setErrorBoundHistory(prev => [...prev, {
                    epoch,
                    bound: newMetrics.errorBound,
                    timestamp: Date.now(),
                }].slice(-100));
            }

            // Update status
            setStatus(prev => prev ? {
                ...prev,
                progress: ((epoch * totalBatches + batch) / ((config?.epochs || 10) * totalBatches)) * 100,
                phase: batch < 30 ? 'forward' : batch < 60 ? 'backward' : batch < 80 ? 'aggregation' : 'proof_generation',
                updatedAt: Date.now(),
            } : prev);

            // Update throughput history
            setThroughputHistory(prev => [...prev, {
                timestamp: Date.now(),
                value: newMetrics.throughput,
            }].slice(-100));

            // Simulate worker activity
            setWorkers(prev => prev.map(w => ({
                ...w,
                status: Math.random() > 0.1 ? 'active' : 'idle',
                lastContribution: Math.random() > 0.7 ? Date.now() : w.lastContribution,
                proofsSubmitted: Math.random() > 0.9 ? w.proofsSubmitted + 1 : w.proofsSubmitted,
            })));

        }, pollingInterval);

        return () => {
            if (pollingRef.current) {
                clearInterval(pollingRef.current);
            }
        };
    }, [enablePolling, pollingInterval, config]);

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
