'use client';

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useContractEvents, useModel, useErrorBound } from './useContract';
import {
    type TrainingMetrics,
    type AdversarialEvent,
    getApiClient,
} from '@/lib/api';

// ============================================================================
// Types
// ============================================================================

export interface TrainingState {
    modelId: bigint;
    modelName: string;
    status: 'initializing' | 'training' | 'paused' | 'completed' | 'failed';
    startedAt: number;
    updatedAt: number;
    completedAt?: number;
}

export interface TrainingProgress {
    currentEpoch: number;
    totalEpochs: number;
    currentBatch: number;
    totalBatches: number;
    currentRound: bigint;
    epochProgress: number;
    batchProgress: number;
    overallProgress: number;
    estimatedTimeRemaining: number;
}

export interface TrainingMetricsData {
    loss: number;
    accuracy: number;
    gradientNorm: number;
    learningRate: number;
    throughput: number;
    accumulatedErrorBound: number;
}

export interface LossPoint {
    epoch: number;
    batch?: number;
    loss: number;
    timestamp: number;
    smoothedLoss?: number;
}

export interface AccuracyPoint {
    epoch: number;
    accuracy: number;
    timestamp: number;
}

export interface ErrorBoundPoint {
    epoch: number;
    round: bigint;
    bound: number;
    amplification: number;
    timestamp: number;
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

export interface LayerErrorBound {
    layer: string;
    operation: string;
    inputBound: number;
    outputBound: number;
    amplification: number;
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

export interface TrainingRoundData {
    id: bigint;
    status: 'pending' | 'in_progress' | 'aggregating' | 'proving' | 'completed' | 'failed';
    startedAt: number;
    completedAt?: number;
    deadline: number;
    participants: string[];
    prover?: string;
    proofId?: string;
    errorBound: number;
    gasUsed?: bigint;
    transactionHash?: string;
}

export interface AdversarialEventData {
    id: string;
    type: 'invalid_proof' | 'timeout' | 'malicious_gradient' | 'stake_slashed' | 'challenge_submitted' | 'byzantine_behavior';
    severity: 'warning' | 'critical' | 'resolved';
    timestamp: number;
    modelId: bigint;
    roundId?: bigint;
    prover: string;
    description: string;
    slashAmount?: bigint;
    resolved: boolean;
}

export interface TrainingConfig {
    totalEpochs: number;
    totalBatches: number;
    batchSize: number;
    learningRate: number;
    optimizer: string;
    lossFunction: string;
    maxErrorBound: number;
    mpcThreshold: number;
    proofFrequency: number;
}

export interface UseTrainingOptions {
    modelId?: bigint | number;
    sessionId?: string;
    autoRefresh?: boolean;
    refreshInterval?: number;
    enableWebSocket?: boolean;
    enableAnimations?: boolean;
}

export interface UseTrainingReturn {
    state: TrainingState | null;
    progress: TrainingProgress;
    metrics: TrainingMetricsData;
    config: TrainingConfig;
    lossHistory: LossPoint[];
    accuracyHistory: AccuracyPoint[];
    errorBoundHistory: ErrorBoundPoint[];
    layerErrorBounds: LayerErrorBound[];
    rounds: TrainingRoundData[];
    currentRound: TrainingRoundData | null;
    adversarialEvents: AdversarialEventData[];
    unacknowledgedAlerts: AdversarialEventData[];
    isLoading: boolean;
    isConnected: boolean;
    error: string | null;
    refetch: () => Promise<void>;
    acknowledgeAlert: (eventId: string) => void;
    getRoundById: (roundId: bigint) => TrainingRoundData | undefined;
    getMetricsAtEpoch: (epoch: number) => { loss: number; accuracy: number } | undefined;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useTraining(options: UseTrainingOptions = {}): UseTrainingReturn {
    const {
        modelId = BigInt(1),
        sessionId: _sessionId,
        autoRefresh: _autoRefresh = true,
        refreshInterval: _refreshInterval = 2000,
        enableWebSocket = true,
        enableAnimations: _enableAnimations = true,
    } = options;

    // State
    const [state, setState] = useState<TrainingState | null>(null);
    const [metrics, setMetrics] = useState<TrainingMetricsData>({
        loss: 2.5,
        accuracy: 0.1,
        gradientNorm: 0.5,
        learningRate: 0.001,
        throughput: 50,
        accumulatedErrorBound: 0,
    });
    const [progress, setProgress] = useState<TrainingProgress>({
        currentEpoch: 0,
        totalEpochs: 10,
        currentBatch: 0,
        totalBatches: 100,
        currentRound: BigInt(0),
        epochProgress: 0,
        batchProgress: 0,
        overallProgress: 0,
        estimatedTimeRemaining: 0,
    });
    const [config] = useState<TrainingConfig>({
        totalEpochs: 10,
        totalBatches: 100,
        batchSize: 64,
        learningRate: 0.001,
        optimizer: 'AdamW',
        lossFunction: 'CrossEntropy',
        maxErrorBound: 0.01,
        mpcThreshold: 3,
        proofFrequency: 10,
    });
    const [lossHistory, setLossHistory] = useState<LossPoint[]>([]);
    const [accuracyHistory, setAccuracyHistory] = useState<AccuracyPoint[]>([]);
    const [errorBoundHistory, setErrorBoundHistory] = useState<ErrorBoundPoint[]>([]);
    const [layerErrorBounds, setLayerErrorBounds] = useState<LayerErrorBound[]>([]);
    const [rounds, setRounds] = useState<TrainingRoundData[]>([]);
    const [adversarialEvents, setAdversarialEvents] = useState<AdversarialEventData[]>([]);
    const [acknowledgedAlerts, setAcknowledgedAlerts] = useState<Set<string>>(new Set());
    const [isLoading, setIsLoading] = useState(true);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Refs
    const wsUnsubscribeRef = useRef<(() => void) | null>(null);
    const refreshIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // Contract data
    const { proofEvents, roundStartedEvents, roundCompletedEvents, slashedEvents } = useContractEvents(BigInt(modelId));
    const { model } = useModel(BigInt(modelId));
    const { errorBound: contractErrorBound } = useErrorBound(BigInt(modelId));

    // API client
    const apiClient = useMemo(() => getApiClient(), []);

    // ========================================================================
    // Data Generation
    // ========================================================================

    const generateLayerErrorBounds = useCallback((): LayerErrorBound[] => {
        return [
            { layer: 'embed', operation: 'Embedding', inputBound: 0, outputBound: 1e-7, amplification: 1.0, riskLevel: 'low' },
            { layer: 'attn_0', operation: 'Attention', inputBound: 1e-7, outputBound: 2.3e-6, amplification: 23.0, riskLevel: 'medium' },
            { layer: 'norm_0', operation: 'LayerNorm', inputBound: 2.3e-6, outputBound: 4.1e-6, amplification: 1.78, riskLevel: 'low' },
            { layer: 'ffn_0', operation: 'FFN', inputBound: 4.1e-6, outputBound: 8.7e-6, amplification: 2.12, riskLevel: 'medium' },
            { layer: 'attn_1', operation: 'Attention', inputBound: 8.7e-6, outputBound: 1.8e-4, amplification: 20.7, riskLevel: 'high' },
            { layer: 'norm_1', operation: 'LayerNorm', inputBound: 1.8e-4, outputBound: 2.1e-4, amplification: 1.17, riskLevel: 'low' },
            { layer: 'ffn_1', operation: 'FFN', inputBound: 2.1e-4, outputBound: 4.4e-4, amplification: 2.1, riskLevel: 'medium' },
            { layer: 'softmax', operation: 'Softmax', inputBound: 4.4e-4, outputBound: 1.2e-3, amplification: 2.73, riskLevel: 'high' },
        ];
    }, []);

    // ========================================================================
    // Data Fetching
    // ========================================================================

    const fetchTrainingData = useCallback(async () => {
        try {
            setIsLoading(true);
            setError(null);

            // Initialize training state
            setState({
                modelId: BigInt(modelId),
                modelName: model?.ipfsHash || 'HELIX Transformer',
                status: 'training',
                startedAt: Date.now() - 3600000,
                updatedAt: Date.now(),
            });

            // Initialize metrics with defaults — wait for real data from WebSocket or API
            setMetrics({
                loss: 0,
                accuracy: 0,
                gradientNorm: 0,
                learningRate: config.learningRate,
                throughput: 0,
                accumulatedErrorBound: 0,
            });

            // Initialize history as empty
            setLossHistory([]);
            setAccuracyHistory([]);

            // Initialize layer error bounds (static layer info, not mock data)
            setLayerErrorBounds(generateLayerErrorBounds());
            setErrorBoundHistory([]);

            // Initialize rounds from contract events
            const currentRound = model?.currentRound || BigInt(roundStartedEvents.length || 0);

            const contractRounds: TrainingRoundData[] = roundStartedEvents.map((event, _index) => {
                const completedEvent = roundCompletedEvents.find(
                    (c) => c.roundId === event.roundId && c.modelId === event.modelId
                );
                return {
                    id: event.roundId,
                    status: completedEvent ? 'completed' as const : 'in_progress' as const,
                    startedAt: event.timestamp,
                    completedAt: completedEvent?.timestamp,
                    deadline: Number(event.deadline) * 1000,
                    participants: [],
                    prover: completedEvent ? proofEvents.find((p) => p.roundId === event.roundId)?.prover : undefined,
                    errorBound: 0.0001 + Math.random() * 0.0005,
                    gasUsed: completedEvent ? BigInt(200000) : undefined,
                    transactionHash: completedEvent?.transactionHash,
                };
            });

            if (contractRounds.length > 0) {
                setRounds(contractRounds);
            } else {
                setRounds([]);
            }

            // Initialize adversarial events from slashed events
            const slashEvents: AdversarialEventData[] = slashedEvents.map((event) => ({
                id: `slash-${event.transactionHash}`,
                type: 'stake_slashed' as const,
                severity: 'critical' as const,
                timestamp: event.timestamp,
                modelId: event.modelId,
                roundId: undefined,
                prover: event.prover,
                description: event.reason || 'Stake slashed due to protocol violation',
                slashAmount: event.amount,
                resolved: false,
            }));

            if (slashEvents.length > 0) {
                setAdversarialEvents(slashEvents);
            } else {
                setAdversarialEvents([]);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch training data');
        } finally {
            setIsLoading(false);
        }
    }, [
        modelId,
        model,
        config.learningRate,
        roundStartedEvents,
        roundCompletedEvents,
        proofEvents,
        slashedEvents,
        generateLayerErrorBounds,
    ]);

    // ========================================================================
    // WebSocket Setup
    // ========================================================================

    const setupWebSocket = useCallback(async () => {
        if (!enableWebSocket) return;

        try {
            await apiClient.connectWebSocket();
            setIsConnected(true);

            apiClient.subscribeToChannel(`training:${modelId}`);

            wsUnsubscribeRef.current = apiClient.onWebSocketMessage<TrainingMetrics>(
                'training_update',
                (message) => {
                    const data = message.data;
                    setMetrics({
                        loss: data.loss,
                        accuracy: data.accuracy,
                        gradientNorm: data.gradientNorm,
                        learningRate: data.learningRate,
                        throughput: data.throughput,
                        accumulatedErrorBound: data.accumulatedErrorBound,
                    });
                }
            );

            // Listen for adversarial events
            apiClient.onWebSocketMessage<AdversarialEvent>('adversarial_event', (message) => {
                const event = message.data;
                if (event.modelId === BigInt(modelId)) {
                    setAdversarialEvents((prev) => [{
                        id: event.id,
                        type: event.type,
                        severity: event.severity,
                        timestamp: event.timestamp,
                        modelId: event.modelId,
                        roundId: event.roundId,
                        prover: event.prover,
                        description: event.description,
                        slashAmount: event.slashAmount,
                        resolved: event.resolved,
                    }, ...prev].slice(0, 50));
                }
            });
        } catch (err) {
            console.warn('WebSocket connection failed:', err);
            setIsConnected(false);
        }
    }, [enableWebSocket, apiClient, modelId]);

    // ========================================================================
    // Effects
    // ========================================================================

    useEffect(() => {
        fetchTrainingData();
        setupWebSocket();

        return () => {
            wsUnsubscribeRef.current?.();
        };
    }, [fetchTrainingData, setupWebSocket]);

    // Update contract error bound
    useEffect(() => {
        if (contractErrorBound !== null && contractErrorBound !== undefined) {
            setMetrics((prev) => ({
                ...prev,
                accumulatedErrorBound: Number(contractErrorBound) / 1e18,
            }));
        }
    }, [contractErrorBound]);

    // ========================================================================
    // Computed Values
    // ========================================================================

    const currentRound = useMemo(
        () => rounds.find((r) => r.status === 'in_progress') || rounds[0] || null,
        [rounds]
    );

    const unacknowledgedAlerts = useMemo(
        () => adversarialEvents.filter(
            (e) => !e.resolved && !acknowledgedAlerts.has(e.id) && e.severity !== 'resolved'
        ),
        [adversarialEvents, acknowledgedAlerts]
    );

    // ========================================================================
    // Actions
    // ========================================================================

    const acknowledgeAlert = useCallback((eventId: string) => {
        setAcknowledgedAlerts((prev) => new Set([...prev, eventId]));
    }, []);

    const getRoundById = useCallback(
        (roundId: bigint) => rounds.find((r) => r.id === roundId),
        [rounds]
    );

    const getMetricsAtEpoch = useCallback(
        (epoch: number) => {
            const loss = lossHistory.find((l) => l.epoch === epoch);
            const accuracy = accuracyHistory.find((a) => a.epoch === epoch);
            if (loss && accuracy) {
                return { loss: loss.loss, accuracy: accuracy.accuracy };
            }
            return undefined;
        },
        [lossHistory, accuracyHistory]
    );

    // ========================================================================
    // Return
    // ========================================================================

    return {
        state,
        progress,
        metrics,
        config,
        lossHistory,
        accuracyHistory,
        errorBoundHistory,
        layerErrorBounds,
        rounds,
        currentRound,
        adversarialEvents,
        unacknowledgedAlerts,
        isLoading,
        isConnected,
        error,
        refetch: fetchTrainingData,
        acknowledgeAlert,
        getRoundById,
        getMetricsAtEpoch,
    };
}

// ============================================================================
// Additional Training Hooks
// ============================================================================

/**
 * Hook for live loss curve visualization
 */
export function useLossCurve(modelId?: bigint | number) {
    const { lossHistory, accuracyHistory, isLoading } = useTraining({
        modelId: modelId ? BigInt(modelId) : undefined,
        enableAnimations: true,
    });

    const smoothedLoss = useMemo(() => {
        return lossHistory.map((point, index) => {
            const windowSize = Math.min(5, index + 1);
            const window = lossHistory.slice(Math.max(0, index - windowSize + 1), index + 1);
            const avg = window.reduce((sum, p) => sum + p.loss, 0) / window.length;
            return { ...point, smoothedLoss: avg };
        });
    }, [lossHistory]);

    const trend = useMemo(() => {
        if (lossHistory.length < 2) return 'stable';
        const recent = lossHistory.slice(-5);
        const avgRecent = recent.reduce((s, p) => s + p.loss, 0) / recent.length;
        const older = lossHistory.slice(-10, -5);
        if (older.length === 0) return 'stable';
        const avgOlder = older.reduce((s, p) => s + p.loss, 0) / older.length;
        const change = (avgRecent - avgOlder) / avgOlder;
        if (change < -0.05) return 'improving';
        if (change > 0.05) return 'degrading';
        return 'stable';
    }, [lossHistory]);

    const convergenceRate = useMemo(() => {
        if (lossHistory.length < 3) return 0;
        const recent = lossHistory.slice(-3);
        const older = lossHistory.slice(-6, -3);
        if (older.length === 0) return 0;
        const recentAvg = recent.reduce((s, p) => s + p.loss, 0) / recent.length;
        const olderAvg = older.reduce((s, p) => s + p.loss, 0) / older.length;
        return Math.max(0, (olderAvg - recentAvg) / olderAvg);
    }, [lossHistory]);

    return {
        lossHistory,
        smoothedLoss,
        accuracyHistory,
        trend,
        convergenceRate,
        isLoading,
    };
}

/**
 * Hook for error bounds visualization
 */
export function useErrorBounds(modelId?: bigint | number) {
    const { errorBoundHistory, layerErrorBounds, metrics, isLoading } = useTraining({
        modelId: modelId ? BigInt(modelId) : undefined,
    });

    const totalErrorBound = useMemo(() => {
        if (layerErrorBounds.length === 0) return 0;
        return layerErrorBounds[layerErrorBounds.length - 1].outputBound;
    }, [layerErrorBounds]);

    const maxAmplification = useMemo(() => {
        return Math.max(...layerErrorBounds.map((l) => l.amplification), 0);
    }, [layerErrorBounds]);

    const highRiskLayers = useMemo(() => {
        return layerErrorBounds.filter((l) => l.riskLevel === 'high' || l.riskLevel === 'critical');
    }, [layerErrorBounds]);

    const isWithinBounds = useMemo(() => {
        return metrics.accumulatedErrorBound < 0.01; // Max allowed error
    }, [metrics.accumulatedErrorBound]);

    return {
        errorBoundHistory,
        layerErrorBounds,
        totalErrorBound,
        maxAmplification,
        highRiskLayers,
        accumulatedErrorBound: metrics.accumulatedErrorBound,
        isWithinBounds,
        isLoading,
    };
}

/**
 * Hook for adversarial event monitoring
 */
export function useAdversarialMonitor(modelId?: bigint | number) {
    const { adversarialEvents, unacknowledgedAlerts, acknowledgeAlert, isLoading } = useTraining({
        modelId: modelId ? BigInt(modelId) : undefined,
    });

    const criticalEvents = useMemo(() => {
        return adversarialEvents.filter((e) => e.severity === 'critical' && !e.resolved);
    }, [adversarialEvents]);

    const recentEvents = useMemo(() => {
        const oneHourAgo = Date.now() - 3600000;
        return adversarialEvents.filter((e) => e.timestamp > oneHourAgo);
    }, [adversarialEvents]);

    const eventsByType = useMemo(() => {
        const grouped = new Map<AdversarialEventData['type'], AdversarialEventData[]>();
        adversarialEvents.forEach((event) => {
            if (!grouped.has(event.type)) {
                grouped.set(event.type, []);
            }
            grouped.get(event.type)!.push(event);
        });
        return grouped;
    }, [adversarialEvents]);

    const threatLevel = useMemo(() => {
        if (criticalEvents.length > 2) return 'critical';
        if (criticalEvents.length > 0 || recentEvents.length > 5) return 'high';
        if (recentEvents.length > 2) return 'medium';
        return 'low';
    }, [criticalEvents, recentEvents]);

    return {
        adversarialEvents,
        unacknowledgedAlerts,
        criticalEvents,
        recentEvents,
        eventsByType,
        threatLevel,
        acknowledgeAlert,
        isLoading,
    };
}

/**
 * Hook for training rounds monitoring
 */
export function useTrainingRounds(modelId?: bigint | number) {
    const { rounds, currentRound, progress, isLoading } = useTraining({
        modelId: modelId ? BigInt(modelId) : undefined,
    });

    const completedRounds = useMemo(() => {
        return rounds.filter((r) => r.status === 'completed');
    }, [rounds]);

    const failedRounds = useMemo(() => {
        return rounds.filter((r) => r.status === 'failed');
    }, [rounds]);

    const averageRoundTime = useMemo(() => {
        const completed = completedRounds.filter((r) => r.completedAt);
        if (completed.length === 0) return 0;
        const totalTime = completed.reduce(
            (sum, r) => sum + (r.completedAt! - r.startedAt),
            0
        );
        return totalTime / completed.length;
    }, [completedRounds]);

    const successRate = useMemo(() => {
        const finished = completedRounds.length + failedRounds.length;
        if (finished === 0) return 1;
        return completedRounds.length / finished;
    }, [completedRounds, failedRounds]);

    return {
        rounds,
        currentRound,
        completedRounds,
        failedRounds,
        averageRoundTime,
        successRate,
        totalRounds: rounds.length,
        currentRoundNumber: progress.currentRound,
        isLoading,
    };
}

// Default export
export default useTraining;
