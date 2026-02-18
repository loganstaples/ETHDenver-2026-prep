'use client';

/**
 * HELIX Worker Health Hook
 * Real-time worker monitoring with health indicators, performance tracking,
 * and stake status integration.
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useWebSocket } from '@/lib/websocket';
import { useContractEvents } from './useContract';
import { formatEther } from 'viem';

// ============================================================================
// Types
// ============================================================================

export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy' | 'offline' | 'unknown';
export type WorkerRole = 'compute' | 'aggregator' | 'verifier';
export type WorkerActivity = 'idle' | 'training' | 'proving' | 'aggregating' | 'syncing';

export interface WorkerMetrics {
    cpu: number;
    memory: number;
    gpu?: number;
    gpuMemory?: number;
    networkIn: number;
    networkOut: number;
    diskUsage: number;
    temperature?: number;
    latency: number;
}

export interface WorkerCapabilities {
    gpuModel?: string;
    gpuMemoryMb: number;
    cudaVersion?: string;
    maxBatchSize: number;
    canTrain: boolean;
    canProve: boolean;
    canAggregate: boolean;
}

export interface WorkerPerformance {
    proofsSubmitted: number;
    proofsVerified: number;
    proofsFailed: number;
    roundsParticipated: number;
    averageProofTime: number;
    successRate: number;
    uptime: number;
    totalEarnings: bigint;
}

export interface WorkerStakeInfo {
    amount: bigint;
    amountFormatted: string;
    lockedUntil: number;
    isLocked: boolean;
    slashed: boolean;
}

export interface WorkerHealth {
    id: string;
    address: string;
    role: WorkerRole;
    status: HealthStatus;
    activity: WorkerActivity;
    lastSeen: number;
    lastHeartbeat: number;
    metrics: WorkerMetrics;
    capabilities: WorkerCapabilities;
    performance: WorkerPerformance;
    stake: WorkerStakeInfo;
    region?: string;
    issues: WorkerIssue[];
}

export interface WorkerIssue {
    id: string;
    severity: 'warning' | 'error' | 'critical';
    type: string;
    message: string;
    timestamp: number;
    resolved: boolean;
}

export interface NetworkHealthSummary {
    totalWorkers: number;
    healthyWorkers: number;
    degradedWorkers: number;
    unhealthyWorkers: number;
    offlineWorkers: number;
    totalStaked: bigint;
    totalStakedFormatted: string;
    averageLatency: number;
    averageUptime: number;
    networkHealth: HealthStatus;
    healthScore: number;
}

export interface UseWorkerHealthOptions {
    modelId?: bigint | number;
    workerAddress?: string;
    enableWebSocket?: boolean;
    enablePolling?: boolean;
    pollingInterval?: number;
    includeOffline?: boolean;
    onWorkerStatusChange?: (worker: WorkerHealth) => void;
    onWorkerIssue?: (worker: WorkerHealth, issue: WorkerIssue) => void;
}

export interface UseWorkerHealthReturn {
    // Workers
    workers: WorkerHealth[];
    selectedWorker: WorkerHealth | null;
    selectWorker: (workerId: string | null) => void;

    // By Role
    computeWorkers: WorkerHealth[];
    aggregators: WorkerHealth[];
    verifiers: WorkerHealth[];

    // Network Summary
    networkSummary: NetworkHealthSummary;

    // Issues
    activeIssues: WorkerIssue[];
    criticalIssues: WorkerIssue[];

    // Connection
    isConnected: boolean;
    isLoading: boolean;
    error: string | null;

    // Actions
    refresh: () => Promise<void>;
    getWorkerById: (workerId: string) => WorkerHealth | undefined;
    getWorkerByAddress: (address: string) => WorkerHealth | undefined;
    getWorkersByStatus: (status: HealthStatus) => WorkerHealth[];
    getWorkersByRole: (role: WorkerRole) => WorkerHealth[];
}

// ============================================================================
// Health Calculation Helpers
// ============================================================================

function calculateHealthStatus(worker: Omit<WorkerHealth, 'status' | 'issues'>): HealthStatus {
    const now = Date.now();
    const timeSinceLastSeen = now - worker.lastSeen;

    // Offline if not seen in 5 minutes
    if (timeSinceLastSeen > 300000) {
        return 'offline';
    }

    // Check for critical metrics
    const { metrics, stake, performance } = worker;

    // Slashed = unhealthy
    if (stake.slashed) {
        return 'unhealthy';
    }

    // High resource usage
    if (metrics.cpu > 95 || metrics.memory > 95) {
        return 'unhealthy';
    }

    if (metrics.gpu && metrics.gpu > 95) {
        return 'unhealthy';
    }

    // Temperature issues
    if (metrics.temperature && metrics.temperature > 85) {
        return 'unhealthy';
    }

    // Low success rate
    if (performance.proofsSubmitted > 10 && performance.successRate < 0.5) {
        return 'unhealthy';
    }

    // Check for degraded conditions
    if (
        metrics.cpu > 80 ||
        metrics.memory > 80 ||
        (metrics.gpu && metrics.gpu > 80) ||
        (metrics.temperature && metrics.temperature > 75) ||
        metrics.latency > 500 ||
        performance.successRate < 0.8
    ) {
        return 'degraded';
    }

    return 'healthy';
}

function detectWorkerIssues(worker: WorkerHealth): WorkerIssue[] {
    const issues: WorkerIssue[] = [];
    const now = Date.now();

    // CPU issues
    if (worker.metrics.cpu > 95) {
        issues.push({
            id: `${worker.id}-cpu-critical`,
            severity: 'critical',
            type: 'high_cpu',
            message: `CPU usage critical: ${worker.metrics.cpu.toFixed(1)}%`,
            timestamp: now,
            resolved: false,
        });
    } else if (worker.metrics.cpu > 80) {
        issues.push({
            id: `${worker.id}-cpu-warning`,
            severity: 'warning',
            type: 'high_cpu',
            message: `CPU usage high: ${worker.metrics.cpu.toFixed(1)}%`,
            timestamp: now,
            resolved: false,
        });
    }

    // Memory issues
    if (worker.metrics.memory > 95) {
        issues.push({
            id: `${worker.id}-memory-critical`,
            severity: 'critical',
            type: 'high_memory',
            message: `Memory usage critical: ${worker.metrics.memory.toFixed(1)}%`,
            timestamp: now,
            resolved: false,
        });
    } else if (worker.metrics.memory > 80) {
        issues.push({
            id: `${worker.id}-memory-warning`,
            severity: 'warning',
            type: 'high_memory',
            message: `Memory usage high: ${worker.metrics.memory.toFixed(1)}%`,
            timestamp: now,
            resolved: false,
        });
    }

    // GPU issues
    if (worker.metrics.gpu) {
        if (worker.metrics.gpu > 95) {
            issues.push({
                id: `${worker.id}-gpu-critical`,
                severity: 'critical',
                type: 'high_gpu',
                message: `GPU usage critical: ${worker.metrics.gpu.toFixed(1)}%`,
                timestamp: now,
                resolved: false,
            });
        }
    }

    // Temperature issues
    if (worker.metrics.temperature) {
        if (worker.metrics.temperature > 85) {
            issues.push({
                id: `${worker.id}-temp-critical`,
                severity: 'critical',
                type: 'high_temperature',
                message: `Temperature critical: ${worker.metrics.temperature.toFixed(0)}°C`,
                timestamp: now,
                resolved: false,
            });
        } else if (worker.metrics.temperature > 75) {
            issues.push({
                id: `${worker.id}-temp-warning`,
                severity: 'warning',
                type: 'high_temperature',
                message: `Temperature elevated: ${worker.metrics.temperature.toFixed(0)}°C`,
                timestamp: now,
                resolved: false,
            });
        }
    }

    // Latency issues
    if (worker.metrics.latency > 1000) {
        issues.push({
            id: `${worker.id}-latency-critical`,
            severity: 'error',
            type: 'high_latency',
            message: `Network latency critical: ${worker.metrics.latency.toFixed(0)}ms`,
            timestamp: now,
            resolved: false,
        });
    } else if (worker.metrics.latency > 500) {
        issues.push({
            id: `${worker.id}-latency-warning`,
            severity: 'warning',
            type: 'high_latency',
            message: `Network latency high: ${worker.metrics.latency.toFixed(0)}ms`,
            timestamp: now,
            resolved: false,
        });
    }

    // Stake issues
    if (worker.stake.slashed) {
        issues.push({
            id: `${worker.id}-slashed`,
            severity: 'critical',
            type: 'slashed',
            message: 'Worker has been slashed',
            timestamp: now,
            resolved: false,
        });
    }

    // Performance issues
    if (worker.performance.proofsSubmitted > 10 && worker.performance.successRate < 0.7) {
        issues.push({
            id: `${worker.id}-success-rate`,
            severity: 'error',
            type: 'low_success_rate',
            message: `Low proof success rate: ${(worker.performance.successRate * 100).toFixed(1)}%`,
            timestamp: now,
            resolved: false,
        });
    }

    // Heartbeat issues
    const timeSinceHeartbeat = now - worker.lastHeartbeat;
    if (timeSinceHeartbeat > 120000) {
        issues.push({
            id: `${worker.id}-heartbeat`,
            severity: 'warning',
            type: 'missed_heartbeat',
            message: `No heartbeat for ${Math.floor(timeSinceHeartbeat / 1000)}s`,
            timestamp: now,
            resolved: false,
        });
    }

    return issues;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useWorkerHealth(options: UseWorkerHealthOptions = {}): UseWorkerHealthReturn {
    const {
        modelId,
        workerAddress: _workerAddress,
        enableWebSocket = true,
        enablePolling = true,
        pollingInterval = 3000,
        includeOffline = true,
        onWorkerStatusChange,
        onWorkerIssue,
    } = options;

    const modelIdBigInt = modelId !== undefined
        ? (typeof modelId === 'number' ? BigInt(modelId) : modelId)
        : BigInt(1);

    // State
    const [workers, setWorkers] = useState<WorkerHealth[]>([]);
    const [selectedWorkerId, setSelectedWorkerId] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Refs
    const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const previousStatusRef = useRef<Map<string, HealthStatus>>(new Map());

    // Contract
    const { proofEvents, stakedEvents, slashedEvents } = useContractEvents(modelIdBigInt);

    // WebSocket
    const { isConnected, on } = useWebSocket({
        autoConnect: enableWebSocket,
        channels: enableWebSocket ? ['workers', `workers:model:${modelIdBigInt.toString()}`] : [],
    });

    // ========================================================================
    // Initialize
    // ========================================================================

    const initializeWorkers = useCallback(async () => {
        try {
            setIsLoading(true);
            setError(null);

            // Build workers from contract events
            const workerMap = new Map<string, Partial<WorkerHealth>>();

            // Process staked events
            stakedEvents.forEach(event => {
                const roles: WorkerRole[] = ['compute', 'aggregator', 'verifier'];
                workerMap.set(event.prover, {
                    id: `worker-${event.prover.slice(2, 10)}`,
                    address: event.prover,
                    role: roles[workerMap.size % 3],
                    activity: 'idle',
                    lastSeen: event.timestamp,
                    lastHeartbeat: event.timestamp,
                    metrics: {
                        cpu: 0,
                        memory: 0,
                        gpu: undefined,
                        gpuMemory: undefined,
                        networkIn: 0,
                        networkOut: 0,
                        diskUsage: 0,
                        temperature: undefined,
                        latency: 0,
                    },
                    capabilities: {
                        gpuModel: undefined,
                        gpuMemoryMb: 0,
                        cudaVersion: undefined,
                        maxBatchSize: 64,
                        canTrain: true,
                        canProve: workerMap.size % 3 === 2,
                        canAggregate: workerMap.size % 3 === 1,
                    },
                    performance: {
                        proofsSubmitted: 0,
                        proofsVerified: 0,
                        proofsFailed: 0,
                        roundsParticipated: 0,
                        averageProofTime: 0,
                        successRate: 1,
                        uptime: 1,
                        totalEarnings: BigInt(0),
                    },
                    stake: {
                        amount: event.amount,
                        amountFormatted: formatEther(event.amount),
                        lockedUntil: 0,
                        isLocked: false,
                        slashed: false,
                    },
                    region: undefined,
                });
            });

            // Process proof events
            proofEvents.forEach(event => {
                const worker = workerMap.get(event.prover);
                if (worker && worker.performance) {
                    worker.performance.proofsSubmitted++;
                    worker.performance.proofsVerified++;
                    worker.lastSeen = Math.max(worker.lastSeen || 0, event.timestamp);
                    worker.lastHeartbeat = event.timestamp;
                    worker.activity = 'idle';
                }
            });

            // Process slashed events
            slashedEvents.forEach(event => {
                const worker = workerMap.get(event.prover);
                if (worker && worker.stake && worker.performance) {
                    worker.stake.slashed = true;
                    worker.performance.proofsFailed++;
                }
            });

            // Finalize workers with health status and issues
            const finalWorkers: WorkerHealth[] = Array.from(workerMap.values()).map(w => {
                const partialWorker = w as Omit<WorkerHealth, 'status' | 'issues'>;
                const status = calculateHealthStatus(partialWorker);
                const worker: WorkerHealth = {
                    ...partialWorker,
                    status,
                    issues: [],
                } as WorkerHealth;
                worker.issues = detectWorkerIssues(worker);
                return worker;
            });

            setWorkers(finalWorkers);

        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load workers');
        } finally {
            setIsLoading(false);
        }
    }, [stakedEvents, proofEvents, slashedEvents]);

    // ========================================================================
    // WebSocket Handlers
    // ========================================================================

    useEffect(() => {
        if (!enableWebSocket) return;

        const unsubMetrics = on<{ address: string; metrics: WorkerMetrics }>('worker_metrics', (msg) => {
            setWorkers(prev => prev.map(w => {
                if (w.address === msg.data.address) {
                    const updated = { ...w, metrics: msg.data.metrics, lastHeartbeat: Date.now() };
                    const newStatus = calculateHealthStatus(updated);
                    const newIssues = detectWorkerIssues({ ...updated, status: newStatus, issues: [] });

                    // Check for status changes
                    const prevStatus = previousStatusRef.current.get(w.id);
                    if (prevStatus && prevStatus !== newStatus) {
                        onWorkerStatusChange?.({ ...updated, status: newStatus, issues: newIssues });
                    }
                    previousStatusRef.current.set(w.id, newStatus);

                    // Check for new critical issues
                    const newCritical = newIssues.filter(
                        i => i.severity === 'critical' && !w.issues.some(oi => oi.id === i.id)
                    );
                    newCritical.forEach(issue => {
                        onWorkerIssue?.({ ...updated, status: newStatus, issues: newIssues }, issue);
                    });

                    return { ...updated, status: newStatus, issues: newIssues };
                }
                return w;
            }));
        });

        const unsubActivity = on<{ address: string; activity: WorkerActivity }>('worker_activity', (msg) => {
            setWorkers(prev => prev.map(w =>
                w.address === msg.data.address
                    ? { ...w, activity: msg.data.activity, lastSeen: Date.now() }
                    : w
            ));
        });

        return () => {
            unsubMetrics();
            unsubActivity();
        };
    }, [enableWebSocket, on, onWorkerStatusChange, onWorkerIssue]);

    // ========================================================================
    // Polling
    // ========================================================================

    useEffect(() => {
        if (!enablePolling) return;
        // Real metrics come from WebSocket worker_metrics/worker_activity messages
        return undefined;
    }, [enablePolling]);

    // ========================================================================
    // Effects
    // ========================================================================

    useEffect(() => {
        initializeWorkers();
    }, [initializeWorkers]);

    // ========================================================================
    // Computed Values
    // ========================================================================

    const selectedWorker = useMemo(
        () => workers.find(w => w.id === selectedWorkerId) || null,
        [workers, selectedWorkerId]
    );

    const computeWorkers = useMemo(
        () => workers.filter(w => w.role === 'compute'),
        [workers]
    );

    const aggregators = useMemo(
        () => workers.filter(w => w.role === 'aggregator'),
        [workers]
    );

    const verifiers = useMemo(
        () => workers.filter(w => w.role === 'verifier'),
        [workers]
    );

    const filteredWorkers = useMemo(
        () => includeOffline ? workers : workers.filter(w => w.status !== 'offline'),
        [workers, includeOffline]
    );

    const activeIssues = useMemo(
        () => workers.flatMap(w => w.issues.filter(i => !i.resolved)),
        [workers]
    );

    const criticalIssues = useMemo(
        () => activeIssues.filter(i => i.severity === 'critical'),
        [activeIssues]
    );

    const networkSummary = useMemo((): NetworkHealthSummary => {
        const healthy = filteredWorkers.filter(w => w.status === 'healthy');
        const degraded = filteredWorkers.filter(w => w.status === 'degraded');
        const unhealthy = filteredWorkers.filter(w => w.status === 'unhealthy');
        const offline = workers.filter(w => w.status === 'offline');

        const totalStaked = filteredWorkers.reduce((sum, w) => sum + w.stake.amount, BigInt(0));
        const avgLatency = filteredWorkers.length > 0
            ? filteredWorkers.reduce((sum, w) => sum + w.metrics.latency, 0) / filteredWorkers.length
            : 0;
        const avgUptime = filteredWorkers.length > 0
            ? filteredWorkers.reduce((sum, w) => sum + w.performance.uptime, 0) / filteredWorkers.length
            : 0;

        // Calculate health score
        const healthFactors = [
            (healthy.length / Math.max(filteredWorkers.length, 1)) * 40,
            avgUptime * 30,
            (1 - Math.min(avgLatency / 500, 1)) * 15,
            (1 - criticalIssues.length / Math.max(filteredWorkers.length, 1)) * 15,
        ];
        const healthScore = Math.round(healthFactors.reduce((s, f) => s + f, 0));

        let networkHealth: HealthStatus = 'healthy';
        if (healthScore < 40) networkHealth = 'unhealthy';
        else if (healthScore < 60) networkHealth = 'degraded';
        else if (unhealthy.length > 0 || criticalIssues.length > 0) networkHealth = 'degraded';

        return {
            totalWorkers: filteredWorkers.length,
            healthyWorkers: healthy.length,
            degradedWorkers: degraded.length,
            unhealthyWorkers: unhealthy.length,
            offlineWorkers: offline.length,
            totalStaked,
            totalStakedFormatted: formatEther(totalStaked),
            averageLatency: avgLatency,
            averageUptime: avgUptime,
            networkHealth,
            healthScore,
        };
    }, [filteredWorkers, workers, criticalIssues]);

    // ========================================================================
    // Actions
    // ========================================================================

    const selectWorker = useCallback((workerId: string | null) => {
        setSelectedWorkerId(workerId);
    }, []);

    const refresh = useCallback(async () => {
        await initializeWorkers();
    }, [initializeWorkers]);

    const getWorkerById = useCallback(
        (workerId: string) => workers.find(w => w.id === workerId),
        [workers]
    );

    const getWorkerByAddress = useCallback(
        (address: string) => workers.find(w => w.address.toLowerCase() === address.toLowerCase()),
        [workers]
    );

    const getWorkersByStatus = useCallback(
        (status: HealthStatus) => workers.filter(w => w.status === status),
        [workers]
    );

    const getWorkersByRole = useCallback(
        (role: WorkerRole) => workers.filter(w => w.role === role),
        [workers]
    );

    // ========================================================================
    // Return
    // ========================================================================

    return {
        workers: filteredWorkers,
        selectedWorker,
        selectWorker,
        computeWorkers,
        aggregators,
        verifiers,
        networkSummary,
        activeIssues,
        criticalIssues,
        isConnected,
        isLoading,
        error,
        refresh,
        getWorkerById,
        getWorkerByAddress,
        getWorkersByStatus,
        getWorkersByRole,
    };
}

export default useWorkerHealth;
