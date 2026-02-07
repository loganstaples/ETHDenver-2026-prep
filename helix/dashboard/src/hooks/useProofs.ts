'use client';

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useContractEvents } from './useContract';
import {
    type ProofInfo,
    type ProofGenerationProgress,
    getApiClient,
} from '@/lib/api';

// ============================================================================
// Types
// ============================================================================

export interface ProofData {
    id: string;
    hash: string;
    modelId: bigint;
    roundId: bigint;
    prover: string;
    type: 'training' | 'aggregation' | 'gradient' | 'computation' | 'verification';
    status: 'generating' | 'pending' | 'verified' | 'failed' | 'challenged';
    createdAt: number;
    verifiedAt?: number;
    size: number;
    generationTime: number;
    verificationTime?: number;
    errorBound: number;
    publicInputsHash: string;
    commitment: bigint;
    gasUsed?: bigint;
    transactionHash?: string;
    blockNumber?: number;
    circuitType: string;
    constraintCount: number;
}

export interface ProofTimelineEvent {
    id: string;
    proofId: string;
    timestamp: number;
    type: 'started' | 'witness_generated' | 'setup_complete' | 'proving' | 'proof_generated' | 'submitted' | 'verified' | 'failed' | 'challenged';
    details: Record<string, unknown>;
    duration?: number;
}

export interface ProofStats {
    total: number;
    verified: number;
    pending: number;
    generating: number;
    failed: number;
    challenged: number;
    averageGenerationTime: number;
    averageVerificationTime: number;
    totalSize: number;
    averageErrorBound: number;
    successRate: number;
}

export interface ProofFilter {
    type?: ProofData['type'];
    status?: ProofData['status'];
    prover?: string;
    modelId?: bigint;
    roundId?: bigint;
    minErrorBound?: number;
    maxErrorBound?: number;
    dateRange?: { start: number; end: number };
}

export interface UseProofsOptions {
    modelId?: bigint | number;
    roundId?: bigint | number;
    autoRefresh?: boolean;
    refreshInterval?: number;
    enableWebSocket?: boolean;
    maxProofs?: number;
    filter?: ProofFilter;
}

export interface UseProofsReturn {
    proofs: ProofData[];
    filteredProofs: ProofData[];
    selectedProof: ProofData | null;
    selectProof: (proofId: string | null) => void;
    stats: ProofStats;
    timeline: ProofTimelineEvent[];
    activeGenerations: ProofGenerationProgress[];
    isLoading: boolean;
    isConnected: boolean;
    error: string | null;
    refetch: () => Promise<void>;
    getProofById: (proofId: string) => ProofData | undefined;
    getProofsByType: (type: ProofData['type']) => ProofData[];
    getProofsByStatus: (status: ProofData['status']) => ProofData[];
    getProofsByRound: (roundId: bigint) => ProofData[];
    updateFilter: (filter: Partial<ProofFilter>) => void;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useProofs(options: UseProofsOptions = {}): UseProofsReturn {
    const {
        modelId,
        roundId,
        autoRefresh = true,
        refreshInterval = 3000,
        enableWebSocket = true,
        maxProofs = 100,
        filter: initialFilter = {},
    } = options;

    // State
    const [proofs, setProofs] = useState<ProofData[]>([]);
    const [timeline, setTimeline] = useState<ProofTimelineEvent[]>([]);
    const [activeGenerations, setActiveGenerations] = useState<ProofGenerationProgress[]>([]);
    const [selectedProofId, setSelectedProofId] = useState<string | null>(null);
    const [filter, setFilter] = useState<ProofFilter>(initialFilter);
    const [isLoading, setIsLoading] = useState(true);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Refs
    const wsUnsubscribeRef = useRef<(() => void) | null>(null);
    const refreshIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const proofIdCounterRef = useRef(0);

    // Contract data
    const { proofEvents } = useContractEvents(modelId ? BigInt(modelId) : undefined);

    // API client
    const apiClient = useMemo(() => getApiClient(), []);

    // ========================================================================
    // Data Generation
    // ========================================================================

    const generateDemoProofs = useCallback((): ProofData[] => {
        const types: ProofData['type'][] = ['training', 'aggregation', 'gradient', 'computation', 'verification'];
        const statuses: ProofData['status'][] = ['verified', 'verified', 'verified', 'pending', 'generating', 'failed'];
        const circuitTypes = ['nova_folding', 'groth16', 'plonk', 'stark'];

        return Array.from({ length: 20 }, (_, index) => {
            const status = statuses[Math.floor(Math.random() * statuses.length)];
            const createdAt = Date.now() - Math.random() * 3600000;
            const generationTime = 100 + Math.random() * 400;

            return {
                id: `proof-${Date.now()}-${index}`,
                hash: `0x${Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
                modelId: modelId ? BigInt(modelId) : BigInt(1),
                roundId: roundId ? BigInt(roundId) : BigInt(Math.floor(Math.random() * 50) + 1),
                prover: `0x${Array(40).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
                type: types[index % types.length],
                status,
                createdAt,
                verifiedAt: status === 'verified' ? createdAt + generationTime + Math.random() * 100 : undefined,
                size: 1024 + Math.floor(Math.random() * 4096),
                generationTime,
                verificationTime: status === 'verified' ? 20 + Math.random() * 80 : undefined,
                errorBound: Math.random() * 0.001,
                publicInputsHash: `0x${Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
                commitment: BigInt(Math.floor(Math.random() * 1e18)),
                gasUsed: status === 'verified' ? BigInt(Math.floor(100000 + Math.random() * 400000)) : undefined,
                transactionHash: status === 'verified' ? `0x${Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}` : undefined,
                blockNumber: status === 'verified' ? Math.floor(18000000 + Math.random() * 1000000) : undefined,
                circuitType: circuitTypes[Math.floor(Math.random() * circuitTypes.length)],
                constraintCount: 10000 + Math.floor(Math.random() * 100000),
            };
        });
    }, [modelId, roundId]);

    const generateDemoTimeline = useCallback((proofList: ProofData[]): ProofTimelineEvent[] => {
        const events: ProofTimelineEvent[] = [];

        proofList.forEach((proof) => {
            const baseTime = proof.createdAt;

            events.push({
                id: `event-${proof.id}-started`,
                proofId: proof.id,
                timestamp: baseTime,
                type: 'started',
                details: { circuitType: proof.circuitType, constraintCount: proof.constraintCount },
            });

            if (proof.status !== 'generating') {
                events.push({
                    id: `event-${proof.id}-witness`,
                    proofId: proof.id,
                    timestamp: baseTime + proof.generationTime * 0.2,
                    type: 'witness_generated',
                    details: { witnessSize: Math.floor(Math.random() * 1000000) },
                    duration: proof.generationTime * 0.2,
                });

                events.push({
                    id: `event-${proof.id}-proving`,
                    proofId: proof.id,
                    timestamp: baseTime + proof.generationTime * 0.3,
                    type: 'proving',
                    details: { progress: 100 },
                    duration: proof.generationTime * 0.7,
                });

                events.push({
                    id: `event-${proof.id}-generated`,
                    proofId: proof.id,
                    timestamp: baseTime + proof.generationTime,
                    type: 'proof_generated',
                    details: { size: proof.size, errorBound: proof.errorBound },
                    duration: proof.generationTime,
                });
            }

            if (proof.status === 'verified' && proof.verifiedAt) {
                events.push({
                    id: `event-${proof.id}-submitted`,
                    proofId: proof.id,
                    timestamp: baseTime + proof.generationTime + 10,
                    type: 'submitted',
                    details: { transactionHash: proof.transactionHash },
                });

                events.push({
                    id: `event-${proof.id}-verified`,
                    proofId: proof.id,
                    timestamp: proof.verifiedAt,
                    type: 'verified',
                    details: { gasUsed: proof.gasUsed?.toString(), blockNumber: proof.blockNumber },
                    duration: proof.verificationTime,
                });
            }

            if (proof.status === 'failed') {
                events.push({
                    id: `event-${proof.id}-failed`,
                    proofId: proof.id,
                    timestamp: baseTime + proof.generationTime,
                    type: 'failed',
                    details: { reason: 'Constraint verification failed', errorCode: 'INVALID_WITNESS' },
                });
            }

            if (proof.status === 'challenged') {
                events.push({
                    id: `event-${proof.id}-challenged`,
                    proofId: proof.id,
                    timestamp: baseTime + proof.generationTime + 500,
                    type: 'challenged',
                    details: { challenger: `0x${Array(40).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}` },
                });
            }
        });

        return events.sort((a, b) => b.timestamp - a.timestamp);
    }, []);

    const generateActiveGenerations = useCallback((): ProofGenerationProgress[] => {
        return Array.from({ length: 3 }, (_, index) => {
            const stages: ProofGenerationProgress['stage'][] = ['witness', 'setup', 'proving', 'verifying'];
            const stage = stages[Math.floor(Math.random() * stages.length)];
            const progress = Math.random() * 100;

            return {
                proofId: `proof-active-${index}`,
                stage,
                progress,
                currentStep: {
                    witness: 'Computing witness vectors',
                    setup: 'Preparing circuit setup',
                    proving: 'Generating ZK proof',
                    verifying: 'Verifying constraints',
                    complete: 'Complete',
                    failed: 'Failed',
                }[stage],
                totalSteps: 4,
                estimatedTimeRemaining: Math.floor((100 - progress) * 5),
                memoryUsage: 2000 + Math.random() * 6000,
                cpuUsage: 60 + Math.random() * 35,
            };
        });
    }, []);

    // ========================================================================
    // Data Fetching
    // ========================================================================

    const fetchProofs = useCallback(async () => {
        try {
            setIsLoading(true);
            setError(null);

            // Convert contract events to proofs
            const eventProofs: ProofData[] = proofEvents.map((event, index) => ({
                id: `proof-${event.transactionHash}-${index}`,
                hash: event.transactionHash,
                modelId: event.modelId,
                roundId: event.roundId,
                prover: event.prover,
                type: 'training' as const,
                status: 'verified' as const,
                createdAt: event.timestamp,
                verifiedAt: event.timestamp,
                size: 2048,
                generationTime: 250,
                verificationTime: 50,
                errorBound: 0.0001,
                publicInputsHash: `0x${event.newCommitment.toString(16).padStart(64, '0')}`,
                commitment: event.newCommitment,
                gasUsed: BigInt(200000),
                transactionHash: event.transactionHash,
                blockNumber: event.blockNumber,
                circuitType: 'nova_folding',
                constraintCount: 50000,
            }));

            if (eventProofs.length > 0) {
                setProofs(eventProofs.slice(0, maxProofs));
                setTimeline(generateDemoTimeline(eventProofs));
            } else {
                // Use demo data
                const demoProofs = generateDemoProofs();
                setProofs(demoProofs);
                setTimeline(generateDemoTimeline(demoProofs));
            }

            setActiveGenerations(generateActiveGenerations());
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch proofs');
        } finally {
            setIsLoading(false);
        }
    }, [proofEvents, maxProofs, generateDemoProofs, generateDemoTimeline, generateActiveGenerations]);

    // ========================================================================
    // WebSocket Setup
    // ========================================================================

    const setupWebSocket = useCallback(async () => {
        if (!enableWebSocket) return;

        try {
            await apiClient.connectWebSocket();
            setIsConnected(true);

            apiClient.subscribeToChannel('proofs');
            if (modelId) {
                apiClient.subscribeToChannel(`proofs:model:${modelId}`);
            }

            wsUnsubscribeRef.current = apiClient.onWebSocketMessage<ProofInfo>(
                'proof_update',
                (message) => {
                    const proofData = message.data;
                    setProofs((prevProofs) => {
                        const existingIndex = prevProofs.findIndex((p) => p.id === proofData.id);

                        const updatedProof: ProofData = {
                            ...proofData,
                        };

                        if (existingIndex >= 0) {
                            const newProofs = [...prevProofs];
                            newProofs[existingIndex] = updatedProof;
                            return newProofs;
                        } else {
                            return [updatedProof, ...prevProofs].slice(0, maxProofs);
                        }
                    });

                    // Add timeline event
                    const eventType: ProofTimelineEvent['type'] = proofData.status === 'verified' ? 'verified' : proofData.status === 'failed' ? 'failed' : 'proof_generated';
                    setTimeline((prev) => [{
                        id: `event-${proofData.id}-${Date.now()}`,
                        proofId: proofData.id,
                        timestamp: Date.now(),
                        type: eventType,
                        details: { status: proofData.status },
                    }, ...prev].slice(0, 200));
                }
            );
        } catch (err) {
            console.warn('WebSocket connection failed:', err);
            setIsConnected(false);
        }
    }, [enableWebSocket, apiClient, modelId, maxProofs]);

    // ========================================================================
    // Effects
    // ========================================================================

    useEffect(() => {
        fetchProofs();
        setupWebSocket();

        return () => {
            wsUnsubscribeRef.current?.();
        };
    }, [fetchProofs, setupWebSocket]);

    // Auto refresh with simulated progress updates
    useEffect(() => {
        if (!autoRefresh) return;

        refreshIntervalRef.current = setInterval(() => {
            // Update active generations progress
            setActiveGenerations((prev) =>
                prev.map((gen) => {
                    const newProgress = Math.min(100, gen.progress + Math.random() * 10);
                    const stages: ProofGenerationProgress['stage'][] = ['witness', 'setup', 'proving', 'verifying', 'complete'];
                    let newStage = gen.stage;

                    if (newProgress >= 100) {
                        const currentIndex = stages.indexOf(gen.stage);
                        if (currentIndex < stages.length - 1) {
                            newStage = stages[currentIndex + 1];
                            return {
                                ...gen,
                                stage: newStage,
                                progress: newStage === 'complete' ? 100 : 0,
                                currentStep: {
                                    witness: 'Computing witness vectors',
                                    setup: 'Preparing circuit setup',
                                    proving: 'Generating ZK proof',
                                    verifying: 'Verifying constraints',
                                    complete: 'Complete',
                                    failed: 'Failed',
                                }[newStage],
                                estimatedTimeRemaining: newStage === 'complete' ? 0 : Math.floor(Math.random() * 200),
                            };
                        }
                    }

                    return {
                        ...gen,
                        progress: newProgress,
                        estimatedTimeRemaining: Math.max(0, Math.floor((100 - newProgress) * 5)),
                        memoryUsage: gen.memoryUsage + (Math.random() - 0.5) * 500,
                        cpuUsage: Math.min(99, Math.max(40, gen.cpuUsage + (Math.random() - 0.5) * 10)),
                    };
                }).filter((gen) => gen.stage !== 'complete' || Math.random() > 0.3)
            );

            // Occasionally add new generating proof
            if (Math.random() > 0.8) {
                proofIdCounterRef.current++;
                setActiveGenerations((prev) => [
                    ...prev,
                    {
                        proofId: `proof-new-${proofIdCounterRef.current}`,
                        stage: 'witness' as const,
                        progress: 0,
                        currentStep: 'Computing witness vectors',
                        totalSteps: 4,
                        estimatedTimeRemaining: 500,
                        memoryUsage: 2000 + Math.random() * 2000,
                        cpuUsage: 60 + Math.random() * 20,
                    },
                ].slice(0, 5));
            }

            // Simulate pending proofs becoming verified
            setProofs((prev) =>
                prev.map((proof) => {
                    if (proof.status === 'pending' && Math.random() > 0.9) {
                        return {
                            ...proof,
                            status: 'verified' as const,
                            verifiedAt: Date.now(),
                            verificationTime: 20 + Math.random() * 80,
                            gasUsed: BigInt(Math.floor(100000 + Math.random() * 400000)),
                            transactionHash: `0x${Array(64).fill(0).map(() => Math.floor(Math.random() * 16).toString(16)).join('')}`,
                            blockNumber: Math.floor(18000000 + Math.random() * 1000000),
                        };
                    }
                    if (proof.status === 'generating' && Math.random() > 0.7) {
                        return {
                            ...proof,
                            status: 'pending' as const,
                        };
                    }
                    return proof;
                })
            );
        }, refreshInterval);

        return () => {
            if (refreshIntervalRef.current) {
                clearInterval(refreshIntervalRef.current);
            }
        };
    }, [autoRefresh, refreshInterval]);

    // Update when contract events change
    useEffect(() => {
        if (proofEvents.length > 0) {
            fetchProofs();
        }
    }, [proofEvents, fetchProofs]);

    // ========================================================================
    // Computed Values
    // ========================================================================

    const filteredProofs = useMemo(() => {
        let result = proofs;

        if (filter.type) {
            result = result.filter((p) => p.type === filter.type);
        }

        if (filter.status) {
            result = result.filter((p) => p.status === filter.status);
        }

        if (filter.prover) {
            result = result.filter((p) => p.prover.toLowerCase() === filter.prover!.toLowerCase());
        }

        if (filter.modelId !== undefined) {
            result = result.filter((p) => p.modelId === filter.modelId);
        }

        if (filter.roundId !== undefined) {
            result = result.filter((p) => p.roundId === filter.roundId);
        }

        if (filter.minErrorBound !== undefined) {
            result = result.filter((p) => p.errorBound >= filter.minErrorBound!);
        }

        if (filter.maxErrorBound !== undefined) {
            result = result.filter((p) => p.errorBound <= filter.maxErrorBound!);
        }

        if (filter.dateRange) {
            result = result.filter(
                (p) => p.createdAt >= filter.dateRange!.start && p.createdAt <= filter.dateRange!.end
            );
        }

        return result;
    }, [proofs, filter]);

    const selectedProof = useMemo(
        () => proofs.find((p) => p.id === selectedProofId) || null,
        [proofs, selectedProofId]
    );

    const stats = useMemo((): ProofStats => {
        const verified = proofs.filter((p) => p.status === 'verified');
        const pending = proofs.filter((p) => p.status === 'pending');
        const generating = proofs.filter((p) => p.status === 'generating');
        const failed = proofs.filter((p) => p.status === 'failed');
        const challenged = proofs.filter((p) => p.status === 'challenged');

        const avgGenTime = proofs.length > 0
            ? proofs.reduce((sum, p) => sum + p.generationTime, 0) / proofs.length
            : 0;

        const verifiedWithTime = verified.filter((p) => p.verificationTime !== undefined);
        const avgVerTime = verifiedWithTime.length > 0
            ? verifiedWithTime.reduce((sum, p) => sum + (p.verificationTime || 0), 0) / verifiedWithTime.length
            : 0;

        const totalSize = proofs.reduce((sum, p) => sum + p.size, 0);
        const avgErrorBound = proofs.length > 0
            ? proofs.reduce((sum, p) => sum + p.errorBound, 0) / proofs.length
            : 0;

        const completed = verified.length + failed.length;
        const successRate = completed > 0 ? verified.length / completed : 0;

        return {
            total: proofs.length,
            verified: verified.length,
            pending: pending.length,
            generating: generating.length,
            failed: failed.length,
            challenged: challenged.length,
            averageGenerationTime: avgGenTime,
            averageVerificationTime: avgVerTime,
            totalSize,
            averageErrorBound: avgErrorBound,
            successRate,
        };
    }, [proofs]);

    // ========================================================================
    // Actions
    // ========================================================================

    const selectProof = useCallback((proofId: string | null) => {
        setSelectedProofId(proofId);
    }, []);

    const getProofById = useCallback(
        (proofId: string) => proofs.find((p) => p.id === proofId),
        [proofs]
    );

    const getProofsByType = useCallback(
        (type: ProofData['type']) => proofs.filter((p) => p.type === type),
        [proofs]
    );

    const getProofsByStatus = useCallback(
        (status: ProofData['status']) => proofs.filter((p) => p.status === status),
        [proofs]
    );

    const getProofsByRound = useCallback(
        (roundId: bigint) => proofs.filter((p) => p.roundId === roundId),
        [proofs]
    );

    const updateFilter = useCallback((newFilter: Partial<ProofFilter>) => {
        setFilter((prev) => ({ ...prev, ...newFilter }));
    }, []);

    // ========================================================================
    // Return
    // ========================================================================

    return {
        proofs,
        filteredProofs,
        selectedProof,
        selectProof,
        stats,
        timeline,
        activeGenerations,
        isLoading,
        isConnected,
        error,
        refetch: fetchProofs,
        getProofById,
        getProofsByType,
        getProofsByStatus,
        getProofsByRound,
        updateFilter,
    };
}

// ============================================================================
// Additional Proof Hooks
// ============================================================================

/**
 * Hook for tracking a specific proof's generation progress
 */
export function useProofGeneration(proofId: string) {
    const [progress, setProgress] = useState<ProofGenerationProgress | null>(null);
    const [isComplete, setIsComplete] = useState(false);
    const [error, _setError] = useState<string | null>(null);

    useEffect(() => {
        const stages: ProofGenerationProgress['stage'][] = ['witness', 'setup', 'proving', 'verifying'];
        let currentStageIndex = 0;
        let currentProgress = 0;

        const interval = setInterval(() => {
            currentProgress += Math.random() * 15;

            if (currentProgress >= 100) {
                currentStageIndex++;
                if (currentStageIndex >= stages.length) {
                    setProgress({
                        proofId,
                        stage: 'complete',
                        progress: 100,
                        currentStep: 'Proof generation complete',
                        totalSteps: 4,
                        estimatedTimeRemaining: 0,
                        memoryUsage: 0,
                        cpuUsage: 0,
                    });
                    setIsComplete(true);
                    clearInterval(interval);
                    return;
                }
                currentProgress = 0;
            }

            const stepDescriptions: Record<string, string> = {
                witness: 'Computing witness vectors',
                setup: 'Preparing circuit setup',
                proving: 'Generating ZK proof',
                verifying: 'Verifying constraints',
            };
            setProgress({
                proofId,
                stage: stages[currentStageIndex],
                progress: currentProgress,
                currentStep: stepDescriptions[stages[currentStageIndex]] || 'Processing',
                totalSteps: 4,
                estimatedTimeRemaining: Math.floor((4 - currentStageIndex) * 100 + (100 - currentProgress) * 2),
                memoryUsage: 4000 + Math.random() * 4000,
                cpuUsage: 70 + Math.random() * 25,
            });
        }, 500);

        return () => clearInterval(interval);
    }, [proofId]);

    return { progress, isComplete, error };
}

/**
 * Hook for proof verification monitoring
 */
export function useProofVerification(proofId: string) {
    const { proofs, isLoading } = useProofs();

    const proof = useMemo(
        () => proofs.find((p) => p.id === proofId),
        [proofs, proofId]
    );

    const verificationStatus = useMemo(() => {
        if (!proof) return 'unknown';
        return proof.status;
    }, [proof]);

    const isVerified = verificationStatus === 'verified';
    const isFailed = verificationStatus === 'failed';
    const isPending = verificationStatus === 'pending' || verificationStatus === 'generating';

    return {
        proof,
        verificationStatus,
        isVerified,
        isFailed,
        isPending,
        isLoading,
    };
}

/**
 * Hook for proof timeline visualization
 */
export function useProofTimeline(modelId?: bigint, roundId?: bigint) {
    const { timeline, isLoading, error } = useProofs({
        modelId: modelId ? Number(modelId) : undefined,
        roundId: roundId ? Number(roundId) : undefined,
    });

    const groupedTimeline = useMemo(() => {
        const groups = new Map<string, ProofTimelineEvent[]>();

        timeline.forEach((event) => {
            const dateKey = new Date(event.timestamp).toDateString();
            if (!groups.has(dateKey)) {
                groups.set(dateKey, []);
            }
            groups.get(dateKey)!.push(event);
        });

        return Array.from(groups.entries()).map(([date, events]) => ({
            date,
            events: events.sort((a, b) => b.timestamp - a.timestamp),
        }));
    }, [timeline]);

    const recentEvents = useMemo(
        () => timeline.slice(0, 10),
        [timeline]
    );

    return {
        timeline,
        groupedTimeline,
        recentEvents,
        isLoading,
        error,
    };
}

// Default export
export default useProofs;
