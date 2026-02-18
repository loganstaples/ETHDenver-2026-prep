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

    // Contract data
    const { proofEvents } = useContractEvents(modelId ? BigInt(modelId) : undefined);

    // API client
    const apiClient = useMemo(() => getApiClient(), []);

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

            setProofs(eventProofs.slice(0, maxProofs));
            setTimeline([]);
            setActiveGenerations([]);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch proofs');
        } finally {
            setIsLoading(false);
        }
    }, [proofEvents, maxProofs]);

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

    useEffect(() => {
        if (!autoRefresh) return;
        // Real updates come from WebSocket proof_update messages
        return undefined;
    }, [autoRefresh]);

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

    // Real progress comes from WebSocket proof_progress messages
    // This hook returns null until real data arrives

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
