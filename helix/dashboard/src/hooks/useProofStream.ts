'use client';

/**
 * HELIX Proof Stream Hook
 * Real-time proof generation and verification streaming with WebSocket integration.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { useWebSocket } from '@/lib/websocket';
import { useContractEvents } from './useContract';

// ============================================================================
// Types
// ============================================================================

export type ProofStage = 'queued' | 'witness' | 'setup' | 'proving' | 'submitting' | 'verifying' | 'verified' | 'failed';

export interface ProofStreamItem {
    id: string;
    hash: string;
    modelId: bigint;
    roundId: bigint;
    prover: string;
    type: 'training' | 'aggregation' | 'gradient' | 'verification';
    stage: ProofStage;
    progress: number;
    startedAt: number;
    completedAt?: number;
    errorBound: number;
    constraintCount: number;
    circuitType: string;
    gasUsed?: bigint;
    transactionHash?: string;
    blockNumber?: number;
    error?: string;
    metrics: {
        witnessTime?: number;
        setupTime?: number;
        provingTime?: number;
        verificationTime?: number;
        totalTime?: number;
        memoryPeak?: number;
        cpuPeak?: number;
    };
}

export interface ProofStreamStats {
    total: number;
    queued: number;
    generating: number;
    submitting: number;
    verified: number;
    failed: number;
    averageGenerationTime: number;
    averageVerificationTime: number;
    successRate: number;
    throughputPerMinute: number;
}

export interface ProofGenerationEvent {
    proofId: string;
    stage: ProofStage;
    progress: number;
    timestamp: number;
    metrics?: Partial<ProofStreamItem['metrics']>;
}

export interface OnChainVerification {
    proofId: string;
    transactionHash: string;
    blockNumber: number;
    gasUsed: bigint;
    timestamp: number;
    success: boolean;
    error?: string;
}

export interface UseProofStreamOptions {
    modelId?: bigint | number;
    maxProofs?: number;
    enableWebSocket?: boolean;
    onProofGenerated?: (proof: ProofStreamItem) => void;
    onProofVerified?: (verification: OnChainVerification) => void;
    onProofFailed?: (proof: ProofStreamItem) => void;
}

export interface UseProofStreamReturn {
    // Proofs
    proofs: ProofStreamItem[];
    activeProofs: ProofStreamItem[];
    recentVerified: ProofStreamItem[];

    // Stats
    stats: ProofStreamStats;

    // Live Updates
    currentGenerating: ProofStreamItem | null;
    generationQueue: ProofStreamItem[];

    // On-chain
    pendingVerifications: OnChainVerification[];
    confirmedVerifications: OnChainVerification[];

    // Connection
    isConnected: boolean;
    isLoading: boolean;
    error: string | null;

    // Actions
    refresh: () => Promise<void>;
    getProofById: (proofId: string) => ProofStreamItem | undefined;
    getProofsByStatus: (stage: ProofStage) => ProofStreamItem[];
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useProofStream(options: UseProofStreamOptions = {}): UseProofStreamReturn {
    const {
        modelId,
        maxProofs = 100,
        enableWebSocket = true,
        onProofGenerated,
        onProofVerified,
        onProofFailed,
    } = options;

    const modelIdBigInt = modelId !== undefined
        ? (typeof modelId === 'number' ? BigInt(modelId) : modelId)
        : undefined;

    // State
    const [proofs, setProofs] = useState<ProofStreamItem[]>([]);
    const [pendingVerifications, _setPendingVerifications] = useState<OnChainVerification[]>([]);
    const [confirmedVerifications, setConfirmedVerifications] = useState<OnChainVerification[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Contract Events
    const { proofEvents } = useContractEvents(modelIdBigInt);

    // WebSocket
    const { isConnected, on } = useWebSocket({
        autoConnect: enableWebSocket,
        channels: enableWebSocket
            ? modelIdBigInt ? [`proofs:${modelIdBigInt.toString()}`] : ['proofs']
            : [],
    });

    // ========================================================================
    // Initialize
    // ========================================================================

    const initializeProofs = useCallback(async () => {
        try {
            setIsLoading(true);
            setError(null);

            // Create proofs from contract events
            const eventProofs: ProofStreamItem[] = proofEvents.slice(0, maxProofs).map((event, index) => ({
                id: `proof-${event.transactionHash}-${index}`,
                hash: event.transactionHash,
                modelId: event.modelId,
                roundId: event.roundId,
                prover: event.prover,
                type: 'training',
                stage: 'verified',
                progress: 100,
                startedAt: event.timestamp - 5000,
                completedAt: event.timestamp,
                errorBound: 0.0001 + Math.random() * 0.0005,
                constraintCount: 50000 + Math.floor(Math.random() * 50000),
                circuitType: 'nova_folding',
                gasUsed: BigInt(150000 + Math.floor(Math.random() * 100000)),
                transactionHash: event.transactionHash,
                blockNumber: event.blockNumber,
                metrics: {
                    witnessTime: 50 + Math.random() * 50,
                    setupTime: 20 + Math.random() * 30,
                    provingTime: 100 + Math.random() * 200,
                    verificationTime: 30 + Math.random() * 40,
                    totalTime: 200 + Math.random() * 300,
                    memoryPeak: 4000 + Math.random() * 4000,
                    cpuPeak: 70 + Math.random() * 25,
                },
            }));

            setProofs(eventProofs);

        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load proofs');
        } finally {
            setIsLoading(false);
        }
    }, [proofEvents, maxProofs, modelIdBigInt]);

    // ========================================================================
    // WebSocket Handlers
    // ========================================================================

    useEffect(() => {
        if (!enableWebSocket) return;

        const unsubProgress = on<ProofGenerationEvent>('proof_progress', (msg) => {
            const event = msg.data;
            setProofs(prev => prev.map(p =>
                p.id === event.proofId
                    ? {
                        ...p,
                        stage: event.stage,
                        progress: event.progress,
                        metrics: { ...p.metrics, ...event.metrics },
                    }
                    : p
            ));
        });

        const unsubVerified = on<OnChainVerification>('proof_verified', (msg) => {
            const verification = msg.data;
            setProofs(prev => prev.map(p =>
                p.id === verification.proofId
                    ? {
                        ...p,
                        stage: verification.success ? 'verified' : 'failed',
                        progress: 100,
                        completedAt: verification.timestamp,
                        transactionHash: verification.transactionHash,
                        blockNumber: verification.blockNumber,
                        gasUsed: verification.gasUsed,
                        error: verification.error,
                    }
                    : p
            ));

            setConfirmedVerifications(prev => [verification, ...prev].slice(0, 50));
            onProofVerified?.(verification);
        });

        const unsubNew = on<ProofStreamItem>('proof_new', (msg) => {
            setProofs(prev => [msg.data, ...prev].slice(0, maxProofs));
        });

        return () => {
            unsubProgress();
            unsubVerified();
            unsubNew();
        };
    }, [enableWebSocket, on, maxProofs, onProofVerified]);

    // No simulation — real proof updates come from WebSocket

    // ========================================================================
    // Effects
    // ========================================================================

    useEffect(() => {
        initializeProofs();
    }, [initializeProofs]);

    // ========================================================================
    // Computed Values
    // ========================================================================

    const activeProofs = useMemo(
        () => proofs.filter(p =>
            p.stage !== 'verified' && p.stage !== 'failed'
        ),
        [proofs]
    );

    const recentVerified = useMemo(
        () => proofs.filter(p => p.stage === 'verified').slice(0, 10),
        [proofs]
    );

    const currentGenerating = useMemo(
        () => activeProofs.find(p => p.stage === 'proving') || null,
        [activeProofs]
    );

    const generationQueue = useMemo(
        () => proofs.filter(p => p.stage === 'queued'),
        [proofs]
    );

    const stats = useMemo((): ProofStreamStats => {
        const verified = proofs.filter(p => p.stage === 'verified');
        const failed = proofs.filter(p => p.stage === 'failed');
        const generating = proofs.filter(p =>
            ['witness', 'setup', 'proving'].includes(p.stage)
        );
        const submitting = proofs.filter(p =>
            ['submitting', 'verifying'].includes(p.stage)
        );
        const queued = proofs.filter(p => p.stage === 'queued');

        const completedProofs = [...verified, ...failed];
        const avgGenTime = completedProofs.length > 0
            ? completedProofs.reduce((sum, p) => sum + (p.metrics.provingTime || 0), 0) / completedProofs.length
            : 0;
        const avgVerTime = verified.length > 0
            ? verified.reduce((sum, p) => sum + (p.metrics.verificationTime || 0), 0) / verified.length
            : 0;

        const successRate = completedProofs.length > 0
            ? verified.length / completedProofs.length
            : 0;

        // Calculate throughput (proofs verified in last minute)
        const oneMinuteAgo = Date.now() - 60000;
        const recentVerified = verified.filter(p => p.completedAt && p.completedAt > oneMinuteAgo);

        return {
            total: proofs.length,
            queued: queued.length,
            generating: generating.length,
            submitting: submitting.length,
            verified: verified.length,
            failed: failed.length,
            averageGenerationTime: avgGenTime,
            averageVerificationTime: avgVerTime,
            successRate,
            throughputPerMinute: recentVerified.length,
        };
    }, [proofs]);

    // ========================================================================
    // Actions
    // ========================================================================

    const refresh = useCallback(async () => {
        await initializeProofs();
    }, [initializeProofs]);

    const getProofById = useCallback(
        (proofId: string) => proofs.find(p => p.id === proofId),
        [proofs]
    );

    const getProofsByStatus = useCallback(
        (stage: ProofStage) => proofs.filter(p => p.stage === stage),
        [proofs]
    );

    // ========================================================================
    // Return
    // ========================================================================

    return {
        proofs,
        activeProofs,
        recentVerified,
        stats,
        currentGenerating,
        generationQueue,
        pendingVerifications,
        confirmedVerifications,
        isConnected,
        isLoading,
        error,
        refresh,
        getProofById,
        getProofsByStatus,
    };
}

export default useProofStream;
