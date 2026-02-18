import { useMemo } from 'react';
import { useContractState, useContractEvents } from '@/hooks/useContract';

export interface DashboardMetrics {
    totalStake: bigint;
    activeProvers: number;
    totalProofs: number;
    avgProofTime: number; // mocked for now
    slashingRate: number;
    networkLoad: number; // 0-100
    recentActivity: any[];
}

export function useDashboardMetrics(selectedModelId: bigint) {
    const {
        nextModelId,
        isLoading: contractLoading,
    } = useContractState();

    const {
        proofEvents,
        roundStartedEvents,
        roundCompletedEvents,
        stakedEvents,
        slashedEvents,
    } = useContractEvents(selectedModelId);

    const metrics = useMemo(() => {
        const totalStake = stakedEvents.reduce((sum, e) => sum + e.amount, BigInt(0));

        // Calculate unique provers
        const uniqueProvers = new Set(proofEvents.map(e => e.prover)).size;

        // Calculate slashing rate (slashed / (slashed + proofs))
        const totalActions = slashedEvents.length + proofEvents.length;
        const slashingRate = totalActions > 0 ? (slashedEvents.length / totalActions) * 100 : 0;

        // Mock network load based on recent activity frequency
        const recentEventCount = proofEvents.filter(e => Date.now() - Number(e.timestamp) * 1000 < 3600000).length;
        const networkLoad = Math.min(Math.round((recentEventCount / 100) * 100), 100);

        // Combine and sort all events
        const allEvents = [
            ...proofEvents.map(e => ({ ...e, type: 'PROOF', severity: 'info', message: `Proof submitted for model #${e.modelId}` })),
            ...roundStartedEvents.map(e => ({ ...e, type: 'ROUND_START', severity: 'success', message: `Round #${e.roundId} started` })),
            ...roundCompletedEvents.map(e => ({ ...e, type: 'ROUND_END', severity: 'success', message: `Round #${e.roundId} completed` })),
            ...stakedEvents.map(e => ({ ...e, type: 'STAKE', severity: 'warning', message: `New stake deposit: ${(Number(e.amount) / 1e18).toFixed(2)} ADI` })),
            ...slashedEvents.map(e => ({ ...e, type: 'SLASH', severity: 'error', message: `Prover slashed for misconduct` })),
        ].sort((a, b) => b.timestamp - a.timestamp);

        return {
            totalStake,
            activeProvers: uniqueProvers,
            totalProofs: proofEvents.length,
            avgProofTime: 1240, // ms, mocked
            slashingRate,
            networkLoad,
            recentActivity: allEvents,
        };
    }, [proofEvents, roundStartedEvents, roundCompletedEvents, stakedEvents, slashedEvents]);

    return {
        metrics,
        nextModelId,
        isLoading: contractLoading,
    };
}
