'use client';

/**
 * HELIX Contract State Hook
 * Unified hook for reading all on-chain state with caching, real-time updates, and event subscriptions.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { useAccount, useChainId, usePublicClient, useBlockNumber } from 'wagmi';
import { formatEther, type Address } from 'viem';
import { useWebSocket } from '@/lib/websocket';
import {
    HELIX_COORDINATOR_ABI,
    getContractAddress,
} from '@/lib/contracts';

// ============================================================================
// Types
// ============================================================================

export interface NetworkState {
    nextModelId: bigint;
    totalModels: number;
    defaultMinStake: bigint;
    defaultMinStakeFormatted: string;
    stakeLockPeriod: number;
    slashPercentage: number;
    maxErrorBound: bigint;
    maxErrorBoundFormatted: number;
    totalSlashings: number;
    totalStaked: bigint;
    totalStakedFormatted: string;
}

export interface ModelDetails {
    id: bigint;
    ipfsHash: string;
    currentCommitment: bigint;
    currentRound: bigint;
    owner: Address;
    minStake: bigint;
    minStakeFormatted: string;
    active: boolean;
    accumulatedErrorBound: bigint;
    errorBoundFormatted: number;
    errorBoundPercentage: number;
}

export interface RoundDetails {
    id: bigint;
    modelId: bigint;
    modelCommitment: bigint;
    newCommitment: bigint;
    isCompleted: boolean;
    deadline: number;
    deadlineFormatted: string;
    prover: Address;
    hasProver: boolean;
    timeRemaining: number;
    isExpired: boolean;
    status: 'pending' | 'active' | 'completed' | 'expired';
}

export interface StakeDetails {
    modelId: bigint;
    prover: Address;
    amount: bigint;
    amountFormatted: string;
    lockedUntil: number;
    isLocked: boolean;
    slashed: boolean;
    canUnstake: boolean;
    lockTimeRemaining: number;
}

export interface RewardInfo {
    modelId: bigint;
    address: Address;
    totalEarned: bigint;
    totalEarnedFormatted: string;
    pendingRewards: bigint;
    pendingRewardsFormatted: string;
    proofsSubmitted: number;
    roundsParticipated: number;
}

export interface ContractEvent {
    type: 'proof_submitted' | 'round_started' | 'round_completed' | 'staked' | 'unstaked' | 'slashed';
    modelId: bigint;
    roundId?: bigint;
    prover?: Address;
    amount?: bigint;
    reason?: string;
    transactionHash: string;
    blockNumber: number;
    timestamp: number;
}

export interface UseContractStateOptions {
    modelId?: bigint | number;
    enableWebSocket?: boolean;
    enableAutoRefresh?: boolean;
    refreshInterval?: number;
    onEvent?: (event: ContractEvent) => void;
}

export interface UseContractStateReturn {
    // Network State
    networkState: NetworkState | null;

    // Model Data
    model: ModelDetails | null;
    models: ModelDetails[];

    // Round Data
    currentRound: RoundDetails | null;
    recentRounds: RoundDetails[];

    // Staking
    userStake: StakeDetails | null;
    totalStaked: bigint;

    // Rewards
    userRewards: RewardInfo | null;

    // Events
    recentEvents: ContractEvent[];

    // Status
    isConnected: boolean;
    isLoading: boolean;
    error: string | null;
    lastUpdated: number;

    // Actions
    refresh: () => Promise<void>;
    fetchModel: (modelId: bigint) => Promise<ModelDetails | null>;
    fetchRound: (modelId: bigint, roundId: bigint) => Promise<RoundDetails | null>;
    fetchStake: (modelId: bigint, prover: Address) => Promise<StakeDetails | null>;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useContractState(options: UseContractStateOptions = {}): UseContractStateReturn {
    const {
        modelId,
        enableWebSocket = true,
        enableAutoRefresh = true,
        refreshInterval: _refreshInterval = 10000,
        onEvent,
    } = options;

    const chainId = useChainId();
    const publicClient = usePublicClient();
    const { address: userAddress, isConnected: walletConnected } = useAccount();
    const { data: blockNumber } = useBlockNumber({ watch: enableAutoRefresh });

    const modelIdBigInt = modelId !== undefined
        ? (typeof modelId === 'number' ? BigInt(modelId) : modelId)
        : undefined;

    // State
    const [networkState, setNetworkState] = useState<NetworkState | null>(null);
    const [model, setModel] = useState<ModelDetails | null>(null);
    const [models, setModels] = useState<ModelDetails[]>([]);
    const [currentRound, setCurrentRound] = useState<RoundDetails | null>(null);
    const [recentRounds, setRecentRounds] = useState<RoundDetails[]>([]);
    const [userStake, setUserStake] = useState<StakeDetails | null>(null);
    const [userRewards, _setUserRewards] = useState<RewardInfo | null>(null);
    const [recentEvents, setRecentEvents] = useState<ContractEvent[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [lastUpdated, setLastUpdated] = useState(Date.now());

    const contractAddress = getContractAddress(chainId, 'helixCoordinator') as Address;

    // WebSocket connection
    const { isConnected: wsConnected, on } = useWebSocket({
        autoConnect: enableWebSocket,
        channels: enableWebSocket
            ? modelIdBigInt
                ? [`contract:${modelIdBigInt.toString()}`, 'contract:events']
                : ['contract:events']
            : [],
    });

    // ========================================================================
    // Fetch Functions
    // ========================================================================

    const fetchNetworkState = useCallback(async (): Promise<NetworkState | null> => {
        if (!publicClient) return null;

        try {
            const results = await Promise.allSettled([
                publicClient.readContract({
                    address: contractAddress,
                    abi: HELIX_COORDINATOR_ABI,
                    functionName: 'nextModelId',
                }),
                publicClient.readContract({
                    address: contractAddress,
                    abi: HELIX_COORDINATOR_ABI,
                    functionName: 'defaultMinStake',
                }),
                publicClient.readContract({
                    address: contractAddress,
                    abi: HELIX_COORDINATOR_ABI,
                    functionName: 'stakeLockPeriod',
                }),
                publicClient.readContract({
                    address: contractAddress,
                    abi: HELIX_COORDINATOR_ABI,
                    functionName: 'slashPercentage',
                }),
                publicClient.readContract({
                    address: contractAddress,
                    abi: HELIX_COORDINATOR_ABI,
                    functionName: 'maxErrorBound',
                }),
                publicClient.readContract({
                    address: contractAddress,
                    abi: HELIX_COORDINATOR_ABI,
                    functionName: 'getSlashingRecordCount',
                }),
            ]);

            // Extract values with defaults for failures
            const nextModelId = results[0].status === 'fulfilled' ? results[0].value as bigint : BigInt(0);
            const defaultMinStake = results[1].status === 'fulfilled' ? results[1].value as bigint : BigInt(0);
            const stakeLockPeriod = results[2].status === 'fulfilled' ? Number(results[2].value) : 86400;
            const slashPercentage = results[3].status === 'fulfilled' ? Number(results[3].value) : 10;
            const maxErrorBound = results[4].status === 'fulfilled' ? results[4].value as bigint : BigInt(1e16);
            const slashingCount = results[5].status === 'fulfilled' ? Number(results[5].value) : 0;

            return {
                nextModelId,
                totalModels: Number(nextModelId),
                defaultMinStake,
                defaultMinStakeFormatted: formatEther(defaultMinStake),
                stakeLockPeriod,
                slashPercentage,
                maxErrorBound,
                maxErrorBoundFormatted: Number(maxErrorBound) / 1e18,
                totalSlashings: slashingCount,
                totalStaked: BigInt(0), // Would need a view function or event aggregation
                totalStakedFormatted: '0',
            };
        } catch (err) {
            console.error('Failed to fetch network state:', err);
            return null;
        }
    }, [publicClient, contractAddress]);

    const fetchModel = useCallback(async (id: bigint): Promise<ModelDetails | null> => {
        if (!publicClient) return null;

        try {
            const [modelData, errorBound] = await Promise.all([
                publicClient.readContract({
                    address: contractAddress,
                    abi: HELIX_COORDINATOR_ABI,
                    functionName: 'models',
                    args: [id],
                }),
                publicClient.readContract({
                    address: contractAddress,
                    abi: HELIX_COORDINATOR_ABI,
                    functionName: 'getAccumulatedErrorBound',
                    args: [id],
                }).catch(() => BigInt(0)),
            ]);

            const [ipfsHash, currentCommitment, roundId, owner, minStake, active] = modelData as [
                string, bigint, bigint, Address, bigint, boolean
            ];

            const maxError = networkState?.maxErrorBound || BigInt(1e16);
            const errorBoundVal = errorBound as bigint;

            return {
                id,
                ipfsHash,
                currentCommitment,
                currentRound: roundId,
                owner,
                minStake,
                minStakeFormatted: formatEther(minStake),
                active,
                accumulatedErrorBound: errorBoundVal,
                errorBoundFormatted: Number(errorBoundVal) / 1e18,
                errorBoundPercentage: maxError > BigInt(0)
                    ? Number((errorBoundVal * BigInt(10000)) / maxError) / 100
                    : 0,
            };
        } catch (err) {
            console.error('Failed to fetch model:', err);
            return null;
        }
    }, [publicClient, contractAddress, networkState]);

    const fetchRound = useCallback(async (modelIdArg: bigint, roundId: bigint): Promise<RoundDetails | null> => {
        if (!publicClient) return null;

        try {
            const roundData = await publicClient.readContract({
                address: contractAddress,
                abi: HELIX_COORDINATOR_ABI,
                functionName: 'rounds',
                args: [modelIdArg, roundId],
            });

            const [modelCommitment, newCommitment, isCompleted, deadline, prover] = roundData as [
                bigint, bigint, boolean, bigint, Address
            ];

            const now = Math.floor(Date.now() / 1000);
            const deadlineNum = Number(deadline);
            const timeRemaining = Math.max(0, deadlineNum - now);
            const isExpired = deadlineNum < now;
            const hasProver = prover !== '0x0000000000000000000000000000000000000000';

            let status: RoundDetails['status'] = 'pending';
            if (isCompleted) {
                status = 'completed';
            } else if (isExpired) {
                status = 'expired';
            } else if (hasProver || timeRemaining > 0) {
                status = 'active';
            }

            return {
                id: roundId,
                modelId: modelIdArg,
                modelCommitment,
                newCommitment,
                isCompleted,
                deadline: deadlineNum,
                deadlineFormatted: new Date(deadlineNum * 1000).toLocaleString(),
                prover,
                hasProver,
                timeRemaining,
                isExpired,
                status,
            };
        } catch (err) {
            console.error('Failed to fetch round:', err);
            return null;
        }
    }, [publicClient, contractAddress]);

    const fetchStake = useCallback(async (modelIdArg: bigint, prover: Address): Promise<StakeDetails | null> => {
        if (!publicClient) return null;

        try {
            const stakeData = await publicClient.readContract({
                address: contractAddress,
                abi: HELIX_COORDINATOR_ABI,
                functionName: 'getStake',
                args: [prover, modelIdArg],
            });

            const [amount, lockedUntil, slashed] = stakeData as [bigint, bigint, boolean];
            const now = Math.floor(Date.now() / 1000);
            const lockedUntilNum = Number(lockedUntil);
            const isLocked = lockedUntilNum > now;

            return {
                modelId: modelIdArg,
                prover,
                amount,
                amountFormatted: formatEther(amount),
                lockedUntil: lockedUntilNum,
                isLocked,
                slashed,
                canUnstake: !isLocked && !slashed && amount > BigInt(0),
                lockTimeRemaining: Math.max(0, lockedUntilNum - now),
            };
        } catch (err) {
            console.error('Failed to fetch stake:', err);
            return null;
        }
    }, [publicClient, contractAddress]);

    // ========================================================================
    // Refresh All Data
    // ========================================================================

    const refresh = useCallback(async () => {
        try {
            setIsLoading(true);
            setError(null);

            // Fetch network state first
            const netState = await fetchNetworkState();
            if (netState) {
                setNetworkState(netState);
            }

            // If we have a model ID, fetch its data
            if (modelIdBigInt !== undefined) {
                const modelData = await fetchModel(modelIdBigInt);
                if (modelData) {
                    setModel(modelData);

                    // Fetch current round
                    if (modelData.currentRound > BigInt(0)) {
                        const roundData = await fetchRound(modelIdBigInt, modelData.currentRound);
                        if (roundData) {
                            setCurrentRound(roundData);
                        }

                        // Fetch recent rounds
                        const roundPromises: Promise<RoundDetails | null>[] = [];
                        const startRound = modelData.currentRound > BigInt(5)
                            ? modelData.currentRound - BigInt(5)
                            : BigInt(1);
                        for (let i = modelData.currentRound; i >= startRound && i > BigInt(0); i--) {
                            roundPromises.push(fetchRound(modelIdBigInt, i));
                        }
                        const rounds = (await Promise.all(roundPromises)).filter((r): r is RoundDetails => r !== null);
                        setRecentRounds(rounds);
                    }
                }

                // Fetch user stake if connected
                if (userAddress) {
                    const stakeData = await fetchStake(modelIdBigInt, userAddress);
                    if (stakeData) {
                        setUserStake(stakeData);
                    }
                }
            }

            // Fetch all models if no specific model ID
            if (modelIdBigInt === undefined && netState && netState.totalModels > 0) {
                const modelPromises: Promise<ModelDetails | null>[] = [];
                for (let i = BigInt(1); i <= BigInt(Math.min(netState.totalModels, 10)); i++) {
                    modelPromises.push(fetchModel(i));
                }
                const allModels = (await Promise.all(modelPromises)).filter((m): m is ModelDetails => m !== null);
                setModels(allModels);
            }

            setLastUpdated(Date.now());
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch contract state');
        } finally {
            setIsLoading(false);
        }
    }, [fetchNetworkState, fetchModel, fetchRound, fetchStake, modelIdBigInt, userAddress]);

    // ========================================================================
    // WebSocket Event Handlers
    // ========================================================================

    useEffect(() => {
        if (!enableWebSocket) return;

        const unsubProof = on<{ modelId: bigint; roundId: bigint; prover: string; newCommitment: bigint; transactionHash: string; blockNumber: number }>('proof_submitted', (msg) => {
            const event: ContractEvent = {
                type: 'proof_submitted',
                modelId: msg.data.modelId,
                roundId: msg.data.roundId,
                prover: msg.data.prover as Address,
                transactionHash: msg.data.transactionHash,
                blockNumber: msg.data.blockNumber,
                timestamp: msg.timestamp,
            };
            setRecentEvents(prev => [event, ...prev].slice(0, 50));
            onEvent?.(event);

            // Refresh round data
            if (modelIdBigInt !== undefined && msg.data.modelId === modelIdBigInt) {
                fetchRound(modelIdBigInt, msg.data.roundId).then(round => {
                    if (round) setCurrentRound(round);
                });
            }
        });

        const unsubRoundStart = on<{ modelId: bigint; roundId: bigint; deadline: bigint; transactionHash: string; blockNumber: number }>('round_started', (msg) => {
            const event: ContractEvent = {
                type: 'round_started',
                modelId: msg.data.modelId,
                roundId: msg.data.roundId,
                transactionHash: msg.data.transactionHash,
                blockNumber: msg.data.blockNumber,
                timestamp: msg.timestamp,
            };
            setRecentEvents(prev => [event, ...prev].slice(0, 50));
            onEvent?.(event);

            if (modelIdBigInt !== undefined && msg.data.modelId === modelIdBigInt) {
                fetchRound(modelIdBigInt, msg.data.roundId).then(round => {
                    if (round) setCurrentRound(round);
                });
            }
        });

        const unsubSlashed = on<{ prover: string; modelId: bigint; amount: bigint; reason: string; transactionHash: string; blockNumber: number }>('slashed', (msg) => {
            const event: ContractEvent = {
                type: 'slashed',
                modelId: msg.data.modelId,
                prover: msg.data.prover as Address,
                amount: msg.data.amount,
                reason: msg.data.reason,
                transactionHash: msg.data.transactionHash,
                blockNumber: msg.data.blockNumber,
                timestamp: msg.timestamp,
            };
            setRecentEvents(prev => [event, ...prev].slice(0, 50));
            onEvent?.(event);

            // Refresh stake if this affects the user
            if (userAddress && msg.data.prover.toLowerCase() === userAddress.toLowerCase() && modelIdBigInt !== undefined) {
                fetchStake(modelIdBigInt, userAddress).then(stake => {
                    if (stake) setUserStake(stake);
                });
            }
        });

        return () => {
            unsubProof();
            unsubRoundStart();
            unsubSlashed();
        };
    }, [enableWebSocket, on, onEvent, modelIdBigInt, userAddress, fetchRound, fetchStake]);

    // ========================================================================
    // Auto-refresh on block changes
    // ========================================================================

    useEffect(() => {
        if (enableAutoRefresh && blockNumber) {
            // Only refresh every few blocks to avoid excessive calls
            if (Number(blockNumber) % 3 === 0) {
                refresh();
            }
        }
    }, [blockNumber, enableAutoRefresh, refresh]);

    // Initial load
    useEffect(() => {
        refresh();
    }, [refresh]);

    // ========================================================================
    // Computed Values
    // ========================================================================

    const totalStaked = useMemo(() => {
        if (userStake) {
            return userStake.amount;
        }
        return BigInt(0);
    }, [userStake]);

    // ========================================================================
    // Return
    // ========================================================================

    return {
        networkState,
        model,
        models,
        currentRound,
        recentRounds,
        userStake,
        totalStaked,
        userRewards,
        recentEvents,
        isConnected: wsConnected && walletConnected,
        isLoading,
        error,
        lastUpdated,
        refresh,
        fetchModel,
        fetchRound,
        fetchStake,
    };
}

export default useContractState;
