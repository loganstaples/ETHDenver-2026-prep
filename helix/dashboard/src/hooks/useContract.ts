'use client';

import { useCallback, useEffect, useState } from 'react';
import { useAccount, useChainId, usePublicClient, useWalletClient } from 'wagmi';
import { formatEther, parseEther, type Address } from 'viem';
import {
    HELIX_COORDINATOR_ABI,
    CONTRACT_ADDRESSES,
    getContractAddress,
    type Model,
    type Round,
    type Stake,
    type ProofSubmittedEvent,
    type RoundStartedEvent,
    type RoundCompletedEvent,
    type StakedEvent,
    type SlashedEvent,
} from '@/lib/contracts';

// Contract state interface
export interface ContractState {
    nextModelId: bigint;
    defaultMinStake: bigint;
    stakeLockPeriod: bigint;
    slashPercentage: bigint;
    maxErrorBound: bigint;
}

// Extended contract state interface with slashing record count
export interface ExtendedContractState extends ContractState {
    slashingRecordCount: bigint;
}

// Hook for reading contract state
export function useContractState() {
    const chainId = useChainId();
    const publicClient = usePublicClient();
    const [state, setState] = useState<ExtendedContractState | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const contractAddress = getContractAddress(chainId, 'helixCoordinator') as Address;

    const fetchState = useCallback(async () => {
        if (!publicClient) return;

        try {
            setLoading(true);
            const [nextModelId, defaultMinStake, stakeLockPeriod, slashPercentage, maxErrorBound, slashingRecordCount] =
                await Promise.all([
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

            setState({
                nextModelId: nextModelId as bigint,
                defaultMinStake: defaultMinStake as bigint,
                stakeLockPeriod: stakeLockPeriod as bigint,
                slashPercentage: slashPercentage as bigint,
                maxErrorBound: maxErrorBound as bigint,
                slashingRecordCount: slashingRecordCount as bigint,
            });
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch contract state');
        } finally {
            setLoading(false);
        }
    }, [publicClient, contractAddress]);

    useEffect(() => {
        fetchState();
    }, [fetchState]);

    // Return spread state values for convenience
    return {
        nextModelId: state?.nextModelId,
        defaultMinStake: state?.defaultMinStake,
        stakeLockPeriod: state?.stakeLockPeriod,
        slashPercentage: state?.slashPercentage,
        maxErrorBound: state?.maxErrorBound,
        slashingRecordCount: state?.slashingRecordCount,
        isLoading: loading,
        error,
        refetch: fetchState,
    };
}

// Hook for reading model data
export function useModel(modelId: bigint | number) {
    const chainId = useChainId();
    const publicClient = usePublicClient();
    const [model, setModel] = useState<Model | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const contractAddress = getContractAddress(chainId, 'helixCoordinator') as Address;

    const fetchModel = useCallback(async () => {
        if (!publicClient) return;

        try {
            setLoading(true);
            const result = await publicClient.readContract({
                address: contractAddress,
                abi: HELIX_COORDINATOR_ABI,
                functionName: 'models',
                args: [BigInt(modelId)],
            });

            const [ipfsHash, currentCommitment, currentRound, owner, minStake, active] = result as [
                string,
                bigint,
                bigint,
                Address,
                bigint,
                boolean,
            ];

            setModel({
                ipfsHash,
                currentCommitment,
                currentRound,
                owner,
                minStake,
                active,
            });
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch model');
        } finally {
            setLoading(false);
        }
    }, [publicClient, contractAddress, modelId]);

    useEffect(() => {
        fetchModel();
    }, [fetchModel]);

    return { model, isLoading: loading, error, refetch: fetchModel };
}

// Hook for reading round data
export function useRound(modelId: bigint | number, roundId: bigint | number) {
    const chainId = useChainId();
    const publicClient = usePublicClient();
    const [round, setRound] = useState<Round | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const contractAddress = getContractAddress(chainId, 'helixCoordinator') as Address;

    const fetchRound = useCallback(async () => {
        if (!publicClient) return;

        try {
            setLoading(true);
            const result = await publicClient.readContract({
                address: contractAddress,
                abi: HELIX_COORDINATOR_ABI,
                functionName: 'rounds',
                args: [BigInt(modelId), BigInt(roundId)],
            });

            const [modelCommitment, newCommitment, isCompleted, deadline, prover] = result as [
                bigint,
                bigint,
                boolean,
                bigint,
                Address,
            ];

            setRound({
                modelCommitment,
                newCommitment,
                isCompleted,
                deadline,
                prover,
            });
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch round');
        } finally {
            setLoading(false);
        }
    }, [publicClient, contractAddress, modelId, roundId]);

    useEffect(() => {
        fetchRound();
    }, [fetchRound]);

    return { round, isLoading: loading, error, refetch: fetchRound };
}

// Hook for reading stake data
export function useStake(modelId: bigint | number, proverAddress?: Address) {
    const chainId = useChainId();
    const publicClient = usePublicClient();
    const { address } = useAccount();
    const [stake, setStake] = useState<Stake | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const contractAddress = getContractAddress(chainId, 'helixCoordinator') as Address;
    const targetAddress = proverAddress || address;

    const fetchStake = useCallback(async () => {
        if (!publicClient || !targetAddress) {
            setLoading(false);
            return;
        }

        try {
            setLoading(true);
            const result = await publicClient.readContract({
                address: contractAddress,
                abi: HELIX_COORDINATOR_ABI,
                functionName: 'getStake',
                args: [targetAddress, BigInt(modelId)],
            });

            const [amount, lockedUntil, slashed] = result as [bigint, bigint, boolean];

            setStake({ amount, lockedUntil, slashed });
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch stake');
        } finally {
            setLoading(false);
        }
    }, [publicClient, contractAddress, modelId, targetAddress]);

    useEffect(() => {
        fetchStake();
    }, [fetchStake]);

    return { stake, loading, isLoading: loading, error, refetch: fetchStake };
}

// Hook for reading error bound data
export function useErrorBound(modelId: bigint | number) {
    const chainId = useChainId();
    const publicClient = usePublicClient();
    const [errorBound, setErrorBound] = useState<bigint | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const contractAddress = getContractAddress(chainId, 'helixCoordinator') as Address;

    const fetchErrorBound = useCallback(async () => {
        if (!publicClient) return;

        try {
            setLoading(true);
            const result = await publicClient.readContract({
                address: contractAddress,
                abi: HELIX_COORDINATOR_ABI,
                functionName: 'getAccumulatedErrorBound',
                args: [BigInt(modelId)],
            });

            setErrorBound(result as bigint);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch error bound');
        } finally {
            setLoading(false);
        }
    }, [publicClient, contractAddress, modelId]);

    useEffect(() => {
        fetchErrorBound();
    }, [fetchErrorBound]);

    return { errorBound, loading, error, refetch: fetchErrorBound };
}

// Hook for staking operations
export function useStaking(modelId: bigint | number) {
    const chainId = useChainId();
    const { data: walletClient } = useWalletClient();
    const publicClient = usePublicClient();
    const { address } = useAccount();
    const [pending, setPending] = useState(false);
    const [txHash, setTxHash] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const contractAddress = getContractAddress(chainId, 'helixCoordinator') as Address;

    const stakeTokens = useCallback(
        async (amount: string) => {
            if (!walletClient || !address) {
                setError('Wallet not connected');
                return;
            }

            try {
                setPending(true);
                setError(null);

                const hash = await walletClient.writeContract({
                    address: contractAddress,
                    abi: HELIX_COORDINATOR_ABI,
                    functionName: 'stake',
                    args: [BigInt(modelId)],
                    value: parseEther(amount),
                });

                setTxHash(hash);

                // Wait for confirmation
                if (publicClient) {
                    await publicClient.waitForTransactionReceipt({ hash });
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Staking failed');
            } finally {
                setPending(false);
            }
        },
        [walletClient, address, publicClient, contractAddress, modelId]
    );

    const unstakeTokens = useCallback(async () => {
        if (!walletClient || !address) {
            setError('Wallet not connected');
            return;
        }

        try {
            setPending(true);
            setError(null);

            const hash = await walletClient.writeContract({
                address: contractAddress,
                abi: HELIX_COORDINATOR_ABI,
                functionName: 'unstake',
                args: [BigInt(modelId)],
            });

            setTxHash(hash);

            if (publicClient) {
                await publicClient.waitForTransactionReceipt({ hash });
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unstaking failed');
        } finally {
            setPending(false);
        }
    }, [walletClient, address, publicClient, contractAddress, modelId]);

    return { stakeTokens, unstakeTokens, pending, txHash, error };
}

// Hook for registering models
export function useModelRegistration() {
    const chainId = useChainId();
    const { data: walletClient } = useWalletClient();
    const publicClient = usePublicClient();
    const { address } = useAccount();
    const [pending, setPending] = useState(false);
    const [txHash, setTxHash] = useState<string | null>(null);
    const [modelId, setModelId] = useState<bigint | null>(null);
    const [error, setError] = useState<string | null>(null);

    const contractAddress = getContractAddress(chainId, 'helixCoordinator') as Address;

    const registerModel = useCallback(
        async (ipfsHash: string, initialCommitment: bigint, minStake: string) => {
            if (!walletClient || !address) {
                setError('Wallet not connected');
                return;
            }

            try {
                setPending(true);
                setError(null);

                const hash = await walletClient.writeContract({
                    address: contractAddress,
                    abi: HELIX_COORDINATOR_ABI,
                    functionName: 'registerModel',
                    args: [ipfsHash, initialCommitment, parseEther(minStake)],
                });

                setTxHash(hash);

                if (publicClient) {
                    const receipt = await publicClient.waitForTransactionReceipt({ hash });
                    // Parse logs to get modelId (from ModelRegistered event)
                    // For simplicity, we'll refetch the nextModelId
                    const newModelId = await publicClient.readContract({
                        address: contractAddress,
                        abi: HELIX_COORDINATOR_ABI,
                        functionName: 'nextModelId',
                    });
                    setModelId((newModelId as bigint) - BigInt(1));
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Registration failed');
            } finally {
                setPending(false);
            }
        },
        [walletClient, address, publicClient, contractAddress]
    );

    return {
        registerModel,
        isRegistering: pending,
        registrationSuccess: !!modelId,
        registrationError: error ? new Error(error) : null,
        txHash,
        modelId,
    };
}

// Hook for starting rounds
export function useRoundManagement(modelId: bigint | number) {
    const chainId = useChainId();
    const { data: walletClient } = useWalletClient();
    const publicClient = usePublicClient();
    const { address } = useAccount();
    const [pending, setPending] = useState(false);
    const [txHash, setTxHash] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const contractAddress = getContractAddress(chainId, 'helixCoordinator') as Address;

    const startRound = useCallback(
        async (durationSeconds: number) => {
            if (!walletClient || !address) {
                setError('Wallet not connected');
                return;
            }

            try {
                setPending(true);
                setError(null);

                const hash = await walletClient.writeContract({
                    address: contractAddress,
                    abi: HELIX_COORDINATOR_ABI,
                    functionName: 'startRound',
                    args: [BigInt(modelId), BigInt(durationSeconds)],
                });

                setTxHash(hash);

                if (publicClient) {
                    await publicClient.waitForTransactionReceipt({ hash });
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Starting round failed');
            } finally {
                setPending(false);
            }
        },
        [walletClient, address, publicClient, contractAddress, modelId]
    );

    return { startRound, isStartingRound: pending, txHash, error };
}

// Hook for real-time event listening
export function useContractEvents(modelId?: bigint | number) {
    const chainId = useChainId();
    const publicClient = usePublicClient();
    const [events, setEvents] = useState<{
        proofSubmitted: ProofSubmittedEvent[];
        roundStarted: RoundStartedEvent[];
        roundCompleted: RoundCompletedEvent[];
        staked: StakedEvent[];
        slashed: SlashedEvent[];
    }>({
        proofSubmitted: [],
        roundStarted: [],
        roundCompleted: [],
        staked: [],
        slashed: [],
    });

    const contractAddress = getContractAddress(chainId, 'helixCoordinator') as Address;

    useEffect(() => {
        if (!publicClient) return;

        // Watch for ProofSubmitted events
        const unwatchProof = publicClient.watchContractEvent({
            address: contractAddress,
            abi: HELIX_COORDINATOR_ABI,
            eventName: 'ProofSubmitted',
            onLogs: (logs) => {
                const newEvents = logs.map((log) => ({
                    modelId: log.args.modelId as bigint,
                    roundId: log.args.roundId as bigint,
                    prover: log.args.prover as string,
                    newCommitment: log.args.newCommitment as bigint,
                    blockNumber: Number(log.blockNumber),
                    transactionHash: log.transactionHash,
                    timestamp: Date.now(),
                }));
                setEvents((prev) => ({
                    ...prev,
                    proofSubmitted: [...prev.proofSubmitted, ...newEvents].slice(-100),
                }));
            },
        });

        // Watch for RoundStarted events
        const unwatchRoundStart = publicClient.watchContractEvent({
            address: contractAddress,
            abi: HELIX_COORDINATOR_ABI,
            eventName: 'RoundStarted',
            onLogs: (logs) => {
                const newEvents = logs.map((log) => ({
                    modelId: log.args.modelId as bigint,
                    roundId: log.args.roundId as bigint,
                    deadline: log.args.deadline as bigint,
                    blockNumber: Number(log.blockNumber),
                    transactionHash: log.transactionHash,
                    timestamp: Date.now(),
                }));
                setEvents((prev) => ({
                    ...prev,
                    roundStarted: [...prev.roundStarted, ...newEvents].slice(-100),
                }));
            },
        });

        // Watch for RoundCompleted events
        const unwatchRoundComplete = publicClient.watchContractEvent({
            address: contractAddress,
            abi: HELIX_COORDINATOR_ABI,
            eventName: 'RoundCompleted',
            onLogs: (logs) => {
                const newEvents = logs.map((log) => ({
                    modelId: log.args.modelId as bigint,
                    roundId: log.args.roundId as bigint,
                    newCommitment: log.args.newCommitment as bigint,
                    blockNumber: Number(log.blockNumber),
                    transactionHash: log.transactionHash,
                    timestamp: Date.now(),
                }));
                setEvents((prev) => ({
                    ...prev,
                    roundCompleted: [...prev.roundCompleted, ...newEvents].slice(-100),
                }));
            },
        });

        // Watch for Staked events
        const unwatchStaked = publicClient.watchContractEvent({
            address: contractAddress,
            abi: HELIX_COORDINATOR_ABI,
            eventName: 'Staked',
            onLogs: (logs) => {
                const newEvents = logs.map((log) => ({
                    prover: log.args.prover as string,
                    modelId: log.args.modelId as bigint,
                    amount: log.args.amount as bigint,
                    blockNumber: Number(log.blockNumber),
                    transactionHash: log.transactionHash,
                    timestamp: Date.now(),
                }));
                setEvents((prev) => ({
                    ...prev,
                    staked: [...prev.staked, ...newEvents].slice(-100),
                }));
            },
        });

        // Watch for Slashed events
        const unwatchSlashed = publicClient.watchContractEvent({
            address: contractAddress,
            abi: HELIX_COORDINATOR_ABI,
            eventName: 'Slashed',
            onLogs: (logs) => {
                const newEvents = logs.map((log) => ({
                    prover: log.args.prover as string,
                    modelId: log.args.modelId as bigint,
                    amount: log.args.amount as bigint,
                    reason: log.args.reason as string,
                    blockNumber: Number(log.blockNumber),
                    transactionHash: log.transactionHash,
                    timestamp: Date.now(),
                }));
                setEvents((prev) => ({
                    ...prev,
                    slashed: [...prev.slashed, ...newEvents].slice(-100),
                }));
            },
        });

        return () => {
            unwatchProof();
            unwatchRoundStart();
            unwatchRoundComplete();
            unwatchStaked();
            unwatchSlashed();
        };
    }, [publicClient, contractAddress]);

    // Return with convenient property names
    return {
        proofEvents: events.proofSubmitted,
        roundStartedEvents: events.roundStarted,
        roundCompletedEvents: events.roundCompleted,
        stakedEvents: events.staked,
        slashedEvents: events.slashed,
    };
}

// Convenience hook combining common contract functions
export function useContract() {
    const chainId = useChainId();
    const { address, isConnected } = useAccount();
    const contractAddress = getContractAddress(chainId, 'helixCoordinator');

    return {
        contractAddress,
        chainId,
        userAddress: address,
        isConnected,
    };
}

export default useContract;
