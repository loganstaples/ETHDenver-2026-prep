'use client';

import { useCallback, useMemo } from 'react';
import { useAccount, useReadContract, useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
import { getContractAddress, HELIX_MODEL_STORE_ABI, type OnChainModelEntry } from '@/lib/contracts';

const ZERO_ADDR = '0x0000000000000000000000000000000000000000';

export interface UseModelRegistryReturn {
  /** Connected wallet address (undefined if not connected) */
  address: string | undefined;
  /** Whether wallet is connected */
  isConnected: boolean;
  /** Whether the HelixModelStore contract is deployed on the current chain */
  isContractDeployed: boolean;
  /** On-chain models for the connected user */
  onChainModels: OnChainModelEntry[];
  /** Whether on-chain models are loading */
  isLoading: boolean;
  /** Register a model on-chain */
  registerModel: (params: {
    version: string;
    rootHash: string;
    accuracy: number; // 0-100 (e.g. 95.5)
    sessionId: string;
  }) => void;
  /** Whether a registration tx is pending */
  isRegistering: boolean;
  /** Whether the latest registration tx is confirming */
  isConfirming: boolean;
  /** Registration error */
  registerError: Error | null;
  /** Whether the latest registration was successful */
  isRegistered: boolean;
  /** Refetch on-chain models */
  refetch: () => void;
}

export function useModelRegistry(): UseModelRegistryReturn {
  const { address, isConnected, chainId } = useAccount();

  const contractAddress = useMemo(() => {
    if (!chainId) return ZERO_ADDR;
    return getContractAddress(chainId, 'helixModelStore');
  }, [chainId]);

  const isContractDeployed = contractAddress !== ZERO_ADDR;

  // ── Read models ──────────────────────────────────────────────────────

  const {
    data: rawModels,
    isLoading,
    refetch,
  } = useReadContract({
    address: contractAddress as `0x${string}`,
    abi: HELIX_MODEL_STORE_ABI,
    functionName: 'getModels',
    args: address ? [address] : undefined,
    query: {
      enabled: isConnected && isContractDeployed && !!address,
    },
  });

  const onChainModels: OnChainModelEntry[] = useMemo(() => {
    if (!rawModels || !Array.isArray(rawModels)) return [];
    return (rawModels as readonly {
      version: string;
      rootHash: string;
      accuracy: bigint;
      timestamp: bigint;
      sessionId: string;
    }[]).map((m) => ({
      version: m.version,
      rootHash: m.rootHash,
      accuracy: Number(m.accuracy) / 1e4,
      timestamp: Number(m.timestamp),
      sessionId: m.sessionId,
    }));
  }, [rawModels]);

  // ── Write: register model ────────────────────────────────────────────

  const {
    writeContract,
    data: txHash,
    isPending: isRegistering,
    error: writeError,
  } = useWriteContract();

  const {
    isLoading: isConfirming,
    isSuccess: isRegistered,
  } = useWaitForTransactionReceipt({ hash: txHash });

  const registerModel = useCallback(
    (params: { version: string; rootHash: string; accuracy: number; sessionId: string }) => {
      if (!isConnected || !isContractDeployed) return;
      // Scale accuracy: 95.5 → 9550
      const scaledAccuracy = BigInt(Math.round(params.accuracy * 100));

      writeContract({
        address: contractAddress as `0x${string}`,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'registerModel',
        args: [params.version, params.rootHash, scaledAccuracy, params.sessionId],
      });
    },
    [isConnected, isContractDeployed, contractAddress, writeContract],
  );

  return {
    address,
    isConnected,
    isContractDeployed,
    onChainModels,
    isLoading,
    registerModel,
    isRegistering,
    isConfirming,
    isRegistered,
    registerError: writeError,
    refetch,
  };
}
