'use client';

import { useCallback, useMemo } from 'react';
import {
  useAccount,
  useReadContract,
  useReadContracts,
  useWriteContract,
  useWaitForTransactionReceipt,
} from 'wagmi';
import { getContractAddress, HELIX_MODEL_STORE_ABI, type OnChainVersion } from '@/lib/contracts';

const ZERO_ADDR = '0x0000000000000000000000000000000000000000';

export interface ModelWithVersions {
  tokenId: number;
  slug: string;
  name: string;
  description: string;
  creator: string;
  createdAt: number;
  isPublic: boolean;
  inferenceFee: number;
  versions: OnChainVersion[];
}

export interface UseModelRegistryReturn {
  address: string | undefined;
  isConnected: boolean;
  isContractDeployed: boolean;
  models: ModelWithVersions[];
  isLoading: boolean;
  createModel: (params: { slug: string; name: string; description: string }) => void;
  addVersion: (params: {
    tokenId: number;
    semver: string;
    rootHash: string;
    accuracy: number;
    sessionId: string;
    weightsStored: boolean;
  }) => void;
  isWritePending: boolean;
  isConfirming: boolean;
  writeError: Error | null;
  isSuccess: boolean;
  refetch: () => void;
}

export function useModelRegistry(): UseModelRegistryReturn {
  const { address, isConnected, chainId } = useAccount();

  const contractAddress = useMemo(() => {
    if (!chainId) return ZERO_ADDR;
    return getContractAddress(chainId, 'helixModelStore');
  }, [chainId]);

  const isContractDeployed = contractAddress !== ZERO_ADDR;
  const addr = contractAddress as `0x${string}`;

  // ── Phase 1: Get balance ──────────────────────────────────────────
  const {
    data: rawBalance,
    isLoading: isLoadingBalance,
    refetch: refetchBalance,
  } = useReadContract({
    address: addr,
    abi: HELIX_MODEL_STORE_ABI,
    functionName: 'balanceOf',
    args: address ? [address] : undefined,
    query: { enabled: isConnected && isContractDeployed && !!address },
  });

  const balance = rawBalance ? Number(rawBalance) : 0;

  // ── Phase 2: Get token IDs ────────────────────────────────────────
  const tokenIdCalls = useMemo(() => {
    if (!address || balance === 0) return [];
    return Array.from({ length: balance }, (_, i) => ({
      address: addr,
      abi: HELIX_MODEL_STORE_ABI,
      functionName: 'tokenOfOwnerByIndex' as const,
      args: [address, BigInt(i)],
    }));
  }, [address, balance, addr]);

  const {
    data: tokenIdResults,
    isLoading: isLoadingTokenIds,
  } = useReadContracts({
    contracts: tokenIdCalls,
    query: { enabled: tokenIdCalls.length > 0 },
  });

  const tokenIds = useMemo(() => {
    if (!tokenIdResults) return [];
    return tokenIdResults
      .filter((r) => r.status === 'success')
      .map((r) => Number(r.result));
  }, [tokenIdResults]);

  // ── Phase 3: Get model data + versions ────────────────────────────
  const modelDataCalls = useMemo(() => {
    if (tokenIds.length === 0) return [];
    const calls: {
      address: `0x${string}`;
      abi: typeof HELIX_MODEL_STORE_ABI;
      functionName: string;
      args: readonly unknown[];
    }[] = [];
    for (const tokenId of tokenIds) {
      calls.push({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'models',
        args: [BigInt(tokenId)],
      });
      calls.push({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'getVersions',
        args: [BigInt(tokenId)],
      });
    }
    return calls;
  }, [tokenIds, addr]);

  const {
    data: modelDataResults,
    isLoading: isLoadingModelData,
    refetch: refetchModelData,
  } = useReadContracts({
    contracts: modelDataCalls,
    query: { enabled: modelDataCalls.length > 0 },
  });

  // ── Parse results ─────────────────────────────────────────────────
  const models: ModelWithVersions[] = useMemo(() => {
    if (!modelDataResults || tokenIds.length === 0) return [];
    const result: ModelWithVersions[] = [];

    for (let i = 0; i < tokenIds.length; i++) {
      const modelResult = modelDataResults[i * 2];
      const versionsResult = modelDataResults[i * 2 + 1];

      if (modelResult?.status !== 'success' || !modelResult.result) continue;

      const m = modelResult.result as unknown as readonly [string, string, string, string, bigint, boolean, number];

      const versions: OnChainVersion[] = [];
      if (versionsResult?.status === 'success' && versionsResult.result) {
        const rawVersions = versionsResult.result as unknown as readonly {
          semver: string;
          rootHash: string;
          accuracy: bigint;
          timestamp: bigint;
          sessionId: string;
          weightsStored: boolean;
        }[];
        for (const v of rawVersions) {
          versions.push({
            semver: v.semver,
            rootHash: v.rootHash,
            accuracy: Number(v.accuracy) / 1e4,
            timestamp: Number(v.timestamp),
            sessionId: v.sessionId,
            weightsStored: v.weightsStored,
          });
        }
      }

      result.push({
        tokenId: tokenIds[i],
        slug: m[0],
        name: m[1],
        description: m[2],
        creator: m[3],
        createdAt: Number(m[4]),
        isPublic: m[5],
        inferenceFee: Number(m[6]),
        versions,
      });
    }

    return result;
  }, [modelDataResults, tokenIds]);

  const isLoading = isLoadingBalance || isLoadingTokenIds || isLoadingModelData;

  // ── Writes ────────────────────────────────────────────────────────
  const {
    writeContract,
    data: txHash,
    isPending: isWritePending,
    error: writeError,
  } = useWriteContract();

  const { isLoading: isConfirming, isSuccess } = useWaitForTransactionReceipt({
    hash: txHash,
  });

  const createModel = useCallback(
    (params: { slug: string; name: string; description: string }) => {
      if (!isConnected || !isContractDeployed) return;
      writeContract({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'createModel',
        args: [params.slug, params.name, params.description],
      });
    },
    [isConnected, isContractDeployed, addr, writeContract],
  );

  const addVersion = useCallback(
    (params: {
      tokenId: number;
      semver: string;
      rootHash: string;
      accuracy: number;
      sessionId: string;
      weightsStored: boolean;
    }) => {
      if (!isConnected || !isContractDeployed) return;
      const scaledAccuracy = BigInt(Math.round(params.accuracy * 100));
      writeContract({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'addVersion',
        args: [
          BigInt(params.tokenId),
          params.semver,
          params.rootHash,
          scaledAccuracy,
          params.sessionId,
          params.weightsStored,
        ],
      });
    },
    [isConnected, isContractDeployed, addr, writeContract],
  );

  const refetch = useCallback(() => {
    refetchBalance();
    refetchModelData();
  }, [refetchBalance, refetchModelData]);

  return {
    address,
    isConnected,
    isContractDeployed,
    models,
    isLoading,
    createModel,
    addVersion,
    isWritePending,
    isConfirming,
    writeError: writeError ?? null,
    isSuccess,
    refetch,
  };
}
