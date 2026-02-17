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
import { USE_MOCK_DATA, MOCK_MODELS, MOCK_CURRENT_USER } from '@/lib/mock-models';

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
  forSale: boolean;
  salePrice: number; // in ETH
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
  setPublic: (params: { tokenId: number; isPublic: boolean }) => void;
  setInferenceFee: (params: { tokenId: number; feeBps: number }) => void;
  setForSale: (params: { tokenId: number; forSale: boolean }) => void;
  setSalePrice: (params: { tokenId: number; priceEth: number }) => void;
  buyModel: (params: { tokenId: number; priceEth: number }) => void;
  isWritePending: boolean;
  isConfirming: boolean;
  writeError: Error | null;
  isSuccess: boolean;
  refetch: () => void;
}

export function useModelRegistry(): UseModelRegistryReturn {
  const { address, isConnected, chainId } = useAccount();
  const live = !USE_MOCK_DATA;

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
    query: { enabled: live && isConnected && isContractDeployed && !!address },
  });

  const balance = rawBalance ? Number(rawBalance) : 0;

  // ── Phase 2: Get token IDs ────────────────────────────────────────
  const tokenIdCalls = useMemo(() => {
    if (!live || !address || balance === 0) return [];
    return Array.from({ length: balance }, (_, i) => ({
      address: addr,
      abi: HELIX_MODEL_STORE_ABI,
      functionName: 'tokenOfOwnerByIndex' as const,
      args: [address, BigInt(i)],
    }));
  }, [live, address, balance, addr]);

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

  // ── Phase 3: Get model data + versions + sale info ────────────────
  const CALLS_PER_TOKEN = 4;
  const modelDataCalls = useMemo(() => {
    if (!live || tokenIds.length === 0) return [];
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
      calls.push({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'isForSale',
        args: [BigInt(tokenId)],
      });
      calls.push({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'salePrice',
        args: [BigInt(tokenId)],
      });
    }
    return calls;
  }, [live, tokenIds, addr]);

  const {
    data: modelDataResults,
    isLoading: isLoadingModelData,
    refetch: refetchModelData,
  } = useReadContracts({
    contracts: modelDataCalls,
    query: { enabled: modelDataCalls.length > 0 },
  });

  // ── Parse on-chain results ─────────────────────────────────────────
  const chainModels: ModelWithVersions[] = useMemo(() => {
    if (USE_MOCK_DATA || !modelDataResults || tokenIds.length === 0) return [];
    const result: ModelWithVersions[] = [];

    for (let i = 0; i < tokenIds.length; i++) {
      const base = i * CALLS_PER_TOKEN;
      const modelResult = modelDataResults[base];
      const versionsResult = modelDataResults[base + 1];
      const forSaleResult = modelDataResults[base + 2];
      const salePriceResult = modelDataResults[base + 3];

      if (modelResult?.status !== 'success' || !modelResult.result) continue;

      const m = modelResult.result as unknown as readonly [string, string, string, string, bigint, boolean, number];
      const forSale = forSaleResult?.status === 'success' ? (forSaleResult.result as unknown as boolean) : false;
      const salePriceWei = salePriceResult?.status === 'success' ? (salePriceResult.result as unknown as bigint) : BigInt(0);
      const salePrice = Number(salePriceWei) / 1e18;

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
        forSale,
        salePrice,
        versions,
      });
    }

    return result;
  }, [modelDataResults, tokenIds]);

  // ── Mock path ──────────────────────────────────────────────────────
  const mockModels: ModelWithVersions[] = useMemo(() => {
    if (!USE_MOCK_DATA) return [];
    return MOCK_MODELS
      .filter((m) => m.owner === MOCK_CURRENT_USER)
      .map((m) => ({
        tokenId: m.tokenId,
        slug: m.slug,
        name: m.name,
        description: m.description,
        creator: m.creator,
        createdAt: m.createdAt,
        isPublic: m.isPublic,
        inferenceFee: m.inferenceFee,
        forSale: m.forSale ?? false,
        salePrice: m.salePrice ?? 0,
        versions: m.versions,
      }));
  }, []);

  const models = USE_MOCK_DATA ? mockModels : chainModels;
  const isLoading = USE_MOCK_DATA ? false : (isLoadingBalance || isLoadingTokenIds || isLoadingModelData);

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
      if (USE_MOCK_DATA || !isConnected || !isContractDeployed) return;
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
      if (USE_MOCK_DATA || !isConnected || !isContractDeployed) return;
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

  const setPublic = useCallback(
    (params: { tokenId: number; isPublic: boolean }) => {
      if (USE_MOCK_DATA || !isConnected || !isContractDeployed) return;
      writeContract({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'setPublic',
        args: [BigInt(params.tokenId), params.isPublic],
      });
    },
    [isConnected, isContractDeployed, addr, writeContract],
  );

  const setInferenceFee = useCallback(
    (params: { tokenId: number; feeBps: number }) => {
      if (USE_MOCK_DATA || !isConnected || !isContractDeployed) return;
      writeContract({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'setInferenceFee',
        args: [BigInt(params.tokenId), params.feeBps],
      });
    },
    [isConnected, isContractDeployed, addr, writeContract],
  );

  const setForSale = useCallback(
    (params: { tokenId: number; forSale: boolean }) => {
      if (USE_MOCK_DATA || !isConnected || !isContractDeployed) return;
      writeContract({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'setForSale',
        args: [BigInt(params.tokenId), params.forSale],
      });
    },
    [isConnected, isContractDeployed, addr, writeContract],
  );

  const setSalePrice = useCallback(
    (params: { tokenId: number; priceEth: number }) => {
      if (USE_MOCK_DATA || !isConnected || !isContractDeployed) return;
      // Convert ETH to wei (BigInt)
      const priceWei = BigInt(Math.round(params.priceEth * 1e18));
      writeContract({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'setSalePrice',
        args: [BigInt(params.tokenId), priceWei],
      });
    },
    [isConnected, isContractDeployed, addr, writeContract],
  );

  const buyModel = useCallback(
    (params: { tokenId: number; priceEth: number }) => {
      if (USE_MOCK_DATA || !isConnected || !isContractDeployed) return;
      const priceWei = BigInt(Math.round(params.priceEth * 1e18));
      writeContract({
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'buyModel',
        args: [BigInt(params.tokenId)],
        value: priceWei,
      });
    },
    [isConnected, isContractDeployed, addr, writeContract],
  );

  const refetch = useCallback(() => {
    if (!USE_MOCK_DATA) {
      refetchBalance();
      refetchModelData();
    }
  }, [refetchBalance, refetchModelData]);

  return {
    address: USE_MOCK_DATA ? (address || MOCK_CURRENT_USER) : address,
    isConnected: USE_MOCK_DATA ? true : isConnected,
    isContractDeployed: USE_MOCK_DATA ? true : isContractDeployed,
    models,
    isLoading,
    createModel,
    addVersion,
    setPublic,
    setInferenceFee,
    setForSale,
    setSalePrice,
    buyModel,
    isWritePending,
    isConfirming,
    writeError: writeError ?? null,
    isSuccess,
    refetch,
  };
}
