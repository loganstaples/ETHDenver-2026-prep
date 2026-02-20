'use client';

import { useMemo, useState, useEffect, useCallback } from 'react';
import {
  useAccount,
  useChainId,
  useReadContract,
  useReadContracts,
} from 'wagmi';
import { getContractAddress, HELIX_MODEL_STORE_ABI, type OnChainVersion } from '@/lib/contracts';

const ZERO_ADDR = '0x0000000000000000000000000000000000000000';

export interface PublicModel {
  tokenId: number;
  slug: string;
  name: string;
  description: string;
  architecture: string;
  creator: string;
  owner: string;
  createdAt: number;
  isPublic: boolean;
  inferenceFee: number;
  forSale: boolean;
  salePrice: number; // in ADI
  versionCount: number;
  versions: OnChainVersion[];
  latestVersion: OnChainVersion | null;
  bestAccuracy: number;
  averageRating: number; // 0-5, default 0 (no backend yet)
  ratingCount: number; // default 0
  inferenceCount: number; // default 0
}

export type ModelFilter = 'all' | 'others' | 'mine';

export function usePublicModels() {
  const { address, isConnected } = useAccount();
  const chainId = useChainId();

  const contractAddress = useMemo(() => {
    return getContractAddress(chainId, 'helixModelStore');
  }, [chainId]);

  const isContractDeployed = contractAddress !== ZERO_ADDR;
  const addr = contractAddress as `0x${string}`;

  // Phase 1: Get total supply
  const {
    data: rawTotalSupply,
    isLoading: isLoadingSupply,
    refetch: refetchSupply,
  } = useReadContract({
    address: addr,
    abi: HELIX_MODEL_STORE_ABI,
    functionName: 'totalSupply',
    query: { enabled: isContractDeployed },
  });

  const totalSupply = rawTotalSupply ? Number(rawTotalSupply) : 0;

  // Phase 2: Get all token IDs via tokenByIndex
  const tokenIdCalls = useMemo(() => {
    if (totalSupply === 0) return [];
    return Array.from({ length: totalSupply }, (_, i) => ({
      address: addr,
      abi: HELIX_MODEL_STORE_ABI,
      functionName: 'tokenByIndex' as const,
      args: [BigInt(i)],
    }));
  }, [totalSupply, addr]);

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

  // Phase 3: Fetch model data + ownerOf + getVersions + isForSale + salePrice for each token
  const CALLS_PER_TOKEN = 5;
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
        functionName: 'ownerOf',
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
  }, [tokenIds, addr]);

  const {
    data: modelDataResults,
    isLoading: isLoadingModelData,
    refetch: refetchModelData,
  } = useReadContracts({
    contracts: modelDataCalls,
    query: { enabled: modelDataCalls.length > 0 },
  });

  // Parse on-chain results into PublicModel[]
  const chainModels: PublicModel[] = useMemo(() => {
    if (!modelDataResults || tokenIds.length === 0) return [];
    const result: PublicModel[] = [];

    for (let i = 0; i < tokenIds.length; i++) {
      const base = i * CALLS_PER_TOKEN;
      const modelResult = modelDataResults[base];
      const ownerResult = modelDataResults[base + 1];
      const versionsResult = modelDataResults[base + 2];
      const forSaleResult = modelDataResults[base + 3];
      const salePriceResult = modelDataResults[base + 4];

      if (modelResult?.status !== 'success' || !modelResult.result) continue;
      if (ownerResult?.status !== 'success' || !ownerResult.result) continue;

      const m = modelResult.result as unknown as readonly [string, string, string, string, string, bigint, boolean, number];
      const owner = ownerResult.result as unknown as string;
      const forSale = forSaleResult?.status === 'success' ? (forSaleResult.result as unknown as boolean) : false;
      const salePriceWei = salePriceResult?.status === 'success' ? (salePriceResult.result as unknown as bigint) : BigInt(0);
      const salePrice = Number(salePriceWei) / 1e18;

      let versionCount = 0;
      const versions: OnChainVersion[] = [];
      let latestVersion: OnChainVersion | null = null;
      let bestAccuracy = 0;

      if (versionsResult?.status === 'success' && versionsResult.result) {
        const rawVersions = versionsResult.result as unknown as readonly {
          semver: string;
          rootHash: string;
          accuracy: bigint;
          timestamp: bigint;
          sessionId: string;
          weightsStored: boolean;
        }[];

        versionCount = rawVersions.length;

        for (const v of rawVersions) {
          const acc = Number(v.accuracy) / 1e4;
          if (acc > bestAccuracy) bestAccuracy = acc;
          versions.push({
            semver: v.semver,
            rootHash: v.rootHash,
            accuracy: acc,
            timestamp: Number(v.timestamp),
            sessionId: v.sessionId,
            weightsStored: v.weightsStored,
          });
        }

        if (versions.length > 0) {
          latestVersion = versions[versions.length - 1];
        }
      }

      result.push({
        tokenId: tokenIds[i],
        slug: m[0],
        name: m[1],
        description: m[2],
        architecture: m[3],
        creator: m[4],
        owner,
        createdAt: Number(m[5]),
        isPublic: m[6],
        inferenceFee: Number(m[7]),
        forSale,
        salePrice,
        versionCount,
        versions,
        latestVersion,
        bestAccuracy,
        averageRating: 0,
        ratingCount: 0,
        inferenceCount: 0,
      });
    }

    return result;
  }, [modelDataResults, tokenIds]);

  // ── Fetch stats from in-memory store ─────────────────────────────
  const [stats, setStats] = useState<Record<string, { inferenceCount: number; averageRating: number; ratingCount: number }>>({});

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch('/api/models/stats');
      if (res.ok) {
        setStats(await res.json());
      }
    } catch {
      // stats are best-effort
    }
  }, []);

  // Fetch on mount and whenever chain models change
  useEffect(() => {
    if (chainModels.length > 0) {
      fetchStats();
    }
  }, [chainModels.length, fetchStats]);

  // Merge real stats into models
  const allModels = useMemo(() => {
    if (Object.keys(stats).length === 0) return chainModels;
    return chainModels.map((m) => {
      const s = stats[String(m.tokenId)];
      if (!s) return m;
      return {
        ...m,
        inferenceCount: s.inferenceCount,
        averageRating: s.averageRating,
        ratingCount: s.ratingCount,
      };
    });
  }, [chainModels, stats]);

  const isLoading = isLoadingSupply || isLoadingTokenIds || isLoadingModelData;

  const refetch = useCallback(() => {
    refetchSupply();
    refetchModelData();
    fetchStats();
  }, [refetchSupply, refetchModelData, fetchStats]);

  return {
    address,
    isConnected,
    isContractDeployed,
    allModels,
    isLoading,
    refetch,
  };
}
