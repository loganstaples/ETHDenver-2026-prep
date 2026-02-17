'use client';

import { useMemo } from 'react';
import {
  useAccount,
  useReadContract,
  useReadContracts,
} from 'wagmi';
import { getContractAddress, HELIX_MODEL_STORE_ABI, type OnChainVersion } from '@/lib/contracts';
import { USE_MOCK_DATA, MOCK_MODELS, MOCK_CURRENT_USER } from '@/lib/mock-models';

const ZERO_ADDR = '0x0000000000000000000000000000000000000000';

export interface PublicModel {
  tokenId: number;
  slug: string;
  name: string;
  description: string;
  creator: string;
  owner: string;
  createdAt: number;
  isPublic: boolean;
  inferenceFee: number;
  forSale: boolean;
  salePrice: number; // in ETH
  versionCount: number;
  latestVersion: OnChainVersion | null;
  bestAccuracy: number;
}

export type ModelFilter = 'all' | 'others' | 'mine';

export function usePublicModels() {
  const { address, isConnected, chainId } = useAccount();
  const live = !USE_MOCK_DATA;

  const contractAddress = useMemo(() => {
    if (!chainId) return ZERO_ADDR;
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
    query: { enabled: live && isContractDeployed },
  });

  const totalSupply = rawTotalSupply ? Number(rawTotalSupply) : 0;

  // Phase 2: Get all token IDs via tokenByIndex
  const tokenIdCalls = useMemo(() => {
    if (!live || totalSupply === 0) return [];
    return Array.from({ length: totalSupply }, (_, i) => ({
      address: addr,
      abi: HELIX_MODEL_STORE_ABI,
      functionName: 'tokenByIndex' as const,
      args: [BigInt(i)],
    }));
  }, [live, totalSupply, addr]);

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
  }, [live, tokenIds, addr]);

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
    if (USE_MOCK_DATA || !modelDataResults || tokenIds.length === 0) return [];
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

      const m = modelResult.result as unknown as readonly [string, string, string, string, bigint, boolean, number];
      const owner = ownerResult.result as unknown as string;
      const forSale = forSaleResult?.status === 'success' ? (forSaleResult.result as unknown as boolean) : false;
      const salePriceWei = salePriceResult?.status === 'success' ? (salePriceResult.result as unknown as bigint) : BigInt(0);
      const salePrice = Number(salePriceWei) / 1e18;

      let versionCount = 0;
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
        }

        if (rawVersions.length > 0) {
          const last = rawVersions[rawVersions.length - 1];
          latestVersion = {
            semver: last.semver,
            rootHash: last.rootHash,
            accuracy: Number(last.accuracy) / 1e4,
            timestamp: Number(last.timestamp),
            sessionId: last.sessionId,
            weightsStored: last.weightsStored,
          };
        }
      }

      result.push({
        tokenId: tokenIds[i],
        slug: m[0],
        name: m[1],
        description: m[2],
        creator: m[3],
        owner,
        createdAt: Number(m[4]),
        isPublic: m[5],
        inferenceFee: Number(m[6]),
        forSale,
        salePrice,
        versionCount,
        latestVersion,
        bestAccuracy,
      });
    }

    return result;
  }, [modelDataResults, tokenIds]);

  // ── Mock path: build PublicModel[] from static data ────────────────
  const mockModels: PublicModel[] = useMemo(() => {
    if (!USE_MOCK_DATA) return [];
    return MOCK_MODELS.map((m) => ({
      ...m,
      forSale: m.forSale ?? false,
      salePrice: m.salePrice ?? 0,
      versionCount: m.versions.length,
      latestVersion: m.versions.length > 0 ? m.versions[m.versions.length - 1] : null,
      bestAccuracy: m.versions.reduce((best, v) => (v.accuracy > best ? v.accuracy : best), 0),
    }));
  }, []);

  const allModels = USE_MOCK_DATA ? mockModels : chainModels;
  const isLoading = USE_MOCK_DATA ? false : (isLoadingSupply || isLoadingTokenIds || isLoadingModelData);

  const refetch = () => {
    if (!USE_MOCK_DATA) {
      refetchSupply();
      refetchModelData();
    }
  };

  return {
    address: USE_MOCK_DATA ? (address || MOCK_CURRENT_USER) : address,
    isConnected: USE_MOCK_DATA ? true : isConnected,
    isContractDeployed: USE_MOCK_DATA ? true : isContractDeployed,
    allModels,
    isLoading,
    refetch,
  };
}
