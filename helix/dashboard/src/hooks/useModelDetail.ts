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

export interface ModelDetail {
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
  salePrice: number; // in ADI
  versions: OnChainVersion[];
}

export function useModelDetail(tokenId: number) {
  const { address, isConnected, chainId } = useAccount();
  const live = !USE_MOCK_DATA;

  const contractAddress = useMemo(() => {
    if (!chainId) return ZERO_ADDR;
    return getContractAddress(chainId, 'helixModelStore');
  }, [chainId]);

  const isContractDeployed = contractAddress !== ZERO_ADDR;
  const addr = contractAddress as `0x${string}`;

  // Batch: models(), ownerOf(), getVersions(), isForSale(), salePrice() in one multicall
  const calls = useMemo(() => {
    if (!live || !isContractDeployed) return [];
    return [
      {
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'models' as const,
        args: [BigInt(tokenId)],
      },
      {
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'ownerOf' as const,
        args: [BigInt(tokenId)],
      },
      {
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'getVersions' as const,
        args: [BigInt(tokenId)],
      },
      {
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'isForSale' as const,
        args: [BigInt(tokenId)],
      },
      {
        address: addr,
        abi: HELIX_MODEL_STORE_ABI,
        functionName: 'salePrice' as const,
        args: [BigInt(tokenId)],
      },
    ];
  }, [live, tokenId, addr, isContractDeployed]);

  const {
    data: results,
    isLoading: isLoadingChain,
    isError: isErrorChain,
    refetch,
  } = useReadContracts({
    contracts: calls,
    query: { enabled: calls.length > 0 },
  });

  // Check access if connected
  const {
    data: hasAccessChain,
  } = useReadContract({
    address: addr,
    abi: HELIX_MODEL_STORE_ABI,
    functionName: 'hasModelAccess',
    args: address ? [BigInt(tokenId), address] : undefined,
    query: { enabled: live && isContractDeployed && isConnected && !!address },
  });

  // ── Parse on-chain data ────────────────────────────────────────────
  const chainModel: ModelDetail | null = useMemo(() => {
    if (USE_MOCK_DATA || !results || results.length < 3) return null;

    const modelResult = results[0];
    const ownerResult = results[1];
    const versionsResult = results[2];
    const forSaleResult = results[3];
    const salePriceResult = results[4];

    if (modelResult?.status !== 'success' || !modelResult.result) return null;
    if (ownerResult?.status !== 'success' || !ownerResult.result) return null;

    const m = modelResult.result as unknown as readonly [string, string, string, string, bigint, boolean, number];
    const owner = ownerResult.result as unknown as string;
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

    return {
      tokenId,
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
      versions,
    };
  }, [results, tokenId]);

  // ── Mock path ──────────────────────────────────────────────────────
  const mockModel: ModelDetail | null = useMemo(() => {
    if (!USE_MOCK_DATA) return null;
    const found = MOCK_MODELS.find((m) => m.tokenId === tokenId);
    if (!found) return null;
    return {
      ...found,
      forSale: found.forSale ?? false,
      salePrice: found.salePrice ?? 0,
    };
  }, [tokenId]);

  const model = USE_MOCK_DATA ? mockModel : chainModel;
  const isLoading = USE_MOCK_DATA ? false : isLoadingChain;
  const isError = USE_MOCK_DATA ? false : isErrorChain;
  const isOwner = USE_MOCK_DATA
    ? (!!model && model.owner === MOCK_CURRENT_USER)
    : (!!address && !!model && model.owner.toLowerCase() === address.toLowerCase());
  const hasAccess = USE_MOCK_DATA
    ? (isOwner || (model?.isPublic ?? false))
    : (hasAccessChain === true);

  return {
    model,
    isLoading,
    isError,
    isOwner,
    hasAccess,
    isConnected: USE_MOCK_DATA ? true : isConnected,
    isContractDeployed: USE_MOCK_DATA ? true : isContractDeployed,
    refetch,
  };
}
