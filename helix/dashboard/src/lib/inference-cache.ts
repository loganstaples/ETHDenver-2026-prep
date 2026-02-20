/**
 * In-memory cache for decrypted model weights.
 * Owner uploads decrypted weights here for public inference.
 * Weights are never persisted to disk or returned via API.
 */

export interface CachedModel {
  tokenId: string;
  version: number;
  weights: Record<string, unknown>;
  sessionId: string;
  cachedAt: number;
  ownerAddress: string;
}

// Global singleton cache (survives across API route invocations in the same process)
const cache = new Map<string, CachedModel>();

function cacheKey(tokenId: string, version: number): string {
  return `${tokenId}-${version}`;
}

export function cacheWeights(
  tokenId: string,
  version: number,
  weights: Record<string, unknown>,
  ownerAddress: string,
): string {
  const sessionId = `inference-${tokenId}-${version}-${Date.now()}`;
  const key = cacheKey(tokenId, version);
  cache.set(key, {
    tokenId,
    version,
    weights,
    sessionId,
    cachedAt: Date.now(),
    ownerAddress,
  });
  return sessionId;
}

export function clearCache(tokenId: string, version?: number): boolean {
  if (version !== undefined) {
    return cache.delete(cacheKey(tokenId, version));
  }
  // Clear all versions for this tokenId
  let deleted = false;
  for (const key of cache.keys()) {
    if (key.startsWith(`${tokenId}-`)) {
      cache.delete(key);
      deleted = true;
    }
  }
  return deleted;
}

export function getCached(tokenId: string, version: number): CachedModel | undefined {
  return cache.get(cacheKey(tokenId, version));
}

export function isReady(tokenId: string, version: number): { ready: boolean; sessionId?: string } {
  const cached = cache.get(cacheKey(tokenId, version));
  if (cached) {
    return { ready: true, sessionId: cached.sessionId };
  }
  return { ready: false };
}

/**
 * Get the weights for inference (internal use only — never expose via API response).
 * Returns the raw weight object for running inference server-side.
 */
export function getWeightsForInference(tokenId: string, version: number): Record<string, unknown> | undefined {
  return cache.get(cacheKey(tokenId, version))?.weights;
}
