/**
 * In-memory per-model stats store (singleton).
 * Tracks inference counts and user ratings for the marketplace.
 * Follows the same pattern as inference-cache.ts.
 */

interface ModelStats {
  inferenceCount: number;
  ratings: Map<string, number>; // wallet address → rating (1-5)
}

// Global singleton (survives across API route invocations in the same process)
const store = new Map<string, ModelStats>();

function getOrCreate(tokenId: string): ModelStats {
  let stats = store.get(tokenId);
  if (!stats) {
    stats = { inferenceCount: 0, ratings: new Map() };
    store.set(tokenId, stats);
  }
  return stats;
}

function computeAverage(ratings: Map<string, number>): number {
  if (ratings.size === 0) return 0;
  let sum = 0;
  for (const r of ratings.values()) sum += r;
  return sum / ratings.size;
}

export function incrementInferenceCount(tokenId: string): void {
  getOrCreate(tokenId).inferenceCount++;
}

export function submitRating(
  tokenId: string,
  wallet: string,
  rating: number,
): { averageRating: number; ratingCount: number } {
  const stats = getOrCreate(tokenId);
  stats.ratings.set(wallet.toLowerCase(), rating);
  return {
    averageRating: computeAverage(stats.ratings),
    ratingCount: stats.ratings.size,
  };
}

export function getUserRating(tokenId: string, wallet: string): number | null {
  const stats = store.get(tokenId);
  if (!stats) return null;
  return stats.ratings.get(wallet.toLowerCase()) ?? null;
}

export interface StatsSnapshot {
  inferenceCount: number;
  averageRating: number;
  ratingCount: number;
}

export function getModelStats(tokenId: string): StatsSnapshot {
  const stats = store.get(tokenId);
  if (!stats) return { inferenceCount: 0, averageRating: 0, ratingCount: 0 };
  return {
    inferenceCount: stats.inferenceCount,
    averageRating: computeAverage(stats.ratings),
    ratingCount: stats.ratings.size,
  };
}

export function getAllModelStats(): Record<string, StatsSnapshot> {
  const result: Record<string, StatsSnapshot> = {};
  for (const [tokenId, stats] of store.entries()) {
    result[tokenId] = {
      inferenceCount: stats.inferenceCount,
      averageRating: computeAverage(stats.ratings),
      ratingCount: stats.ratings.size,
    };
  }
  return result;
}
