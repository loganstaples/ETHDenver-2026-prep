/**
 * Per-model stats store with JSON file persistence.
 * Tracks inference counts and user ratings for the marketplace.
 * Data persists at ./helix-data/model-stats.json (relative to dashboard cwd).
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';

const DATA_DIR = join(process.cwd(), 'helix-data');
const DATA_FILE = join(DATA_DIR, 'model-stats.json');

interface PersistedModelStats {
  inferenceCount: number;
  ratings: Record<string, number>; // wallet address → rating (1-5)
}

type StoreData = Record<string, PersistedModelStats>;

let store: StoreData | null = null;

function load(): StoreData {
  if (store !== null) return store;
  if (existsSync(DATA_FILE)) {
    try {
      store = JSON.parse(readFileSync(DATA_FILE, 'utf-8')) as StoreData;
    } catch {
      store = {};
    }
  } else {
    store = {};
  }
  return store;
}

function save(): void {
  mkdirSync(DATA_DIR, { recursive: true });
  writeFileSync(DATA_FILE, JSON.stringify(store, null, 2), 'utf-8');
}

function getOrCreate(tokenId: string): PersistedModelStats {
  const data = load();
  if (!data[tokenId]) {
    data[tokenId] = { inferenceCount: 0, ratings: {} };
  }
  return data[tokenId];
}

function computeAverage(ratings: Record<string, number>): number {
  const values = Object.values(ratings);
  if (values.length === 0) return 0;
  let sum = 0;
  for (const r of values) sum += r;
  return sum / values.length;
}

export function incrementInferenceCount(tokenId: string): void {
  getOrCreate(tokenId).inferenceCount++;
  save();
}

export function submitRating(
  tokenId: string,
  wallet: string,
  rating: number,
): { averageRating: number; ratingCount: number } {
  const stats = getOrCreate(tokenId);
  stats.ratings[wallet.toLowerCase()] = rating;
  save();
  return {
    averageRating: computeAverage(stats.ratings),
    ratingCount: Object.keys(stats.ratings).length,
  };
}

export function getUserRating(tokenId: string, wallet: string): number | null {
  const data = load();
  const stats = data[tokenId];
  if (!stats) return null;
  return stats.ratings[wallet.toLowerCase()] ?? null;
}

export interface StatsSnapshot {
  inferenceCount: number;
  averageRating: number;
  ratingCount: number;
}

export function getModelStats(tokenId: string): StatsSnapshot {
  const data = load();
  const stats = data[tokenId];
  if (!stats) return { inferenceCount: 0, averageRating: 0, ratingCount: 0 };
  return {
    inferenceCount: stats.inferenceCount,
    averageRating: computeAverage(stats.ratings),
    ratingCount: Object.keys(stats.ratings).length,
  };
}

export function getAllModelStats(): Record<string, StatsSnapshot> {
  const data = load();
  const result: Record<string, StatsSnapshot> = {};
  for (const [tokenId, stats] of Object.entries(data)) {
    result[tokenId] = {
      inferenceCount: stats.inferenceCount,
      averageRating: computeAverage(stats.ratings),
      ratingCount: Object.keys(stats.ratings).length,
    };
  }
  return result;
}
