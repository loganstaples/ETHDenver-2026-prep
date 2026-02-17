import { NextRequest, NextResponse } from 'next/server';
import { Indexer } from '@0glabs/0g-ts-sdk';
import { readFile, unlink, rmdir, mkdtemp } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';
const ZG_INDEXER = process.env.ZG_INDEXER_URL || 'https://indexer-storage-testnet-turbo.0g.ai';
const BACKEND_TIMEOUT_MS = 15_000;

// ============================================================================
// Types
// ============================================================================

interface ModelWeights {
  w1: number[];
  b1: number[];
  w2: number[];
  b2: number[];
}

// In-memory cache to avoid re-downloading for repeated inference
const modelCache = new Map<string, ModelWeights>();

// ============================================================================
// Neural Network Forward Pass
// ============================================================================

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map((l) => Math.exp(l - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map((e) => e / sumExp);
}

function flatten(arr: number[] | number[][]): number[] {
  if (arr.length > 0 && Array.isArray(arr[0])) {
    return (arr as number[][]).flat();
  }
  return arr as number[];
}

function forwardPass(weights: ModelWeights, pixels: number[]): number[] {
  const { w1, b1, w2, b2 } = weights;
  const inputSize = pixels.length;
  const hiddenSize = b1.length;
  const outputSize = b2.length;

  // Layer 1: hidden = ReLU(W1 * input + b1)
  const hidden = new Array(hiddenSize);
  for (let i = 0; i < hiddenSize; i++) {
    let sum = b1[i];
    for (let j = 0; j < inputSize; j++) {
      sum += w1[i * inputSize + j] * pixels[j];
    }
    hidden[i] = Math.max(0, sum); // ReLU
  }

  // Layer 2: output = Softmax(W2 * hidden + b2)
  const logits = new Array(outputSize);
  for (let i = 0; i < outputSize; i++) {
    let sum = b2[i];
    for (let j = 0; j < hiddenSize; j++) {
      sum += w2[i * hiddenSize + j] * hidden[j];
    }
    logits[i] = sum;
  }

  return softmax(logits);
}

// ============================================================================
// Weight Extraction
// ============================================================================

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function extractWeights(raw: any): ModelWeights {
  if (raw.w1 && raw.b1 && raw.w2 && raw.b2) {
    return {
      w1: flatten(raw.w1),
      b1: flatten(raw.b1),
      w2: flatten(raw.w2),
      b2: flatten(raw.b2),
    };
  }

  if (raw.layers && Array.isArray(raw.layers) && raw.layers.length >= 2) {
    return {
      w1: flatten(raw.layers[0].weights || raw.layers[0].w),
      b1: flatten(raw.layers[0].bias || raw.layers[0].b),
      w2: flatten(raw.layers[1].weights || raw.layers[1].w),
      b2: flatten(raw.layers[1].bias || raw.layers[1].b),
    };
  }

  throw new Error('Unrecognized weight format. Expected { w1, b1, w2, b2 } or { layers: [...] }');
}

// ============================================================================
// Model Loading
// ============================================================================

async function loadModelFromBackend(sessionId: string): Promise<ModelWeights> {
  if (modelCache.has(`backend:${sessionId}`)) {
    return modelCache.get(`backend:${sessionId}`)!;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE}/api/training/sessions/${sessionId}/model`, {
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error(`Backend returned HTTP ${res.status}`);
    }

    const data = await res.json();
    const weights = extractWeights(data.weights || data);
    modelCache.set(`backend:${sessionId}`, weights);
    return weights;
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('Backend model fetch timed out (15s)');
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

async function loadModelFrom0G(rootHash: string): Promise<ModelWeights> {
  if (modelCache.has(`0g:${rootHash}`)) {
    return modelCache.get(`0g:${rootHash}`)!;
  }

  const indexer = new Indexer(ZG_INDEXER);
  const tmpDir = await mkdtemp(join(tmpdir(), 'helix-infer-'));
  const tmpPath = join(tmpDir, 'model.json');

  try {
    // 0G SDK returns Error objects instead of throwing — must check return value
    const downloadResult = await indexer.download(rootHash, tmpPath, true);
    if (downloadResult instanceof Error) {
      throw new Error(`0G download failed: ${downloadResult.message}`);
    }

    const content = await readFile(tmpPath, 'utf-8');
    const artifact = JSON.parse(content);
    const weights = extractWeights(artifact.weights || artifact);
    modelCache.set(`0g:${rootHash}`, weights);
    return weights;
  } finally {
    await unlink(tmpPath).catch(() => {});
    await rmdir(tmpDir).catch(() => {});
  }
}

// ============================================================================
// API Handler
// ============================================================================

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { session_id, root_hash, pixels } = body;

    if (!pixels || !Array.isArray(pixels) || pixels.length !== 784) {
      return NextResponse.json(
        { error: 'pixels must be an array of 784 numbers (28x28 image)' },
        { status: 400 },
      );
    }

    if (!session_id && !root_hash) {
      return NextResponse.json(
        { error: 'Either session_id or root_hash is required' },
        { status: 400 },
      );
    }

    let weights: ModelWeights;
    let modelSource: string;
    let cached = false;

    // Prefer 0G Storage if root_hash is provided
    if (root_hash) {
      cached = modelCache.has(`0g:${root_hash}`);
      try {
        weights = await loadModelFrom0G(root_hash);
        modelSource = '0g-storage';
      } catch {
        // Fallback to backend if 0G fails and session_id available
        if (session_id) {
          cached = modelCache.has(`backend:${session_id}`);
          weights = await loadModelFromBackend(session_id);
          modelSource = 'backend-fallback';
        } else {
          throw new Error('Failed to load model from 0G Storage');
        }
      }
    } else {
      cached = modelCache.has(`backend:${session_id}`);
      weights = await loadModelFromBackend(session_id);
      modelSource = 'backend';
    }

    const probabilities = forwardPass(weights, pixels);
    const prediction = probabilities.indexOf(Math.max(...probabilities));
    const confidence = probabilities[prediction];

    return NextResponse.json({
      prediction,
      confidence,
      probabilities,
      model_source: modelSource,
      cached,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[Inference] Error:', message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
