import { NextRequest, NextResponse } from 'next/server';
import { getWeightsForInference } from '@/lib/inference-cache';
import { incrementInferenceCount } from '@/lib/model-stats';

/**
 * Run inference on cached model weights.
 * Returns prediction results only — never returns the weights themselves.
 *
 * For now this performs a simple forward pass for MNIST-style networks.
 * The weights are expected to have the format from MPC training sessions.
 */
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    const { tokenId } = await params;
    const body = await req.json();
    const { input, version } = body;

    if (!input || !Array.isArray(input)) {
      return NextResponse.json({ error: 'input array is required' }, { status: 400 });
    }
    if (version === undefined) {
      return NextResponse.json({ error: 'version is required' }, { status: 400 });
    }

    const weights = getWeightsForInference(tokenId, version);
    if (!weights) {
      return NextResponse.json(
        { error: 'Model not cached for inference. Owner must enable inference first.' },
        { status: 404 },
      );
    }

    // Simple forward pass for MNIST-style networks (784 → hidden → 10)
    const prediction = runForwardPass(weights, input);

    // Track inference count for marketplace stats
    incrementInferenceCount(tokenId);

    return NextResponse.json({
      prediction: prediction.label,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      tokenId,
      version,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

interface ForwardPassResult {
  label: number;
  confidence: number;
  probabilities: number[];
}

/**
 * Get a weight value from either a flat 1D array or a 2D array.
 * Flat layout: W[row][col] = W[row * numCols + col]  (row-major)
 * 2D layout:   W[row][col]
 */
function getWeight(w: number[] | number[][], row: number, col: number, numCols: number): number {
  if (Array.isArray(w[0])) {
    return (w as number[][])[row]?.[col] ?? 0;
  }
  return (w as number[])[row * numCols + col] ?? 0;
}

function runForwardPass(
  weights: Record<string, unknown>,
  input: number[],
): ForwardPassResult {
  const w = weights as Record<string, unknown>;

  let hiddenW: number[] | number[][] | undefined;
  let hiddenB: number[] | undefined;
  let outputW: number[] | number[][] | undefined;
  let outputB: number[] | undefined;

  // Format 1: MPC training output { w1, b1, w2, b2 } (flat arrays)
  if (w.w1 && w.b1 && w.w2 && w.b2) {
    hiddenW = w.w1 as number[];
    hiddenB = w.b1 as number[];
    outputW = w.w2 as number[];
    outputB = w.b2 as number[];
  }
  // Format 2: Nested { weights: { layer0_weight, layer0_bias, ... } }
  else if (w.weights && typeof w.weights === 'object') {
    const inner = w.weights as Record<string, unknown>;
    hiddenW = (inner.layer0_weight ?? inner.hidden_weights) as number[] | number[][] | undefined;
    hiddenB = (inner.layer0_bias ?? inner.hidden_bias) as number[] | undefined;
    outputW = (inner.layer1_weight ?? inner.output_weights) as number[] | number[][] | undefined;
    outputB = (inner.layer1_bias ?? inner.output_bias) as number[] | undefined;
  }
  // Format 3: Top-level { hidden_weights, hidden_bias, ... }
  else {
    hiddenW = w.hidden_weights as number[] | number[][] | undefined;
    hiddenB = w.hidden_bias as number[] | undefined;
    outputW = w.output_weights as number[] | number[][] | undefined;
    outputB = w.output_bias as number[] | undefined;
  }

  if (!hiddenW || !hiddenB || !outputW || !outputB) {
    throw new Error('Weight format not recognized. Please re-upload weights in the standard format.');
  }

  const inputSize = input.length;
  const hiddenSize = hiddenB.length;
  const outputSize = outputB.length;

  // Layer 1: hidden = ReLU(W_h * input + b_h)
  // W_h is [hiddenSize x inputSize], row j = weights for hidden unit j
  const hidden = new Array(hiddenSize);
  for (let j = 0; j < hiddenSize; j++) {
    let sum = hiddenB[j] ?? 0;
    for (let i = 0; i < inputSize; i++) {
      sum += input[i] * getWeight(hiddenW, j, i, inputSize);
    }
    hidden[j] = Math.max(0, sum); // ReLU
  }

  // Layer 2: logits = W_o * hidden + b_o
  // W_o is [outputSize x hiddenSize], row j = weights for output unit j
  const logits = new Array(outputSize);
  for (let j = 0; j < outputSize; j++) {
    let sum = outputB[j] ?? 0;
    for (let i = 0; i < hiddenSize; i++) {
      sum += hidden[i] * getWeight(outputW, j, i, hiddenSize);
    }
    logits[j] = sum;
  }

  // Softmax
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l: number) => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((a: number, b: number) => a + b, 0);
  const probabilities = exps.map((e: number) => e / sumExp);

  const label = probabilities.indexOf(Math.max(...probabilities));
  const confidence = probabilities[label];

  return { label, confidence, probabilities };
}
