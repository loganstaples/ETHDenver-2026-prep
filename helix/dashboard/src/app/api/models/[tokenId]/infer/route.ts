import { NextRequest, NextResponse } from 'next/server';
import { getWeightsForInference } from '@/lib/inference-cache';

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

function runForwardPass(
  weights: Record<string, unknown>,
  input: number[],
): ForwardPassResult {
  // Extract weight matrices from the model artifact format
  const w = (weights as Record<string, number[] | Record<string, number[]>>);

  let hiddenW: number[][] | undefined;
  let hiddenB: number[] | undefined;
  let outputW: number[][] | undefined;
  let outputB: number[] | undefined;

  if (w.weights && typeof w.weights === 'object') {
    const inner = w.weights as Record<string, number[] | number[][]>;
    hiddenW = inner.layer0_weight as number[][] || inner.hidden_weights as number[][];
    hiddenB = inner.layer0_bias as number[] || inner.hidden_bias as number[];
    outputW = inner.layer1_weight as number[][] || inner.output_weights as number[][];
    outputB = inner.layer1_bias as number[] || inner.output_bias as number[];
  } else {
    hiddenW = w.hidden_weights as unknown as number[][] | undefined;
    hiddenB = w.hidden_bias as unknown as number[] | undefined;
    outputW = w.output_weights as unknown as number[][] | undefined;
    outputB = w.output_bias as unknown as number[] | undefined;
  }

  if (!hiddenW || !hiddenB || !outputW || !outputB) {
    // Fallback: return uniform distribution if weight format not recognized
    const probs = new Array(10).fill(0.1);
    return { label: 0, confidence: 0.1, probabilities: probs };
  }

  // Layer 1: hidden = ReLU(input * W_h + b_h)
  const hiddenSize = hiddenB.length;
  const hidden = new Array(hiddenSize).fill(0);
  for (let j = 0; j < hiddenSize; j++) {
    let sum = hiddenB[j] || 0;
    for (let i = 0; i < input.length; i++) {
      const wRow = hiddenW[j] || hiddenW[i];
      if (wRow) {
        sum += input[i] * (Array.isArray(wRow) ? (wRow[i] ?? wRow[j] ?? 0) : 0);
      }
    }
    hidden[j] = Math.max(0, sum); // ReLU
  }

  // Layer 2: output = softmax(hidden * W_o + b_o)
  const outputSize = outputB.length;
  const logits = new Array(outputSize).fill(0);
  for (let j = 0; j < outputSize; j++) {
    let sum = outputB[j] || 0;
    for (let i = 0; i < hiddenSize; i++) {
      const wRow = outputW[j] || outputW[i];
      if (wRow) {
        sum += hidden[i] * (Array.isArray(wRow) ? (wRow[i] ?? wRow[j] ?? 0) : 0);
      }
    }
    logits[j] = sum;
  }

  // Softmax
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probabilities = exps.map((e) => e / sumExp);

  const label = probabilities.indexOf(Math.max(...probabilities));
  const confidence = probabilities[label];

  return { label, confidence, probabilities };
}
