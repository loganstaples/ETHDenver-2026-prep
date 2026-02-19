import { NextRequest, NextResponse } from 'next/server';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';
const BACKEND_TIMEOUT_MS = 30_000;

// ============================================================================
// API Handler — Proxy to Rust MPC inference backend
// Supports both session-based (legacy) and model-based (new) inference.
// ============================================================================

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { session_id, model_token_id, model_version_index, pixels, num_parties, wallet_address, payment_tx } = body;

    if (!pixels || !Array.isArray(pixels) || pixels.length !== 784) {
      return NextResponse.json(
        { error: 'pixels must be an array of 784 numbers (28x28 image)' },
        { status: 400 },
      );
    }

    // Determine which backend endpoint to use
    const isModelBased = model_token_id !== undefined && model_token_id !== null;

    if (!isModelBased && !session_id) {
      return NextResponse.json(
        { error: 'session_id or model_token_id is required' },
        { status: 400 },
      );
    }

    const backendUrl = isModelBased
      ? `${API_BASE}/api/models/inference`
      : `${API_BASE}/api/inference`;

    const backendBody = isModelBased
      ? {
          model_token_id,
          model_version_index: model_version_index ?? 0,
          pixels,
          num_parties: num_parties || 3,
          wallet_address,
          payment_tx,
        }
      : {
          session_id,
          pixels,
          num_parties: num_parties || 3,
        };

    // Forward to Rust backend
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), BACKEND_TIMEOUT_MS);

    try {
      const res = await fetch(backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(backendBody),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ error: `Backend HTTP ${res.status}` }));
        return NextResponse.json(
          { error: errBody.error || `Backend returned HTTP ${res.status}` },
          { status: res.status },
        );
      }

      const data = await res.json();
      return NextResponse.json(data);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        return NextResponse.json(
          { error: 'MPC inference timed out (30s)' },
          { status: 504 },
        );
      }
      throw err;
    } finally {
      clearTimeout(timeout);
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[Inference] Error:', message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
