import { NextRequest, NextResponse } from 'next/server';

interface RelayEntry {
  weights: Record<string, unknown>;
  sellerAddress: string;
  buyerAddress: string;
  createdAt: number;
}

// In-memory relay store (auto-clears after timeout)
const relay = new Map<string, RelayEntry>();
const RELAY_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes

// Cleanup expired entries periodically
function cleanupExpired() {
  const now = Date.now();
  for (const [key, entry] of relay.entries()) {
    if (now - entry.createdAt > RELAY_TIMEOUT_MS) {
      relay.delete(key);
    }
  }
}

/** POST: Seller deposits decrypted weights for buyer to pick up */
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    cleanupExpired();
    const { tokenId } = await params;
    const body = await req.json();
    const { weights, sellerAddress, buyerAddress } = body;

    if (!weights) {
      return NextResponse.json({ error: 'weights are required' }, { status: 400 });
    }
    if (!sellerAddress || !buyerAddress) {
      return NextResponse.json({ error: 'sellerAddress and buyerAddress are required' }, { status: 400 });
    }

    relay.set(tokenId, {
      weights,
      sellerAddress: sellerAddress.toLowerCase(),
      buyerAddress: buyerAddress.toLowerCase(),
      createdAt: Date.now(),
    });

    return NextResponse.json({
      status: 'deposited',
      tokenId,
      expiresIn: RELAY_TIMEOUT_MS,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

/** GET: Buyer fetches decrypted weights (does NOT auto-delete — buyer calls DELETE after confirming upload) */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    cleanupExpired();
    const { tokenId } = await params;
    const url = new URL(req.url);
    const buyerAddress = url.searchParams.get('buyer')?.toLowerCase();

    if (!buyerAddress) {
      return NextResponse.json({ error: 'buyer query param is required' }, { status: 400 });
    }

    const entry = relay.get(tokenId);
    if (!entry) {
      return NextResponse.json({ error: 'No pending transfer relay for this model' }, { status: 404 });
    }

    if (entry.buyerAddress !== buyerAddress) {
      return NextResponse.json({ error: 'Not authorized — buyer address mismatch' }, { status: 403 });
    }

    return NextResponse.json({
      status: 'retrieved',
      weights: entry.weights,
      tokenId,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

/** DELETE: Buyer confirms upload succeeded, clear relay entry */
export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    const { tokenId } = await params;
    const url = new URL(req.url);
    const buyerAddress = url.searchParams.get('buyer')?.toLowerCase();

    if (!buyerAddress) {
      return NextResponse.json({ error: 'buyer query param is required' }, { status: 400 });
    }

    const entry = relay.get(tokenId);
    if (!entry) {
      return NextResponse.json({ status: 'already_cleared' });
    }

    if (entry.buyerAddress !== buyerAddress) {
      return NextResponse.json({ error: 'Not authorized — buyer address mismatch' }, { status: 403 });
    }

    relay.delete(tokenId);
    return NextResponse.json({ status: 'cleared', tokenId });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
