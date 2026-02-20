import { NextRequest, NextResponse } from 'next/server';
import { submitRating, getModelStats, getUserRating } from '@/lib/model-stats';

/** POST /api/models/{tokenId}/rate — submit a rating (1-5) */
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    const { tokenId } = await params;
    const body = await req.json();
    const { wallet, rating } = body;

    if (!wallet || typeof wallet !== 'string') {
      return NextResponse.json({ error: 'wallet address is required' }, { status: 400 });
    }
    if (typeof rating !== 'number' || rating < 1 || rating > 5 || !Number.isInteger(rating)) {
      return NextResponse.json({ error: 'rating must be an integer from 1 to 5' }, { status: 400 });
    }

    const result = submitRating(tokenId, wallet, rating);
    return NextResponse.json({ tokenId, ...result });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

/** GET /api/models/{tokenId}/rate?wallet={address} — get user's rating + aggregates */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    const { tokenId } = await params;
    const url = new URL(req.url);
    const wallet = url.searchParams.get('wallet');

    const stats = getModelStats(tokenId);
    const userRating = wallet ? getUserRating(tokenId, wallet) : null;

    return NextResponse.json({
      tokenId,
      ...stats,
      userRating,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
