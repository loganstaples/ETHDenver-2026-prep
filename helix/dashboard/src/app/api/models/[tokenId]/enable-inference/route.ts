import { NextRequest, NextResponse } from 'next/server';
import { cacheWeights } from '@/lib/inference-cache';

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    const { tokenId } = await params;
    const body = await req.json();
    const { weights, version, ownerAddress } = body;

    if (!weights) {
      return NextResponse.json({ error: 'weights are required' }, { status: 400 });
    }
    if (version === undefined || version === null) {
      return NextResponse.json({ error: 'version is required' }, { status: 400 });
    }
    if (!ownerAddress) {
      return NextResponse.json({ error: 'ownerAddress is required' }, { status: 400 });
    }

    const sessionId = cacheWeights(tokenId, version, weights, ownerAddress);

    return NextResponse.json({
      status: 'cached',
      sessionId,
      tokenId,
      version,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
