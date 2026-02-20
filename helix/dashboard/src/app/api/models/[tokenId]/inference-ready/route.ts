import { NextRequest, NextResponse } from 'next/server';
import { isReady } from '@/lib/inference-cache';

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    const { tokenId } = await params;
    const url = new URL(req.url);
    const version = parseInt(url.searchParams.get('version') || '0', 10);

    const result = isReady(tokenId, version);
    return NextResponse.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
