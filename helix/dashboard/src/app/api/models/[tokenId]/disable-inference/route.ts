import { NextRequest, NextResponse } from 'next/server';
import { clearCache } from '@/lib/inference-cache';

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    const { tokenId } = await params;
    const url = new URL(req.url);
    const version = url.searchParams.get('version');

    const deleted = clearCache(
      tokenId,
      version !== null ? parseInt(version, 10) : undefined,
    );

    return NextResponse.json({
      status: deleted ? 'cleared' : 'not_found',
      tokenId,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
