import { NextRequest, NextResponse } from 'next/server';
import { Indexer } from '@0glabs/0g-ts-sdk';
import { readFile, unlink, mkdtemp } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';

const ZG_INDEXER = process.env.ZG_INDEXER_URL || 'https://indexer-storage-testnet-turbo.0g.ai';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { rootHash } = body;

    if (!rootHash) {
      return NextResponse.json(
        { error: 'rootHash is required' },
        { status: 400 },
      );
    }

    const indexer = new Indexer(ZG_INDEXER);

    // Download from 0G to a temp file
    const tmpDir = await mkdtemp(join(tmpdir(), 'helix-0g-dl-'));
    const tmpPath = join(tmpDir, `model-${rootHash.slice(0, 12)}.bin`);

    const err = await indexer.download(rootHash, tmpPath, true);
    if (err) {
      await unlink(tmpPath).catch(() => {});
      return NextResponse.json(
        { error: `Download from 0G failed: ${err}` },
        { status: 500 },
      );
    }

    // Read file and return as base64
    const data = await readFile(tmpPath);
    await unlink(tmpPath).catch(() => {});

    // Try to parse as JSON first — if it is, return the JSON directly
    try {
      const text = data.toString('utf-8');
      const parsed = JSON.parse(text);
      return NextResponse.json({ data: parsed, encoding: 'json' });
    } catch {
      // Not JSON (probably encrypted binary) — return as base64
      const base64 = data.toString('base64');
      return NextResponse.json({ data: base64, encoding: 'base64' });
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[0G Storage] Fetch error:', message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
