import { NextRequest, NextResponse } from 'next/server';
import { Indexer } from '@0glabs/0g-ts-sdk';
import { readFile, unlink, mkdtemp } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { gunzipSync } from 'zlib';

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

    // Read file and decompress if gzipped
    let data = await readFile(tmpPath);
    const rawSize = data.length;
    await unlink(tmpPath).catch(() => {});

    // Detect gzip magic bytes (1f 8b) and decompress.
    // We also try unconditionally as a fallback in case the magic bytes are
    // at a slight offset (e.g. sector alignment metadata from the storage layer).
    let decompressed = false;
    if (data.length >= 2 && data[0] === 0x1f && data[1] === 0x8b) {
      try {
        data = gunzipSync(data);
        decompressed = true;
        console.log(`[0G Storage] Decompressed: ${rawSize} → ${data.length} bytes`);
      } catch (gzErr) {
        console.warn('[0G Storage] Gzip decompression failed (magic bytes present):', gzErr);
        // Try again — scan for gzip header in first 256 bytes in case of prefix
        for (let i = 1; i < Math.min(data.length - 1, 256); i++) {
          if (data[i] === 0x1f && data[i + 1] === 0x8b) {
            try {
              data = gunzipSync(data.subarray(i));
              decompressed = true;
              console.log(`[0G Storage] Decompressed from offset ${i}: ${rawSize} → ${data.length} bytes`);
              break;
            } catch {
              // Continue scanning
            }
          }
        }
      }
    }

    if (!decompressed) {
      console.log(`[0G Storage] Data not gzipped (${rawSize} bytes), first bytes: ${data[0]?.toString(16)} ${data[1]?.toString(16)}`);
    }

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
