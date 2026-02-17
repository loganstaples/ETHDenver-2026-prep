import { NextRequest, NextResponse } from 'next/server';
import { Indexer, ZgFile } from '@0glabs/0g-ts-sdk';
import { ethers } from 'ethers';
import { writeFile, unlink, mkdtemp } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';

// 0G Storage testnet configuration
const ZG_RPC = process.env.ZG_RPC_URL || 'https://evmrpc-testnet.0g.ai';
const ZG_INDEXER = process.env.ZG_INDEXER_URL || 'https://indexer-storage-testnet-turbo.0g.ai';
const ZG_PRIVATE_KEY = process.env.ZG_PRIVATE_KEY || '';

export async function POST(req: NextRequest) {
  try {
    if (!ZG_PRIVATE_KEY) {
      return NextResponse.json(
        { error: 'ZG_PRIVATE_KEY not configured. Set it in .env.local to enable 0G Storage.' },
        { status: 503 },
      );
    }

    const body = await req.json();
    const { session_id, weights, accuracy, version } = body;

    if (!session_id || !weights) {
      return NextResponse.json(
        { error: 'session_id and weights are required' },
        { status: 400 },
      );
    }

    // Build the model artifact JSON
    const modelArtifact = {
      version: version || '1.0.0',
      framework: 'helix-mpc',
      session_id,
      accuracy,
      timestamp: new Date().toISOString(),
      architecture: [784, 128, 10],
      weights,
    };

    // Write to a temp file (0G SDK needs a file path)
    const tmpDir = await mkdtemp(join(tmpdir(), 'helix-0g-'));
    const tmpPath = join(tmpDir, `helix-model-${session_id.slice(0, 8)}.json`);
    await writeFile(tmpPath, JSON.stringify(modelArtifact));

    // Initialize 0G Storage client
    const provider = new ethers.JsonRpcProvider(ZG_RPC);
    const signer = new ethers.Wallet(ZG_PRIVATE_KEY, provider);
    const indexer = new Indexer(ZG_INDEXER);

    // Upload to 0G Storage
    const file = await ZgFile.fromFilePath(tmpPath);
    const [tree, treeErr] = await file.merkleTree();

    if (treeErr || !tree) {
      await file.close();
      await unlink(tmpPath).catch(() => {});
      return NextResponse.json(
        { error: `Failed to compute merkle tree: ${treeErr}` },
        { status: 500 },
      );
    }

    const rootHash = tree.rootHash();

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [tx, uploadErr] = await indexer.upload(file, ZG_RPC, signer as any);
    await file.close();
    await unlink(tmpPath).catch(() => {});

    if (uploadErr || !tx) {
      return NextResponse.json(
        { error: `Upload failed: ${uploadErr}` },
        { status: 500 },
      );
    }

    return NextResponse.json({
      status: 'stored',
      root_hash: rootHash,
      tx_hash: typeof tx === 'object' && tx !== null ? (tx as { txHash?: string }).txHash : tx,
      explorer_url: `https://storagescan-galileo.0g.ai/file/${rootHash}`,
      retrieval_url: `${ZG_INDEXER}/file/${rootHash}`,
      session_id,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[0G Storage] Upload error:', message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
