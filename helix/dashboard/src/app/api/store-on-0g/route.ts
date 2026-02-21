import { NextRequest, NextResponse } from 'next/server';
import {
  Indexer,
  ZgFile,
  Uploader,
  StorageNode,
  calculatePrice,
  getMarketContract,
  txWithGasAdjustment,
} from '@0glabs/0g-ts-sdk';
import { ethers } from 'ethers';
import { writeFile, unlink, mkdtemp } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { gzipSync } from 'zlib';

// 0G Storage testnet configuration
const ZG_RPC = process.env.ZG_RPC_URL || 'https://evmrpc-testnet.0g.ai';
const ZG_INDEXER =
  process.env.ZG_INDEXER_URL ||
  'https://indexer-storage-testnet-turbo.0g.ai';
const ZG_PRIVATE_KEY = process.env.ZG_PRIVATE_KEY || '';

// Corrected ABI for the upgraded FixedPriceFlow contract (Dec 2025).
// The SDK v0.3.3 ships an outdated ABI — the on-chain contract now wraps
// (Submission, address sender) in an outer tuple for submit().
const CORRECTED_FLOW_ABI = [
  'function submit(((uint256 length, bytes tags, (bytes32 root, uint256 height)[] nodes) submission, address sender)) payable',
  'function market() view returns (address)',
  'event Submit(address indexed sender, bytes32 indexed identity, uint256 submissionIndex, uint256 startPos, uint256 length, (uint256, bytes, (bytes32, uint256)[]) submission)',
];

/**
 * Upload a file to 0G Storage using the corrected contract ABI.
 * Mirrors Indexer.upload() but patches the submit() call.
 */
async function uploadTo0G(
  file: InstanceType<typeof ZgFile>,
  signer: ethers.Wallet,
  rpcUrl: string,
  indexerUrl: string,
) {
  const indexer = new Indexer(indexerUrl);

  // 1. Discover storage nodes via indexer
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [clients, selectErr] = await (indexer as any).selectNodes(1);
  if (selectErr) throw new Error(`Node selection failed: ${selectErr}`);

  // 2. Get flow contract address from the first storage node
  const status = await clients[0].getStatus();
  if (!status) throw new Error('Failed to get storage node status');
  const flowAddress: string = status.networkIdentity.flowAddress;
  console.log(`[0G Storage] Flow contract: ${flowAddress}`);

  // 3. Create flow contract with CORRECTED ABI
  const flow = new ethers.Contract(flowAddress, CORRECTED_FLOW_ABI, signer);

  // 4. Construct Uploader with the corrected flow contract
  const uploader = new Uploader(
    clients,
    rpcUrl,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    flow as any,
  );

  // 5. Monkey-patch submitTransaction to wrap (submission, sender) in the
  //    outer tuple that the upgraded contract expects.
  const signerAddress = await signer.getAddress();
  const provider = new ethers.JsonRpcProvider(rpcUrl);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (uploader as any).submitTransaction = async function (
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    submission: any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    opts: any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    retryOpts: any,
  ) {
    const marketAddr = await flow.market();
    const marketContract = getMarketContract(
      marketAddr,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      provider as any,
    );
    const pricePerSector = await marketContract.pricePerSector();
    const fee =
      opts.fee > 0 ? opts.fee : calculatePrice(submission, pricePerSector);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const txOpts: any = { value: fee, nonce: opts.nonce };
    const suggestedGasPrice = (await provider.getFeeData()).gasPrice;
    if (!suggestedGasPrice) {
      return [null, new Error('Failed to get gas price')];
    }
    txOpts.gasPrice = suggestedGasPrice;

    console.log('[0G Storage] Submitting transaction with fee:', fee.toString());

    // Key fix: pass [submission, signerAddress] as the single tuple arg
    const [txReceipt, txErr] = await txWithGasAdjustment(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      flow as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      provider as any,
      'submit',
      [[submission, signerAddress]],
      txOpts,
      retryOpts,
    );

    if (txReceipt === null || txErr !== null) {
      return [null, new Error('Failed to submit transaction: ' + txErr)];
    }
    return [txReceipt, null];
  };

  // 6. Run the upload (merkle tree, submit tx, upload segments, finalize)
  return await uploader.uploadFile(file, {
    tags: '0x',
    finalityRequired: true,
    taskSize: 10,
    expectedReplica: 1,
    skipTx: false,
    fee: BigInt('0'),
  });
}

export async function POST(req: NextRequest) {
  try {
    if (!ZG_PRIVATE_KEY) {
      return NextResponse.json(
        {
          error:
            'ZG_PRIVATE_KEY not configured. Set it in .env.local to enable 0G Storage.',
        },
        { status: 503 },
      );
    }

    const body = await req.json();
    const { session_id, weights, accuracy, version, encrypted, encryptedPayload } =
      body;

    if (!session_id) {
      return NextResponse.json(
        { error: 'session_id is required' },
        { status: 400 },
      );
    }
    if (!encrypted && !weights) {
      return NextResponse.json(
        { error: 'weights are required for non-encrypted uploads' },
        { status: 400 },
      );
    }

    // Write to a temp file (0G SDK needs a file path)
    const tmpDir = await mkdtemp(join(tmpdir(), 'helix-0g-'));
    const tmpPath = join(tmpDir, `helix-model-${session_id.slice(0, 8)}.json`);

    if (encrypted && encryptedPayload) {
      // Gzip the encrypted binary to reduce 0G storage cost (~30-50% smaller)
      const raw = Buffer.from(encryptedPayload, 'base64');
      const compressed = gzipSync(raw);
      console.log(
        `[0G Storage] Encrypted payload: ${raw.length} bytes → gzipped: ${compressed.length} bytes (${Math.round((compressed.length / raw.length) * 100)}%)`,
      );
      await writeFile(tmpPath, compressed);
    } else {
      // Build the model artifact JSON and gzip it
      const modelArtifact = {
        version: version || '1.0.0',
        framework: 'helix-mpc',
        session_id,
        accuracy,
        timestamp: new Date().toISOString(),
        architecture: [784, 128, 10],
        weights,
      };
      const jsonBuf = Buffer.from(JSON.stringify(modelArtifact));
      const compressed = gzipSync(jsonBuf);
      console.log(
        `[0G Storage] JSON payload: ${jsonBuf.length} bytes → gzipped: ${compressed.length} bytes (${Math.round((compressed.length / jsonBuf.length) * 100)}%)`,
      );
      await writeFile(tmpPath, compressed);
    }

    // Initialize signer
    const signer = new ethers.Wallet(
      ZG_PRIVATE_KEY,
      new ethers.JsonRpcProvider(ZG_RPC),
    );

    // Prepare file and compute root hash
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

    // Upload using corrected ABI
    const [tx, uploadErr] = await uploadTo0G(file, signer, ZG_RPC, ZG_INDEXER);
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
      tx_hash:
        typeof tx === 'object' && tx !== null
          ? (tx as { txHash?: string }).txHash
          : tx,
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
