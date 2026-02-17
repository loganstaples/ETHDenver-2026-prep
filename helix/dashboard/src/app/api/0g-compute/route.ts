import { NextRequest, NextResponse } from 'next/server';
import { ethers } from 'ethers';
import { createZGComputeNetworkBroker } from '@0glabs/0g-serving-broker';

// 0G Compute Network configuration
const ZG_COMPUTE_RPC = process.env.ZG_COMPUTE_RPC || 'https://evmrpc-testnet.0g.ai';
const ZG_COMPUTE_KEY = process.env.ZG_COMPUTE_PRIVATE_KEY || process.env.ZG_PRIVATE_KEY || '';

// Default to Qwen 2.5 7B on testnet
const DEFAULT_PROVIDER = process.env.ZG_COMPUTE_PROVIDER || '0xa48f01287233509FD694a22Bf840225062E67836';

// Singleton broker (reuse across requests)
let brokerPromise: ReturnType<typeof createZGComputeNetworkBroker> | null = null;

function getBroker() {
  if (!brokerPromise) {
    const provider = new ethers.JsonRpcProvider(ZG_COMPUTE_RPC);
    const wallet = new ethers.Wallet(ZG_COMPUTE_KEY, provider);
    brokerPromise = createZGComputeNetworkBroker(wallet);
  }
  return brokerPromise;
}

// ============================================================================
// Pixel → ASCII Art conversion for LLM digit classification
// ============================================================================

function pixelsToAsciiArt(pixels: number[]): string {
  const chars = ' .:-=+*#%@';
  const rows: string[] = [];
  for (let y = 0; y < 28; y++) {
    let row = '';
    for (let x = 0; x < 28; x++) {
      const val = pixels[y * 28 + x];
      const idx = Math.min(chars.length - 1, Math.floor(val * (chars.length - 1)));
      row += chars[idx];
    }
    rows.push(row);
  }
  return rows.join('\n');
}

function buildClassificationPrompt(pixels: number[]): string {
  const ascii = pixelsToAsciiArt(pixels);
  return `You are an expert at recognizing handwritten digits. Below is a 28x28 grayscale image of a handwritten digit rendered as ASCII art, where ' ' (space) means white/background and '@' means darkest/ink.

\`\`\`
${ascii}
\`\`\`

What single digit (0-9) is shown in this image? Think step by step about the shape, then respond with ONLY a JSON object in this exact format:
{"digit": <number>, "confidence": <0-100>}

Do not include any other text.`;
}

// ============================================================================
// 0G Compute Setup (ledger + provider acknowledgment)
// ============================================================================

async function ensureSetup(broker: Awaited<ReturnType<typeof createZGComputeNetworkBroker>>, providerAddress: string) {
  // Check/create ledger
  try {
    await broker.ledger.getLedger();
  } catch {
    // Ledger doesn't exist — create with 3 OG minimum
    await broker.ledger.addLedger(3);
  }

  // Acknowledge provider (one-time on-chain tx)
  try {
    const isAcked = await broker.inference.acknowledged(providerAddress);
    if (!isAcked) {
      await broker.inference.acknowledgeProviderSigner(providerAddress);
    }
  } catch {
    // May already be acknowledged
  }

  // Transfer funds to provider if needed
  try {
    const account = await broker.inference.getAccount(providerAddress);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const balance = (account as any)?.balance ?? BigInt(0);
    if (BigInt(balance) < ethers.parseEther('0.1')) {
      await broker.ledger.transferFund(
        providerAddress,
        'inference',
        ethers.parseEther('1.0'),
      );
    }
  } catch {
    // Try transferring anyway
    try {
      await broker.ledger.transferFund(
        providerAddress,
        'inference',
        ethers.parseEther('1.0'),
      );
    } catch {
      // May already have funds
    }
  }
}

// ============================================================================
// API Handler
// ============================================================================

export async function POST(req: NextRequest) {
  try {
    if (!ZG_COMPUTE_KEY) {
      return NextResponse.json(
        { error: 'ZG_COMPUTE_PRIVATE_KEY not configured. Set it in .env.local to enable 0G Compute.' },
        { status: 503 },
      );
    }

    const body = await req.json();
    const { pixels } = body;

    if (!pixels || !Array.isArray(pixels) || pixels.length !== 784) {
      return NextResponse.json(
        { error: 'pixels must be an array of 784 numbers (28x28 image)' },
        { status: 400 },
      );
    }

    const providerAddress = DEFAULT_PROVIDER;
    const broker = await getBroker();

    // Ensure ledger + provider setup
    await ensureSetup(broker, providerAddress);

    // Get service endpoint and model
    const { endpoint, model } = await broker.inference.getServiceMetadata(providerAddress);

    // Build the classification prompt
    const prompt = buildClassificationPrompt(pixels);

    // Generate single-use auth headers
    const headers = await broker.inference.getRequestHeaders(providerAddress, prompt);
    const requestHeaders: Record<string, string> = { 'Content-Type': 'application/json' };
    for (const [key, value] of Object.entries(headers)) {
      if (typeof value === 'string') {
        requestHeaders[key] = value;
      }
    }

    // Make OpenAI-compatible request to 0G Compute provider
    const response = await fetch(`${endpoint}/chat/completions`, {
      method: 'POST',
      headers: requestHeaders,
      body: JSON.stringify({
        model,
        messages: [
          { role: 'user', content: prompt },
        ],
        temperature: 0.1,
        max_tokens: 50,
      }),
    });

    if (!response.ok) {
      const errText = await response.text().catch(() => '');
      throw new Error(`0G Compute returned HTTP ${response.status}: ${errText.slice(0, 200)}`);
    }

    const completion = await response.json();
    const content = completion.choices?.[0]?.message?.content || '';
    const chatId = completion.id || '';

    // Process response (verify + settle payment)
    try {
      await broker.inference.processResponse(providerAddress, chatId, content);
    } catch {
      // Non-fatal: response is still valid even if settlement fails
      console.warn('[0G Compute] processResponse failed, continuing');
    }

    // Parse the LLM's classification response
    let prediction = -1;
    let confidence = 0;

    // Try to parse JSON response
    const jsonMatch = content.match(/\{[^}]*"digit"\s*:\s*(\d)\s*[^}]*"confidence"\s*:\s*(\d+)/);
    if (jsonMatch) {
      prediction = parseInt(jsonMatch[1], 10);
      confidence = parseInt(jsonMatch[2], 10) / 100;
    } else {
      // Fallback: find first single digit in response
      const digitMatch = content.match(/\b(\d)\b/);
      if (digitMatch) {
        prediction = parseInt(digitMatch[1], 10);
        confidence = 0.5; // Unknown confidence
      }
    }

    if (prediction < 0 || prediction > 9) {
      throw new Error(`0G Compute LLM could not classify the digit. Response: ${content}`);
    }

    return NextResponse.json({
      prediction,
      confidence,
      raw_response: content,
      model_used: model,
      provider: providerAddress,
      source: '0g-compute',
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[0G Compute] Error:', message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

// ============================================================================
// GET: List available services
// ============================================================================

export async function GET() {
  try {
    if (!ZG_COMPUTE_KEY) {
      return NextResponse.json(
        { error: 'ZG_COMPUTE_PRIVATE_KEY not configured' },
        { status: 503 },
      );
    }

    const broker = await getBroker();
    const services = await broker.inference.listService();

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const formatted = (services as any[]).map((s) => ({
      provider: s.provider,
      name: s.name || s.model || 'unknown',
      url: s.url,
      inputPrice: s.inputPrice?.toString() || '0',
      outputPrice: s.outputPrice?.toString() || '0',
    }));

    return NextResponse.json({ services: formatted });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
