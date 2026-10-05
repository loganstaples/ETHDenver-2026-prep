# 0G AI Config Assistant Implementation Plan

**Goal:** Add a toggle on the training page that opens a modal where users describe their training goals, pay a small ADI fee, and get training config auto-filled by an AI model running on 0G Compute's decentralized GPU network.

**Architecture:** Server-side 0G broker (already exists at `/api/0g-compute/route.ts`) handles inference. New `/api/0g-compute/config-assist/route.ts` endpoint sends a system prompt for config generation. Frontend `AIConfigModal` handles the UX: text input, one wagmi native transfer for payment, loading state, and animated config apply. The modal reuses the existing `Modal` component and matches the glassmorphic dark theme.

**Tech Stack:** Next.js App Router, wagmi v2 (useSendTransaction), framer-motion, existing `@0glabs/0g-serving-broker` singleton, Lucide icons, Tailwind CSS.

---

### Task 1: Extract shared 0G broker singleton

The existing `/api/0g-compute/route.ts` has an inline broker singleton + `ensureSetup()`. Extract these to a shared module so the new config-assist route can reuse them.

**Files:**
- Create: `helix/dashboard/src/lib/0g-broker.ts`
- Modify: `helix/dashboard/src/app/api/0g-compute/route.ts`

**Step 1: Create the shared broker module**

Create `helix/dashboard/src/lib/0g-broker.ts`:

```typescript
import { ethers } from 'ethers';
import { createZGComputeNetworkBroker } from '@0glabs/0g-serving-broker';

export const ZG_COMPUTE_RPC = process.env.ZG_COMPUTE_RPC || 'https://evmrpc-testnet.0g.ai';
export const ZG_COMPUTE_KEY = process.env.ZG_COMPUTE_PRIVATE_KEY || process.env.ZG_PRIVATE_KEY || '';
export const DEFAULT_PROVIDER = process.env.ZG_COMPUTE_PROVIDER || '0xa48f01287233509FD694a22Bf840225062E67836';

let brokerPromise: ReturnType<typeof createZGComputeNetworkBroker> | null = null;

export function getBroker() {
  if (!brokerPromise) {
    const provider = new ethers.JsonRpcProvider(ZG_COMPUTE_RPC);
    const wallet = new ethers.Wallet(ZG_COMPUTE_KEY, provider);
    brokerPromise = createZGComputeNetworkBroker(wallet);
  }
  return brokerPromise;
}

export async function ensureSetup(
  broker: Awaited<ReturnType<typeof createZGComputeNetworkBroker>>,
  providerAddress: string,
) {
  try {
    await broker.ledger.getLedger();
  } catch {
    await broker.ledger.addLedger(3);
  }

  try {
    const isAcked = await broker.inference.acknowledged(providerAddress);
    if (!isAcked) {
      await broker.inference.acknowledgeProviderSigner(providerAddress);
    }
  } catch {
    // May already be acknowledged
  }

  try {
    const account = await broker.inference.getAccount(providerAddress);
    const balance = (account as any)?.balance ?? BigInt(0);
    if (BigInt(balance) < ethers.parseEther('0.001')) {
      await broker.ledger.transferFund(providerAddress, 'inference', ethers.parseEther('0.01'));
    }
  } catch {
    try {
      await broker.ledger.transferFund(providerAddress, 'inference', ethers.parseEther('0.01'));
    } catch {
      // May already have funds
    }
  }
}
```

**Step 2: Update existing route to import from shared module**

In `helix/dashboard/src/app/api/0g-compute/route.ts`, replace the inline `getBroker`, `ensureSetup`, constants, and `ethers`/`createZGComputeNetworkBroker` imports with:

```typescript
import { getBroker, ensureSetup, ZG_COMPUTE_KEY, DEFAULT_PROVIDER } from '@/lib/0g-broker';
```

Remove the duplicate code (lines 1-104 are replaced by the import + remaining route-specific code like `pixelsToAsciiArt`, `buildClassificationPrompt`, and the `POST`/`GET` handlers).

**Step 3: Verify the existing 0G Compute route still works**

Run: `cd helix/dashboard && npx next build` — or at minimum `npx tsc --noEmit`
Expected: No type errors.

**Step 4: Commit**

```bash
git add helix/dashboard/src/lib/0g-broker.ts helix/dashboard/src/app/api/0g-compute/route.ts
git commit -m "refactor: extract shared 0G broker singleton to lib/0g-broker.ts"
```

---

### Task 2: Create the config-assist API route

New API route that accepts a user message, calls 0G Compute with a system prompt for config generation, and returns parsed settings.

**Files:**
- Create: `helix/dashboard/src/app/api/0g-compute/config-assist/route.ts`

**Step 1: Create the config-assist route**

Create `helix/dashboard/src/app/api/0g-compute/config-assist/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { getBroker, ensureSetup, ZG_COMPUTE_KEY, DEFAULT_PROVIDER } from '@/lib/0g-broker';

const SYSTEM_PROMPT = `You are a helpful ML training configuration assistant for HELIX, a decentralized verifiable ML training protocol. The user is configuring an MNIST neural network (784 → hidden → 10) trained via MPC (multi-party computation) with multiple workers verifying each other.

Available settings:
- hiddenSize: Hidden layer neurons (16-512, multiples of 16). Higher = more model capacity but slower training. 64-128 is typical for MNIST.
- numSteps: Training steps (10-10000). More steps = better accuracy but longer training. 100-300 for quick tests, 500-2000 for good results.
- learningRate: Learning rate (0.00001-1). Controls how fast the model learns. 0.01-0.1 typical for MNIST MPC training.
- checkpointFreq: How often to create verifiable checkpoints (10-500). Lower = more verification but more overhead. Match roughly to numSteps/4.
- workers: Number of MPC workers (1-10). More = better security guarantees but more communication overhead. 3 is the sweet spot for demos.
- zkMode: "off", "always", or "risk". Zero-knowledge proofs for additional verification. "off" for speed, "always" for maximum security, "risk" for smart adaptive mode.
- stakePerWorker: Economic stake per worker in ADI tokens (0-1). Higher stakes = stronger economic security guarantees.

Based on the user's description, suggest optimal settings. Respond ONLY with valid JSON (no markdown, no backticks):
{"hiddenSize":<number>,"numSteps":<number>,"learningRate":<number>,"checkpointFreq":<number>,"workers":<number>,"zkMode":"<off|always|risk>","stakePerWorker":<number>,"explanation":"<1-2 sentence explanation>"}`;

export async function POST(req: NextRequest) {
  try {
    if (!ZG_COMPUTE_KEY) {
      return NextResponse.json(
        { error: 'ZG_COMPUTE_PRIVATE_KEY not configured. Set it in .env.local to enable 0G Compute.' },
        { status: 503 },
      );
    }

    const body = await req.json();
    const { message } = body;

    if (!message || typeof message !== 'string' || message.trim().length === 0) {
      return NextResponse.json({ error: 'message is required' }, { status: 400 });
    }

    const providerAddress = DEFAULT_PROVIDER;
    const broker = await getBroker();
    await ensureSetup(broker, providerAddress);

    const { endpoint, model } = await broker.inference.getServiceMetadata(providerAddress);

    const userContent = message.trim().slice(0, 1000);

    const headers = await broker.inference.getRequestHeaders(providerAddress, userContent);
    const requestHeaders: Record<string, string> = { 'Content-Type': 'application/json' };
    for (const [key, value] of Object.entries(headers)) {
      if (typeof value === 'string') {
        requestHeaders[key] = value;
      }
    }

    const response = await fetch(`${endpoint}/chat/completions`, {
      method: 'POST',
      headers: requestHeaders,
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          { role: 'user', content: userContent },
        ],
        temperature: 0.3,
        max_tokens: 300,
      }),
    });

    if (!response.ok) {
      const errText = await response.text().catch(() => '');
      throw new Error(`0G Compute returned HTTP ${response.status}: ${errText.slice(0, 200)}`);
    }

    const completion = await response.json();
    const content = completion.choices?.[0]?.message?.content || '';
    const chatId = completion.id || '';

    // Process response (verify + settle)
    try {
      await broker.inference.processResponse(providerAddress, chatId, content);
    } catch {
      console.warn('[0G Config Assist] processResponse failed, continuing');
    }

    // Parse the JSON config from the LLM response
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error(`Could not parse config from LLM response: ${content.slice(0, 200)}`);
    }

    let config;
    try {
      config = JSON.parse(jsonMatch[0]);
    } catch {
      throw new Error(`Invalid JSON in LLM response: ${jsonMatch[0].slice(0, 200)}`);
    }

    // Clamp values to valid ranges
    const result = {
      hiddenSize: Math.max(16, Math.min(512, Math.round((config.hiddenSize || 128) / 16) * 16)),
      numSteps: Math.max(10, Math.min(10000, Math.round(config.numSteps || 200))),
      learningRate: Math.max(0.00001, Math.min(1, config.learningRate || 0.05)),
      checkpointFreq: Math.max(10, Math.min(500, Math.round((config.checkpointFreq || 50) / 10) * 10)),
      workers: Math.max(1, Math.min(10, Math.round(config.workers || 3))),
      zkMode: ['off', 'always', 'risk'].includes(config.zkMode) ? config.zkMode : 'off',
      stakePerWorker: Math.max(0, Math.min(1, config.stakePerWorker || 0.0001)),
      explanation: typeof config.explanation === 'string' ? config.explanation : 'Settings configured based on your requirements.',
    };

    return NextResponse.json({
      config: result,
      model,
      provider: providerAddress,
      source: '0g-compute',
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[0G Config Assist] Error:', message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
```

**Step 2: Type-check**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No new type errors.

**Step 3: Commit**

```bash
git add helix/dashboard/src/app/api/0g-compute/config-assist/route.ts
git commit -m "feat: add 0G Compute config-assist API route for AI training configuration"
```

---

### Task 3: Create the AIConfigModal component

The modal that handles the full flow: greeting → user input → payment → inference → result → apply.

**Files:**
- Create: `helix/dashboard/src/components/train/AIConfigModal.tsx`

**Step 1: Create the modal component**

Create `helix/dashboard/src/components/train/AIConfigModal.tsx`:

```tsx
'use client';

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Loader2, CheckCircle, ArrowRight, Zap, ExternalLink } from 'lucide-react';
import { useSendTransaction, useWaitForTransactionReceipt, useAccount } from 'wagmi';
import { parseEther } from 'viem';
import { Modal } from '@/components/ui/Modal';
import { cn } from '@/lib/utils';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

/** Address that receives the small inference fee (service wallet) */
const INFERENCE_FEE_ADDRESS = (process.env.NEXT_PUBLIC_0G_FEE_ADDRESS || '0x0000000000000000000000000000000000000000') as `0x${string}`;
const INFERENCE_FEE_AMOUNT = '0.0001'; // ADI

export interface AIConfigResult {
  hiddenSize: number;
  numSteps: number;
  learningRate: number;
  checkpointFreq: number;
  workers: number;
  zkMode: 'off' | 'always' | 'risk';
  stakePerWorker: number;
  explanation: string;
}

interface AIConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  onApply: (config: AIConfigResult) => void;
  walletConnected: boolean;
}

type ModalStep = 'input' | 'paying' | 'inferring' | 'result' | 'error';

export function AIConfigModal({ isOpen, onClose, onApply, walletConnected }: AIConfigModalProps) {
  const [step, setStep] = useState<ModalStep>('input');
  const [message, setMessage] = useState('');
  const [result, setResult] = useState<AIConfigResult | null>(null);
  const [modelName, setModelName] = useState('');
  const [error, setError] = useState('');

  const { address } = useAccount();
  const {
    sendTransaction,
    data: txHash,
    isPending: isSendingTx,
    reset: resetTx,
  } = useSendTransaction();
  const { isLoading: isConfirmingTx, isSuccess: txConfirmed } = useWaitForTransactionReceipt({ hash: txHash });

  const resetState = useCallback(() => {
    setStep('input');
    setMessage('');
    setResult(null);
    setModelName('');
    setError('');
    resetTx();
  }, [resetTx]);

  const handleClose = () => {
    resetState();
    onClose();
  };

  const callInference = useCallback(async (userMessage: string) => {
    setStep('inferring');
    try {
      const res = await fetch(`${API_BASE}/api/0g-compute/config-assist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: 'Request failed' }));
        throw new Error(err.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data.config);
      setModelName(data.model || '0G Compute');
      setStep('result');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inference failed');
      setStep('error');
    }
  }, []);

  const handleSubmit = async () => {
    if (!message.trim() || !walletConnected) return;

    // If fee address is zero address (not configured), skip payment
    if (INFERENCE_FEE_ADDRESS === '0x0000000000000000000000000000000000000000') {
      await callInference(message);
      return;
    }

    setStep('paying');
    try {
      sendTransaction({
        to: INFERENCE_FEE_ADDRESS,
        value: parseEther(INFERENCE_FEE_AMOUNT),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Payment failed');
      setStep('error');
    }
  };

  // When tx confirms, trigger inference
  const hasCalledInference = useState(false);
  if (txConfirmed && step === 'paying' && !hasCalledInference[0]) {
    hasCalledInference[1](true);
    callInference(message);
  }

  const handleApply = () => {
    if (result) {
      onApply(result);
      handleClose();
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} className="max-w-md">
      <div className="space-y-5">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 border border-purple-500/20 flex items-center justify-center">
            <Sparkles size={20} className="text-purple-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">0G AI Assistant</h2>
            <p className="text-xs text-helix-dim">Powered by 0G Compute Network</p>
          </div>
        </div>

        <AnimatePresence mode="wait">
          {/* INPUT step */}
          {step === 'input' && (
            <motion.div
              key="input"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="space-y-4"
            >
              <p className="text-sm text-helix-text2">
                Hey! Tell me about your training goals and I&apos;ll configure everything for you.
              </p>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="e.g. I want to train quickly just to test the system, don't need high accuracy..."
                rows={3}
                className="w-full px-4 py-3 bg-helix-surface border border-helix-border rounded-xl text-sm text-white placeholder:text-helix-dim resize-none focus:outline-none focus:border-helix-border2 transition-colors"
              />
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs text-helix-dim">
                  <Zap size={12} />
                  <span>Costs {INFERENCE_FEE_AMOUNT} ADI</span>
                </div>
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={!message.trim() || !walletConnected}
                  className={cn(
                    'flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all',
                    message.trim() && walletConnected
                      ? 'bg-white text-black hover:bg-zinc-200'
                      : 'bg-helix-border text-helix-muted cursor-not-allowed',
                  )}
                >
                  Configure for me
                  <ArrowRight size={14} />
                </button>
              </div>
            </motion.div>
          )}

          {/* PAYING step */}
          {step === 'paying' && (
            <motion.div
              key="paying"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="flex flex-col items-center gap-4 py-6"
            >
              <Loader2 size={32} className="animate-spin text-purple-400" />
              <div className="text-center space-y-1">
                <p className="text-sm font-medium text-white">
                  {isSendingTx ? 'Confirm in your wallet...' : 'Confirming payment...'}
                </p>
                <p className="text-xs text-helix-dim">
                  Sending {INFERENCE_FEE_AMOUNT} ADI for 0G inference
                </p>
                {txHash && (
                  <p className="text-xs text-helix-dim font-mono">
                    tx: {txHash.slice(0, 10)}...{txHash.slice(-8)}
                  </p>
                )}
              </div>
            </motion.div>
          )}

          {/* INFERRING step */}
          {step === 'inferring' && (
            <motion.div
              key="inferring"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="flex flex-col items-center gap-4 py-6"
            >
              <div className="relative">
                <Loader2 size={32} className="animate-spin text-purple-400" />
                <Sparkles size={14} className="absolute -top-1 -right-1 text-purple-300 animate-pulse" />
              </div>
              <div className="text-center space-y-1">
                <p className="text-sm font-medium text-white">Thinking via 0G Compute...</p>
                <p className="text-xs text-helix-dim">Decentralized AI inference in progress</p>
              </div>
            </motion.div>
          )}

          {/* RESULT step */}
          {step === 'result' && result && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="space-y-4"
            >
              {/* Explanation */}
              <div className="px-4 py-3 rounded-xl bg-purple-500/[0.06] border border-purple-500/15">
                <p className="text-sm text-purple-200">{result.explanation}</p>
              </div>

              {/* Settings grid */}
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: 'Hidden Size', value: result.hiddenSize },
                  { label: 'Steps', value: result.numSteps },
                  { label: 'Learning Rate', value: result.learningRate },
                  { label: 'Checkpoint Freq', value: result.checkpointFreq },
                  { label: 'Workers', value: result.workers },
                  { label: 'ZK Mode', value: result.zkMode },
                  { label: 'Stake/Worker', value: `${result.stakePerWorker} ADI` },
                ].map((item) => (
                  <div
                    key={item.label}
                    className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03] border border-white/[0.06]"
                  >
                    <span className="text-xs text-helix-dim">{item.label}</span>
                    <span className="text-xs font-medium text-white">{item.value}</span>
                  </div>
                ))}
              </div>

              {/* Model attribution */}
              <p className="text-[11px] text-helix-dim text-center">
                Generated by {modelName} on 0G Compute Network
              </p>

              {/* Apply button */}
              <button
                type="button"
                onClick={handleApply}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-white text-black text-sm font-semibold hover:bg-zinc-200 transition-colors"
              >
                <CheckCircle size={16} />
                Apply Settings
              </button>
            </motion.div>
          )}

          {/* ERROR step */}
          {step === 'error' && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="space-y-4"
            >
              <div className="px-4 py-3 rounded-xl bg-red-500/[0.06] border border-red-500/15">
                <p className="text-sm text-red-300">{error}</p>
              </div>
              <button
                type="button"
                onClick={resetState}
                className="w-full py-2.5 rounded-xl bg-white/[0.06] text-sm text-helix-text2 hover:text-white hover:bg-white/[0.1] transition-colors"
              >
                Try Again
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </Modal>
  );
}
```

**Step 2: Type-check**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No new type errors.

**Step 3: Commit**

```bash
git add helix/dashboard/src/components/train/AIConfigModal.tsx
git commit -m "feat: add AIConfigModal component for 0G AI config assistant"
```

---

### Task 4: Wire the modal into the training page

Add the toggle, state, and config callbacks to `ConfigForm` and `TrainPageInner`.

**Files:**
- Modify: `helix/dashboard/src/app/train/page.tsx`

**Step 1: Add import and state**

At the top of `train/page.tsx`, add to imports:

```typescript
import { AIConfigModal, type AIConfigResult } from '@/components/train/AIConfigModal';
```

Inside the `ConfigForm` function (around line 719, after existing state declarations), add:

```typescript
const [showAIAssist, setShowAIAssist] = useState(false);
```

**Step 2: Add the AI config apply handler**

Inside `ConfigForm`, after `handleVersionChange` (around line 780), add:

```typescript
const handleAIConfigApply = (config: AIConfigResult) => {
  setHiddenSize(config.hiddenSize);
  setNumSteps(config.numSteps);
  setLearningRate(config.learningRate);
  setCheckpointFreq(config.checkpointFreq);
  setMinWorkers(config.workers);
  setZkMode(config.zkMode);
  setStakePerWorkerEth(config.stakePerWorker);
};
```

**Step 3: Add the toggle UI in the right column**

In the ConfigForm JSX, at the start of the right column (find the comment `{/* ── RIGHT COLUMN ─────`}, around line 1109), add a new block **before** the Cost Estimator card:

```tsx
{/* AI Config Assistant */}
<div className="rounded-2xl bg-gradient-to-br from-purple-500/[0.04] to-blue-500/[0.04] border border-purple-500/10 overflow-hidden">
  <div className="flex items-center justify-between px-5 py-4">
    <div className="flex items-center gap-3">
      <Sparkles size={16} className="text-purple-400" />
      <div>
        <p className="text-base text-white">AI Config Assistant</p>
        <p className="text-xs text-helix-dim mt-0.5">Powered by 0G Compute</p>
      </div>
    </div>
    <Toggle on={showAIAssist} onToggle={() => setShowAIAssist(!showAIAssist)} />
  </div>
</div>
```

**Step 4: Add the modal render**

At the end of the ConfigForm return JSX (just before the closing `</div>` of the top-level space-y-6 div, around line 1321), add:

```tsx
<AIConfigModal
  isOpen={showAIAssist}
  onClose={() => setShowAIAssist(false)}
  onApply={handleAIConfigApply}
  walletConnected={walletConnected}
/>
```

**Step 5: Type-check and verify**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No type errors.

**Step 6: Commit**

```bash
git add helix/dashboard/src/app/train/page.tsx
git commit -m "feat: wire AI Config Assistant toggle and modal into training page"
```

---

### Task 5: Add env var for fee address and update .env.example

**Files:**
- Modify: `helix/dashboard/.env.example`

**Step 1: Add the 0G Compute config section to .env.example**

After the existing 0G Storage section (around line 91), add:

```env
# =============================================================================
# 0G Compute (AI inference via decentralized GPUs)
# =============================================================================

# Private key for server-side 0G Compute broker (same as ZG_PRIVATE_KEY or separate)
# ZG_COMPUTE_PRIVATE_KEY=

# Address that receives the small inference fee from users (your service wallet)
# NEXT_PUBLIC_0G_FEE_ADDRESS=0xYourServiceWalletAddress

# 0G Compute provider address (default: Qwen 2.5 7B on testnet)
# ZG_COMPUTE_PROVIDER=0xa48f01287233509FD694a22Bf840225062E67836
```

**Step 2: Commit**

```bash
git add helix/dashboard/.env.example
git commit -m "docs: add 0G Compute env vars to .env.example"
```

---

### Task 6: Fix the payment-to-inference race condition

The `AIConfigModal` uses a `useState` boolean inside the render body to track whether inference has been called after payment confirms. This is fragile. Refactor to use a `useEffect`.

**Files:**
- Modify: `helix/dashboard/src/components/train/AIConfigModal.tsx`

**Step 1: Replace the render-body check with a useEffect**

Remove the `hasCalledInference` useState + conditional block. Replace with:

```typescript
// When tx confirms, trigger inference
const [inferenceTriggered, setInferenceTriggered] = useState(false);

useEffect(() => {
  if (txConfirmed && step === 'paying' && !inferenceTriggered) {
    setInferenceTriggered(true);
    callInference(message);
  }
}, [txConfirmed, step, inferenceTriggered, callInference, message]);
```

Also add `inferenceTriggered` to the `resetState` callback:

```typescript
const resetState = useCallback(() => {
  setStep('input');
  setMessage('');
  setResult(null);
  setModelName('');
  setError('');
  setInferenceTriggered(false);
  resetTx();
}, [resetTx]);
```

**Step 2: Type-check**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No type errors.

**Step 3: Commit**

```bash
git add helix/dashboard/src/components/train/AIConfigModal.tsx
git commit -m "fix: use useEffect for payment-to-inference transition in AIConfigModal"
```

---

### Task 7: Visual polish and build verification

**Files:**
- Possibly tweak: `helix/dashboard/src/components/train/AIConfigModal.tsx`
- Possibly tweak: `helix/dashboard/src/app/train/page.tsx`

**Step 1: Run full build**

Run: `cd helix/dashboard && npm run build`
Expected: Build succeeds with no errors.

**Step 2: Run lint**

Run: `cd helix/dashboard && npm run lint`
Expected: No new lint errors (fix any that appear).

**Step 3: Manual verification checklist**

- [ ] Toggle appears in right column of training config
- [ ] Toggle opens the modal
- [ ] Modal shows greeting and text area
- [ ] Submit triggers MetaMask payment
- [ ] After payment, shows "Thinking via 0G Compute..."
- [ ] Result shows settings grid and explanation
- [ ] "Apply Settings" fills config form and closes modal
- [ ] Escape / X closes modal and resets state
- [ ] Error state shows "Try Again" button

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: 0G AI Config Assistant — toggle, modal, and 0G Compute inference for auto-config"
```
