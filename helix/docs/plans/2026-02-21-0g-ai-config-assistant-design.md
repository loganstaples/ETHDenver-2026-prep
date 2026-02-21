# 0G AI Config Assistant — Design

## Overview

A toggle on the training page that opens a modal powered by 0G Compute inference. The user describes their training goals in natural language, pays a small OG fee (one MetaMask approval), and an AI model on 0G's decentralized GPU network analyzes the request and auto-fills the training config settings.

## User Flow

1. User sees "AI Config Assistant" toggle in the right column of the training config (above ZK Proofs section)
2. Toggle opens a medium modal with:
   - Friendly greeting: "Hey! Tell me about your training goals and I'll configure everything."
   - Text area for the user's description
   - "Configure for me" button
3. User types something like "I want to train quickly, just testing" or "Maximum accuracy, I have time and budget"
4. Clicks "Configure for me" → MetaMask pops up for a small OG payment (one approval)
5. Modal shows loading state: "Thinking via 0G Compute..." with model name and tx hash
6. AI response comes back with suggested settings + brief explanation
7. User clicks "Apply Settings" → config form auto-fills, modal closes

## Architecture

### Existing Infrastructure (reused)

- **`/api/0g-compute/route.ts`** — Already has a singleton broker with `ensureSetup()`, provider acknowledgment, fund management, and OpenAI-compatible inference. Currently used for digit classification.
- **`@0glabs/0g-serving-broker`** and **`openai`** — Already in package.json.
- **Server-side wallet** — `ZG_COMPUTE_PRIVATE_KEY` env var, pre-funded on 0G testnet.

### New: API Endpoint

Add a new route `/api/0g-compute/config-assist/route.ts` that:
1. Accepts `{ message: string, txHash?: string }` — user's description + optional payment tx hash
2. Reuses the singleton broker from the parent route (extract to shared module)
3. Sends a system prompt + user message to 0G Compute
4. Parses the JSON config response
5. Returns `{ config: TrainingConfig, explanation: string, model: string, cost: string }`

### New: Frontend Components

**`AIConfigModal.tsx`** — Modal component:
- Uses existing `Modal` component as base
- States: `idle` → `paying` → `inferring` → `result` → `applied`
- Payment: wagmi `useSendTransaction` to send OG to a service address (one MetaMask tx)
- Inference: calls `/api/0g-compute/config-assist` with message + tx hash
- Result display: shows suggested settings + explanation + "Apply" button

### Modified: `train/page.tsx`

- Add `aiAssistEnabled` state and toggle in right column
- Add `AIConfigModal` with callbacks to set each config value
- Config setters passed down: `setHiddenSize`, `setNumSteps`, `setLearningRate`, etc.

## System Prompt

```
You are a helpful ML training configuration assistant for HELIX, a decentralized verifiable ML training protocol. The user is configuring an MNIST neural network (784 → hidden → 10) trained via MPC (multi-party computation).

Available settings you can configure:
- hiddenSize: Hidden layer neurons (16-512, step 16). Higher = more capacity but slower.
- numSteps: Training steps (10-10000). More = better accuracy but longer.
- learningRate: Learning rate (0.00001-1). Controls convergence speed.
- checkpointFreq: How often to checkpoint (10-500). Lower = more verification overhead.
- workers: Number of MPC workers (1-10). More = better security but more communication.
- zkMode: "off", "always", or "risk". ZK proofs for verification. "off" for speed, "always" for max security.
- stakePerWorker: Stake per worker in ADI tokens (0+). Economic security.

Based on the user's description, suggest optimal settings. Respond ONLY with a JSON object:
{
  "hiddenSize": <number>,
  "numSteps": <number>,
  "learningRate": <number>,
  "checkpointFreq": <number>,
  "workers": <number>,
  "zkMode": "<off|always|risk>",
  "stakePerWorker": <number>,
  "explanation": "<1-2 sentence explanation of your choices>"
}
```

## Payment Flow

1. Frontend sends a small OG payment (0.0001 OG) to the service wallet address via wagmi
2. One MetaMask approval
3. Frontend shows the tx hash while waiting for confirmation
4. After tx confirms, calls the API with the message and tx hash
5. API verifies tx (optional for hackathon), runs inference, returns config

The payment amount is tiny (testnet tokens) but visually demonstrates the decentralized payment → AI inference flow.

## New Files

- `helix/dashboard/src/app/api/0g-compute/config-assist/route.ts`
- `helix/dashboard/src/components/train/AIConfigModal.tsx`
- `helix/dashboard/src/lib/0g-broker.ts` (shared broker singleton, extracted from existing route)

## Modified Files

- `helix/dashboard/src/app/train/page.tsx` — Add toggle + modal + config callbacks
- `helix/dashboard/src/app/api/0g-compute/route.ts` — Import broker from shared module
