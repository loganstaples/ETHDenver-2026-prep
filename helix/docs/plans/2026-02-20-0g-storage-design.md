# 0G Storage Integration Design

**Date:** 2026-02-20
**Status:** Approved
**Approach:** Wallet-derived AES keys with re-upload on NFT transfer

## Overview

Integrate 0G (Zero Gravity) testnet decentralized storage as the primary storage layer for model weights. Models are ERC-721 NFTs (HelixModelStore). Only the model owner can download, train, or view weights. Public models allow non-owner inference without exposing weights. On NFT sale/transfer, weights are re-encrypted and re-uploaded for the new owner.

## Requirements

1. Model weights stored encrypted on 0G — only owner can decrypt
2. Only owner can download weights or train the model further
3. Public models allow anyone to run inference (without seeing weights)
4. Owner pays for 0G storage themselves (from their wallet)
5. NFT transfer requires re-encryption + re-upload for new owner
6. System set up completely but no live 0G transactions yet

## What Already Exists

| Component | Status | Notes |
|-----------|--------|-------|
| `HelixModelStore.sol` | Complete | ERC-721 with rootHash, weightsStored, isPublic, marketplace |
| `model-encryption.ts` | Complete | AES-GCM encrypt/decrypt with wallet-derived keys |
| `store-on-0g/route.ts` | Working | 0G upload via `@0glabs/0g-ts-sdk`, supports encrypted payloads |
| `fetch-from-0g/route.ts` | Working | 0G download via indexer |
| `buyModel()` | Exists | Immediate transfer — needs escrow for key handoff |

## Architecture

### Encryption Scheme

Wallet-derived AES-GCM keys (existing `model-encryption.ts`):

```
sign("helix-model-key-{tokenId}") → signature
SHA-256(signature) → 32-byte AES-GCM key
encrypt(key, weightsJSON) → [12-byte IV][ciphertext]
```

Only the owner's wallet can reproduce the signature, so only the owner can derive the decryption key. The ciphertext is safe to store publicly on 0G.

### Layer 1: Smart Contract (`HelixModelStore.sol`)

**Changes needed:**

1. **Two-step escrow transfer** replacing immediate `buyModel()`:
   - `buyModel(tokenId)` — escrows payment, records buyer, marks `transferPending`
   - `completeTransfer(tokenId, newRootHash)` — seller provides re-encrypted rootHash, NFT transfers, payment released
   - `cancelSale(tokenId)` — either party cancels, buyer refunded
   - `transferDeadline` — if seller doesn't complete within 24h, buyer can cancel

2. **New state:**
   ```solidity
   struct PendingTransfer {
       address buyer;
       uint256 payment;
       uint40 deadline;
   }
   mapping(uint256 => PendingTransfer) public pendingTransfers;
   ```

3. **Existing state (no changes):**
   - `Model.isPublic` — controls inference access
   - `Version.rootHash` — 0G storage pointer
   - `Version.weightsStored` — whether weights are on 0G
   - `onlyModelOwner` — for addVersion, setPublic, etc.
   - `hasModelAccess()` — returns true for owner or public models

### Layer 2: API Routes (Next.js)

**Existing (no changes needed):**
- `POST /api/store-on-0g` — upload encrypted blob to 0G
- `POST /api/fetch-from-0g` — download blob from 0G by rootHash (ciphertext is safe to serve publicly)

**New endpoints:**
- `POST /api/models/:tokenId/enable-inference` — Owner sends decrypted weights. Backend caches in-memory keyed by `(tokenId, versionIndex)`. Returns `session_id`.
- `DELETE /api/models/:tokenId/disable-inference` — Owner clears cache, stops public inference.
- `GET /api/models/:tokenId/inference-ready?version=N` — Returns `{ ready, session_id? }`. (Partially exists.)
- `POST /api/models/:tokenId/infer` — Runs inference on cached weights. Checks model is public OR caller is owner. Returns prediction only, never weights.
- `POST /api/models/:tokenId/transfer-relay` — Seller sends decrypted weights. Backend holds temporarily (one-time, in-memory). Buyer fetches once, re-encrypts, re-uploads to 0G. Cleared after fetch or timeout.

**Training enforcement:**
- `POST /api/training/start` — Verify caller is model owner (wallet signature check against on-chain `ownerOf(tokenId)`) before allowing training.

### Layer 3: Dashboard Flows

#### 3a. Owner Creates Model + Uploads Weights
1. Fill out model form (name, slug, architecture)
2. Upload initial weights (JSON or file)
3. Toggle "Store on 0G"
4. If 0G: sign message → derive AES key → encrypt client-side → upload to 0G (owner wallet pays gas) → `createModel()` + `addVersion(rootHash, weightsStored: true)`
5. If no 0G: weights stay on backend only

#### 3b. Owner Downloads Weights
1. Read `rootHash` from contract via `getVersion(tokenId, index)`
2. Fetch encrypted blob via `/api/fetch-from-0g`
3. Sign message → derive AES key → decrypt client-side
4. Display or download plaintext weights

#### 3c. Owner Trains Further
1. Select model + version to continue from
2. Fetch + decrypt weights (same as 3b)
3. Upload decrypted weights to backend for MPC training
4. Training runs (existing MPC flow)
5. On completion: encrypt new weights → upload to 0G → `addVersion()` with new rootHash

#### 3d. Owner Enables Public Inference
1. Click "Enable Public Inference" on model page
2. Sign → decrypt weights client-side
3. Send decrypted weights to `POST /api/models/:tokenId/enable-inference`
4. Backend caches in memory (never persisted, never exposed)
5. Owner can revoke via "Disable Public Inference"

#### 3e. Non-Owner Runs Inference (Public Model)
1. See public model → click "Run Inference"
2. Check `GET /api/models/:tokenId/inference-ready`
3. If ready → submit input → `POST /api/models/:tokenId/infer` → get prediction
4. If not ready → "Model owner hasn't enabled inference yet"
5. Non-owner never sees weights, never downloads from 0G

#### 3f. NFT Transfer (Sale)
1. Seller: `setForSale()` + `setSalePrice()`
2. Buyer: `buyModel()` → payment escrowed, `transferPending` set
3. Seller sees "Pending Transfer" notification
4. Seller clicks "Complete Transfer":
   - Decrypts weights client-side
   - Sends to backend via transfer-relay endpoint
   - Buyer fetches from relay (one-time), re-encrypts with their wallet-derived key
   - Buyer uploads to 0G → gets new rootHash
   - Seller (or buyer) calls `completeTransfer(tokenId, newRootHash)`
   - NFT transfers, payment released
5. Timeout (24h): buyer can `cancelSale()` for refund

### 0G Storage Payment

Current `store-on-0g/route.ts` uses a server-side `ZG_PRIVATE_KEY` to pay. Per requirements, the **owner should pay from their own wallet**. Two options:

- **Option A (recommended for hackathon):** Keep server-side key for now. The server pays 0G gas. Simple, avoids requiring users to have 0G testnet tokens. Can switch to user-pays later.
- **Option B (production):** Move 0G upload to client-side using `@0glabs/0g-ts-sdk` in the browser with the owner's wallet as signer. Owner needs 0G testnet tokens.

For now: keep Option A (server pays), document that production should use Option B.

## Access Control Summary

| Action | Owner | Non-Owner (Public) | Non-Owner (Private) |
|--------|-------|---------------------|---------------------|
| Download weights from 0G | Yes (decrypt) | No (can't decrypt) | No |
| View plaintext weights | Yes | No | No |
| Train model | Yes | No | No |
| Run inference | Yes (direct) | Yes (via cache) | No |
| Add version | Yes | No | No |
| Set public/private | Yes | No | No |
| Buy model | N/A | Yes (if for sale) | Yes (if for sale) |

## Security Properties

1. **Encryption at rest:** Weights encrypted with AES-256-GCM before upload to 0G
2. **Key derivation:** Only the owner's wallet can sign the message that derives the decryption key
3. **Ciphertext safety:** Encrypted blobs on 0G are useless without the owner's wallet
4. **Inference isolation:** Backend holds decrypted weights in memory only, never returns them via API
5. **Transfer safety:** Escrow prevents payment loss if seller doesn't re-encrypt; timeout protects buyer
6. **Training gating:** On-chain ownership check before training session start

## Files to Create/Modify

### Modify
- `helix/contracts/src/core/HelixModelStore.sol` — Add PendingTransfer, escrow buyModel, completeTransfer, cancelSale
- `helix/contracts/test/HelixModelStore.t.sol` — Tests for new transfer flow
- `helix/dashboard/src/app/api/store-on-0g/route.ts` — Accept encrypted flag, owner-pays documentation
- `helix/dashboard/src/app/inference/page.tsx` — Wire up inference-ready check and public inference flow
- `helix/dashboard/src/app/train/page.tsx` — Wire up 0G encrypt+upload after training
- `helix/dashboard/src/app/models/[id]/page.tsx` — Add enable/disable inference, transfer UI
- `helix/dashboard/src/app/my-models/page.tsx` — Show pending transfers
- `helix/dashboard/src/lib/model-encryption.ts` — No changes needed (already complete)

### Create
- `helix/dashboard/src/app/api/models/[tokenId]/enable-inference/route.ts` — Cache decrypted weights
- `helix/dashboard/src/app/api/models/[tokenId]/disable-inference/route.ts` — Clear cache
- `helix/dashboard/src/app/api/models/[tokenId]/inference-ready/route.ts` — Check cache status
- `helix/dashboard/src/app/api/models/[tokenId]/infer/route.ts` — Run inference on cached weights
- `helix/dashboard/src/app/api/models/[tokenId]/transfer-relay/route.ts` — Temporary weight relay for transfers
- `helix/dashboard/src/lib/0g-client.ts` — Shared 0G SDK helpers (indexer init, upload, download)

## Deployment Target

- ADI Chain (primary) — EVM-compatible, contracts deploy with Foundry
- 0G Galileo Testnet — for storage layer
- Both testnets, no mainnet transactions
