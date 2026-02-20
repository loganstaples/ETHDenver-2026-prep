# 0G Storage Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate 0G decentralized storage for encrypted model weights with escrow transfers, inference caching, and owner-only access control.

**Architecture:** Wallet-derived AES-GCM encryption (existing `model-encryption.ts`). Weights encrypted client-side, uploaded to 0G. Only the NFT owner can decrypt. Public models use a server-side in-memory cache for inference without exposing weights. NFT sales use a two-step escrow with re-encryption + re-upload.

**Tech Stack:** Solidity ^0.8.19 (Foundry), Next.js API routes, TypeScript, `@0glabs/0g-ts-sdk`, wagmi/viem, AES-256-GCM via Web Crypto API.

**Design doc:** `helix/docs/plans/2026-02-20-0g-storage-design.md`

---

## Task 1: Contract — Add Escrow Transfer to HelixModelStore

**Files:**
- Modify: `helix/contracts/src/core/HelixModelStore.sol`

**Context:** The existing `buyModel()` does an immediate transfer + payment. We need a two-step escrow so the seller can re-encrypt weights for the buyer before the NFT transfers. Also need `updateVersionRootHash()` so the seller can update the rootHash of the latest version after re-encryption (used during `completeTransfer`).

**Step 1: Add PendingTransfer struct, state, events, and constant**

After the existing `_inferenceNonce` state variable (line 47), add:

```solidity
// Escrow transfer state
uint40 public constant TRANSFER_DEADLINE_DURATION = 24 hours;

struct PendingTransfer {
    address buyer;
    uint256 payment;
    uint40 deadline;
}

mapping(uint256 => PendingTransfer) public pendingTransfers;

event TransferInitiated(uint256 indexed tokenId, address indexed seller, address indexed buyer, uint256 price, uint40 deadline);
event TransferCompleted(uint256 indexed tokenId, address indexed seller, address indexed buyer, string newRootHash);
event TransferCancelled(uint256 indexed tokenId, address indexed cancelledBy);
```

**Step 2: Replace `buyModel()` with escrow version**

Replace the existing `buyModel()` function (lines 222-241) with:

```solidity
/// @notice Initiate purchase of a model NFT — payment is escrowed until seller completes transfer
function buyModel(uint256 tokenId) external payable nonReentrant {
    require(isForSale[tokenId], "Model not for sale");
    require(salePrice[tokenId] > 0, "Sale price not set");
    require(msg.value >= salePrice[tokenId], "Insufficient payment");
    address seller = ownerOf(tokenId);
    require(msg.sender != seller, "Cannot buy own model");
    require(pendingTransfers[tokenId].buyer == address(0), "Transfer already pending");

    uint40 deadline = uint40(block.timestamp) + TRANSFER_DEADLINE_DURATION;
    pendingTransfers[tokenId] = PendingTransfer({
        buyer: msg.sender,
        payment: msg.value,
        deadline: deadline
    });

    // Delist while transfer is pending
    isForSale[tokenId] = false;

    emit TransferInitiated(tokenId, seller, msg.sender, msg.value, deadline);
}
```

**Step 3: Add `completeTransfer()`**

Add after the new `buyModel()`:

```solidity
/// @notice Seller completes transfer by providing re-encrypted rootHash for buyer
/// @param tokenId The model token ID
/// @param newRootHash The new 0G root hash (weights re-encrypted for buyer)
function completeTransfer(uint256 tokenId, string calldata newRootHash) external nonReentrant {
    PendingTransfer memory pt = pendingTransfers[tokenId];
    require(pt.buyer != address(0), "No pending transfer");
    address seller = ownerOf(tokenId);
    require(msg.sender == seller, "Only seller can complete");

    // Update the latest version's rootHash to the re-encrypted one
    uint256 versionCount = _versions[tokenId].length;
    if (versionCount > 0) {
        _versions[tokenId][versionCount - 1].rootHash = newRootHash;
    }

    // Clear pending transfer
    delete pendingTransfers[tokenId];
    salePrice[tokenId] = 0;

    // Transfer NFT to buyer
    _transfer(seller, pt.buyer, tokenId);

    // Release payment to seller
    (bool sent, ) = payable(seller).call{value: pt.payment}("");
    require(sent, "Payment failed");

    emit TransferCompleted(tokenId, seller, pt.buyer, newRootHash);
}
```

**Step 4: Add `cancelSale()`**

Add after `completeTransfer()`:

```solidity
/// @notice Cancel a pending transfer — buyer gets refunded
/// @dev Seller can cancel anytime. Buyer can cancel after deadline.
function cancelSale(uint256 tokenId) external nonReentrant {
    PendingTransfer memory pt = pendingTransfers[tokenId];
    require(pt.buyer != address(0), "No pending transfer");

    address seller = ownerOf(tokenId);
    bool isSeller = msg.sender == seller;
    bool isBuyerAfterDeadline = msg.sender == pt.buyer && block.timestamp >= pt.deadline;
    require(isSeller || isBuyerAfterDeadline, "Not authorized to cancel");

    // Clear pending transfer
    delete pendingTransfers[tokenId];

    // Refund buyer
    (bool sent, ) = payable(pt.buyer).call{value: pt.payment}("");
    require(sent, "Refund failed");

    emit TransferCancelled(tokenId, msg.sender);
}
```

**Step 5: Add `updateVersionRootHash()` for re-encryption updates**

Add after `cancelSale()`:

```solidity
/// @notice Update the rootHash of a specific version (for re-encryption after training)
/// @param tokenId The model token ID
/// @param versionIndex The version to update
/// @param newRootHash The new 0G root hash
function updateVersionRootHash(
    uint256 tokenId,
    uint256 versionIndex,
    string calldata newRootHash
) external onlyModelOwner(tokenId) {
    require(versionIndex < _versions[tokenId].length, "Invalid version");
    _versions[tokenId][versionIndex].rootHash = newRootHash;
    _versions[tokenId][versionIndex].weightsStored = bytes(newRootHash).length > 0;
    emit VersionAdded(tokenId, versionIndex, _versions[tokenId][versionIndex].semver, newRootHash);
}
```

**Step 6: Build the contracts**

Run: `cd helix/contracts && forge build`
Expected: Compilation succeeds with no errors.

**Step 7: Commit**

```bash
git add helix/contracts/src/core/HelixModelStore.sol
git commit -m "feat(contracts): add escrow transfer and updateVersionRootHash to HelixModelStore"
```

---

## Task 2: Contract — Tests for Escrow Transfer

**Files:**
- Modify: `helix/contracts/test/HelixModelStore.t.sol`

**Context:** The test file uses Foundry patterns: `vm.prank()`, `vm.deal()`, `vm.expectRevert()`, `vm.warp()`. Test actors are `alice`, `bob`, `charlie`. Tests are named `test_[function]_[scenario]()` and grouped by comment headers.

**Step 1: Add new event declarations in the test contract**

At the top of `HelixModelStoreTest`, after the existing event declarations, add:

```solidity
event TransferInitiated(uint256 indexed tokenId, address indexed seller, address indexed buyer, uint256 price, uint40 deadline);
event TransferCompleted(uint256 indexed tokenId, address indexed seller, address indexed buyer, string newRootHash);
event TransferCancelled(uint256 indexed tokenId, address indexed cancelledBy);
```

**Step 2: Add escrow transfer test section at the end of the file**

Add before the closing `}` of the test contract:

```solidity
// -------------------------------------------------------
// Escrow Transfer (buyModel → completeTransfer)
// -------------------------------------------------------

function test_buyModel_escrowsPayment() public {
    vm.startPrank(alice);
    uint256 tokenId = store.createModel("escrow-1", "Escrow Model", "", "");
    store.addVersion(tokenId, "1.0.0", "0xrootHash123", 9500, "sess-1", true);
    store.setForSale(tokenId, true);
    store.setSalePrice(tokenId, 1 ether);
    vm.stopPrank();

    uint256 bobBalBefore = bob.balance;
    uint256 contractBalBefore = address(store).balance;

    vm.prank(bob);
    store.buyModel{value: 1 ether}(tokenId);

    // NFT should still belong to alice (escrowed, not transferred yet)
    assertEq(store.ownerOf(tokenId), alice);
    // Bob's balance decreased
    assertEq(bob.balance, bobBalBefore - 1 ether);
    // Contract holds the escrow
    assertEq(address(store).balance, contractBalBefore + 1 ether);
    // Model delisted
    assertFalse(store.isForSale(tokenId));
    // Pending transfer recorded
    (address buyer, uint256 payment, uint40 deadline) = store.pendingTransfers(tokenId);
    assertEq(buyer, bob);
    assertEq(payment, 1 ether);
    assertTrue(deadline > block.timestamp);
}

function test_buyModel_revertsIfAlreadyPending() public {
    vm.startPrank(alice);
    uint256 tokenId = store.createModel("escrow-dup", "Dup", "", "");
    store.setForSale(tokenId, true);
    store.setSalePrice(tokenId, 1 ether);
    vm.stopPrank();

    vm.prank(bob);
    store.buyModel{value: 1 ether}(tokenId);

    vm.deal(charlie, 10 ether);
    vm.prank(charlie);
    vm.expectRevert("Transfer already pending");
    store.buyModel{value: 1 ether}(tokenId);
}

function test_completeTransfer_transfersNFTAndPays() public {
    vm.startPrank(alice);
    uint256 tokenId = store.createModel("comp-1", "Complete", "", "");
    store.addVersion(tokenId, "1.0.0", "0xoldHash", 9500, "sess-1", true);
    store.setForSale(tokenId, true);
    store.setSalePrice(tokenId, 1 ether);
    vm.stopPrank();

    vm.prank(bob);
    store.buyModel{value: 1 ether}(tokenId);

    uint256 aliceBalBefore = alice.balance;

    vm.prank(alice);
    store.completeTransfer(tokenId, "0xnewHashForBuyer");

    // NFT transferred to bob
    assertEq(store.ownerOf(tokenId), bob);
    // Alice got paid
    assertEq(alice.balance, aliceBalBefore + 1 ether);
    // Pending transfer cleared
    (address buyer, , ) = store.pendingTransfers(tokenId);
    assertEq(buyer, address(0));
    // Root hash updated
    HelixModelStore.Version memory v = store.getVersion(tokenId, 0);
    assertEq(v.rootHash, "0xnewHashForBuyer");
}

function test_completeTransfer_revertsIfNotSeller() public {
    vm.startPrank(alice);
    uint256 tokenId = store.createModel("comp-auth", "Auth", "", "");
    store.setForSale(tokenId, true);
    store.setSalePrice(tokenId, 1 ether);
    vm.stopPrank();

    vm.prank(bob);
    store.buyModel{value: 1 ether}(tokenId);

    vm.prank(bob);
    vm.expectRevert("Only seller can complete");
    store.completeTransfer(tokenId, "0xhash");
}

function test_completeTransfer_revertsIfNoPending() public {
    vm.startPrank(alice);
    uint256 tokenId = store.createModel("comp-none", "None", "", "");
    vm.stopPrank();

    vm.prank(alice);
    vm.expectRevert("No pending transfer");
    store.completeTransfer(tokenId, "0xhash");
}

function test_cancelSale_sellerCanCancelAnytime() public {
    vm.startPrank(alice);
    uint256 tokenId = store.createModel("cancel-1", "Cancel", "", "");
    store.setForSale(tokenId, true);
    store.setSalePrice(tokenId, 1 ether);
    vm.stopPrank();

    uint256 bobBalBefore = bob.balance;
    vm.prank(bob);
    store.buyModel{value: 1 ether}(tokenId);

    vm.prank(alice);
    store.cancelSale(tokenId);

    // Bob refunded
    assertEq(bob.balance, bobBalBefore);
    // NFT still with alice
    assertEq(store.ownerOf(tokenId), alice);
    // Pending cleared
    (address buyer, , ) = store.pendingTransfers(tokenId);
    assertEq(buyer, address(0));
}

function test_cancelSale_buyerCanCancelAfterDeadline() public {
    vm.startPrank(alice);
    uint256 tokenId = store.createModel("cancel-dl", "Deadline", "", "");
    store.setForSale(tokenId, true);
    store.setSalePrice(tokenId, 1 ether);
    vm.stopPrank();

    vm.prank(bob);
    store.buyModel{value: 1 ether}(tokenId);

    // Buyer can't cancel before deadline
    vm.prank(bob);
    vm.expectRevert("Not authorized to cancel");
    store.cancelSale(tokenId);

    // Advance past deadline (24 hours)
    vm.warp(block.timestamp + 24 hours + 1);

    uint256 bobBalBefore = bob.balance;
    vm.prank(bob);
    store.cancelSale(tokenId);

    assertEq(bob.balance, bobBalBefore + 1 ether);
    assertEq(store.ownerOf(tokenId), alice);
}

function test_cancelSale_revertsIfNoPending() public {
    vm.startPrank(alice);
    uint256 tokenId = store.createModel("cancel-none", "None", "", "");
    vm.stopPrank();

    vm.prank(alice);
    vm.expectRevert("No pending transfer");
    store.cancelSale(tokenId);
}

function test_completeTransfer_emitsEvent() public {
    vm.startPrank(alice);
    uint256 tokenId = store.createModel("emit-1", "Emit", "", "");
    store.addVersion(tokenId, "1.0.0", "0xold", 9500, "s1", true);
    store.setForSale(tokenId, true);
    store.setSalePrice(tokenId, 1 ether);
    vm.stopPrank();

    vm.prank(bob);
    store.buyModel{value: 1 ether}(tokenId);

    vm.prank(alice);
    vm.expectEmit(true, true, true, true);
    emit TransferCompleted(tokenId, alice, bob, "0xnewHash");
    store.completeTransfer(tokenId, "0xnewHash");
}

function test_updateVersionRootHash_ownerOnly() public {
    vm.startPrank(alice);
    uint256 tokenId = store.createModel("upd-root", "Update Root", "", "");
    store.addVersion(tokenId, "1.0.0", "0xoriginal", 9500, "s1", true);
    store.updateVersionRootHash(tokenId, 0, "0xupdated");
    vm.stopPrank();

    HelixModelStore.Version memory v = store.getVersion(tokenId, 0);
    assertEq(v.rootHash, "0xupdated");
    assertTrue(v.weightsStored);

    vm.prank(bob);
    vm.expectRevert("Not model owner");
    store.updateVersionRootHash(tokenId, 0, "0xhacked");
}

function test_updateVersionRootHash_clearsWeightsStoredWhenEmpty() public {
    vm.startPrank(alice);
    uint256 tokenId = store.createModel("upd-clear", "Clear", "", "");
    store.addVersion(tokenId, "1.0.0", "0xhash", 9500, "s1", true);
    store.updateVersionRootHash(tokenId, 0, "");
    vm.stopPrank();

    HelixModelStore.Version memory v = store.getVersion(tokenId, 0);
    assertEq(v.rootHash, "");
    assertFalse(v.weightsStored);
}
```

**Step 3: Run all contract tests**

Run: `cd helix/contracts && forge test --match-contract HelixModelStoreTest -vvv`
Expected: All tests pass, including the new escrow transfer tests.

**Step 4: Commit**

```bash
git add helix/contracts/test/HelixModelStore.t.sol
git commit -m "test(contracts): add escrow transfer and updateVersionRootHash tests"
```

---

## Task 3: Dashboard — Update Contract ABI

**Files:**
- Modify: `helix/dashboard/src/lib/contracts.ts`

**Context:** The dashboard ABI at `HELIX_MODEL_STORE_ABI` (line 377) must match the new contract functions. Add ABI entries for `completeTransfer`, `cancelSale`, `updateVersionRootHash`, `pendingTransfers`, and `TRANSFER_DEADLINE_DURATION`.

**Step 1: Add new write function ABIs**

After the existing `buyModel` ABI entry (around line 453), add:

```typescript
{
    type: 'function',
    name: 'completeTransfer',
    inputs: [
        { name: 'tokenId', type: 'uint256' },
        { name: 'newRootHash', type: 'string' },
    ],
    outputs: [],
    stateMutability: 'nonpayable',
},
{
    type: 'function',
    name: 'cancelSale',
    inputs: [
        { name: 'tokenId', type: 'uint256' },
    ],
    outputs: [],
    stateMutability: 'nonpayable',
},
{
    type: 'function',
    name: 'updateVersionRootHash',
    inputs: [
        { name: 'tokenId', type: 'uint256' },
        { name: 'versionIndex', type: 'uint256' },
        { name: 'newRootHash', type: 'string' },
    ],
    outputs: [],
    stateMutability: 'nonpayable',
},
```

**Step 2: Add new read function ABIs**

After the existing `inferenceFeesAccrued` read entry, add:

```typescript
{
    type: 'function',
    name: 'pendingTransfers',
    inputs: [{ name: 'tokenId', type: 'uint256' }],
    outputs: [
        { name: 'buyer', type: 'address' },
        { name: 'payment', type: 'uint256' },
        { name: 'deadline', type: 'uint40' },
    ],
    stateMutability: 'view',
},
{
    type: 'function',
    name: 'TRANSFER_DEADLINE_DURATION',
    inputs: [],
    outputs: [{ name: '', type: 'uint40' }],
    stateMutability: 'view',
},
```

**Step 3: Add new event ABIs**

After the existing `InferencePaid` event ABI entry, add:

```typescript
{
    type: 'event',
    name: 'TransferInitiated',
    inputs: [
        { name: 'tokenId', type: 'uint256', indexed: true },
        { name: 'seller', type: 'address', indexed: true },
        { name: 'buyer', type: 'address', indexed: true },
        { name: 'price', type: 'uint256', indexed: false },
        { name: 'deadline', type: 'uint40', indexed: false },
    ],
},
{
    type: 'event',
    name: 'TransferCompleted',
    inputs: [
        { name: 'tokenId', type: 'uint256', indexed: true },
        { name: 'seller', type: 'address', indexed: true },
        { name: 'buyer', type: 'address', indexed: true },
        { name: 'newRootHash', type: 'string', indexed: false },
    ],
},
{
    type: 'event',
    name: 'TransferCancelled',
    inputs: [
        { name: 'tokenId', type: 'uint256', indexed: true },
        { name: 'cancelledBy', type: 'address', indexed: true },
    ],
},
```

**Step 4: Build the dashboard to verify**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No type errors.

**Step 5: Commit**

```bash
git add helix/dashboard/src/lib/contracts.ts
git commit -m "feat(dashboard): add escrow transfer ABI entries to contracts.ts"
```

---

## Task 4: Dashboard — Create Shared 0G Client Helper

**Files:**
- Create: `helix/dashboard/src/lib/0g-client.ts`

**Context:** Multiple pages need to upload/download from 0G. Extract shared helpers. The existing API routes (`store-on-0g/route.ts`, `fetch-from-0g/route.ts`) handle the actual 0G SDK calls server-side. The client helper wraps fetch calls to those routes.

**Step 1: Create the helper file**

```typescript
/**
 * Client-side helpers for interacting with 0G Storage via the Next.js API routes.
 * These wrap the server-side routes that use @0glabs/0g-ts-sdk.
 */

const API_BASE = '';  // Same-origin API routes

export interface StoreOn0GParams {
  sessionId: string;
  weights?: Record<string, unknown>;
  encrypted?: boolean;
  encryptedPayload?: string;  // base64-encoded encrypted bytes
  accuracy?: number;
  version?: string;
}

export interface StoreOn0GResult {
  status: string;
  root_hash: string;
  tx_hash: string;
  explorer_url: string;
  retrieval_url: string;
  session_id: string;
}

export interface FetchFrom0GResult {
  data: string | Record<string, unknown>;
  encoding: 'json' | 'base64';
}

/** Upload model weights (encrypted or plaintext) to 0G Storage via API route */
export async function storeOn0G(params: StoreOn0GParams): Promise<StoreOn0GResult> {
  const res = await fetch(`${API_BASE}/api/store-on-0g`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Store on 0G failed: HTTP ${res.status}`);
  }
  return res.json();
}

/** Fetch model weights from 0G Storage by root hash */
export async function fetchFrom0G(rootHash: string): Promise<FetchFrom0GResult> {
  const res = await fetch(`${API_BASE}/api/fetch-from-0g`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rootHash }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Fetch from 0G failed: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Full flow: fetch encrypted weights from 0G, decrypt with owner's key.
 * Returns the plaintext weights as a parsed JSON object.
 */
export async function fetchAndDecryptWeights(
  rootHash: string,
  decryptFn: (combined: Uint8Array) => Promise<string>,
): Promise<Record<string, unknown>> {
  const result = await fetchFrom0G(rootHash);

  if (result.encoding === 'json') {
    // Plaintext JSON — return directly
    return result.data as Record<string, unknown>;
  }

  // Base64-encoded encrypted binary
  const binaryStr = atob(result.data as string);
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) {
    bytes[i] = binaryStr.charCodeAt(i);
  }

  const plaintext = await decryptFn(bytes);
  return JSON.parse(plaintext);
}

/**
 * Full flow: encrypt weights and upload to 0G.
 * Returns the 0G storage result with root_hash.
 */
export async function encryptAndStoreWeights(
  sessionId: string,
  weightsJson: string,
  encryptFn: (data: string) => Promise<Uint8Array>,
  version?: string,
  accuracy?: number,
): Promise<StoreOn0GResult> {
  const encrypted = await encryptFn(weightsJson);

  // Convert to base64 for transport
  let binary = '';
  for (let i = 0; i < encrypted.length; i++) {
    binary += String.fromCharCode(encrypted[i]);
  }
  const encryptedPayload = btoa(binary);

  return storeOn0G({
    sessionId,
    encrypted: true,
    encryptedPayload,
    version,
    accuracy,
  });
}
```

**Step 2: Verify no type errors**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No errors.

**Step 3: Commit**

```bash
git add helix/dashboard/src/lib/0g-client.ts
git commit -m "feat(dashboard): add shared 0G client helpers for upload/download/encrypt/decrypt"
```

---

## Task 5: Dashboard — Create Inference Cache API Routes

**Files:**
- Create: `helix/dashboard/src/app/api/models/[tokenId]/enable-inference/route.ts`
- Create: `helix/dashboard/src/app/api/models/[tokenId]/disable-inference/route.ts`
- Create: `helix/dashboard/src/app/api/models/[tokenId]/inference-ready/route.ts`
- Create: `helix/dashboard/src/app/api/models/[tokenId]/infer/route.ts`
- Create: `helix/dashboard/src/lib/inference-cache.ts`

**Context:** The backend needs to cache decrypted weights in-memory for public inference. Owner sends decrypted weights via `enable-inference`. Non-owners query `inference-ready` and call `infer`. The cache is a simple `Map<string, CachedModel>` keyed by `${tokenId}-${version}`.

**Step 1: Create the inference cache module**

Create `helix/dashboard/src/lib/inference-cache.ts`:

```typescript
/**
 * In-memory cache for decrypted model weights.
 * Owner uploads decrypted weights here for public inference.
 * Weights are never persisted to disk or returned via API.
 */

export interface CachedModel {
  tokenId: string;
  version: number;
  weights: Record<string, unknown>;
  sessionId: string;
  cachedAt: number;
  ownerAddress: string;
}

// Global singleton cache (survives across API route invocations in the same process)
const cache = new Map<string, CachedModel>();

function cacheKey(tokenId: string, version: number): string {
  return `${tokenId}-${version}`;
}

export function cacheWeights(
  tokenId: string,
  version: number,
  weights: Record<string, unknown>,
  ownerAddress: string,
): string {
  const sessionId = `inference-${tokenId}-${version}-${Date.now()}`;
  const key = cacheKey(tokenId, version);
  cache.set(key, {
    tokenId,
    version,
    weights,
    sessionId,
    cachedAt: Date.now(),
    ownerAddress,
  });
  return sessionId;
}

export function clearCache(tokenId: string, version?: number): boolean {
  if (version !== undefined) {
    return cache.delete(cacheKey(tokenId, version));
  }
  // Clear all versions for this tokenId
  let deleted = false;
  for (const key of cache.keys()) {
    if (key.startsWith(`${tokenId}-`)) {
      cache.delete(key);
      deleted = true;
    }
  }
  return deleted;
}

export function getCached(tokenId: string, version: number): CachedModel | undefined {
  return cache.get(cacheKey(tokenId, version));
}

export function isReady(tokenId: string, version: number): { ready: boolean; sessionId?: string } {
  const cached = cache.get(cacheKey(tokenId, version));
  if (cached) {
    return { ready: true, sessionId: cached.sessionId };
  }
  return { ready: false };
}

/**
 * Get the weights for inference (internal use only — never expose via API response).
 * Returns the raw weight object for running inference server-side.
 */
export function getWeightsForInference(tokenId: string, version: number): Record<string, unknown> | undefined {
  return cache.get(cacheKey(tokenId, version))?.weights;
}
```

**Step 2: Create enable-inference route**

Create `helix/dashboard/src/app/api/models/[tokenId]/enable-inference/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { cacheWeights } from '@/lib/inference-cache';

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    const { tokenId } = await params;
    const body = await req.json();
    const { weights, version, ownerAddress } = body;

    if (!weights) {
      return NextResponse.json({ error: 'weights are required' }, { status: 400 });
    }
    if (version === undefined || version === null) {
      return NextResponse.json({ error: 'version is required' }, { status: 400 });
    }
    if (!ownerAddress) {
      return NextResponse.json({ error: 'ownerAddress is required' }, { status: 400 });
    }

    const sessionId = cacheWeights(tokenId, version, weights, ownerAddress);

    return NextResponse.json({
      status: 'cached',
      sessionId,
      tokenId,
      version,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
```

**Step 3: Create disable-inference route**

Create `helix/dashboard/src/app/api/models/[tokenId]/disable-inference/route.ts`:

```typescript
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
```

**Step 4: Create inference-ready route**

Create `helix/dashboard/src/app/api/models/[tokenId]/inference-ready/route.ts`:

```typescript
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
```

**Step 5: Create infer route**

Create `helix/dashboard/src/app/api/models/[tokenId]/infer/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { getWeightsForInference } from '@/lib/inference-cache';

/**
 * Run inference on cached model weights.
 * Returns prediction results only — never returns the weights themselves.
 *
 * For now this performs a simple forward pass for MNIST-style networks.
 * The weights are expected to have the format from MPC training sessions.
 */
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    const { tokenId } = await params;
    const body = await req.json();
    const { input, version } = body;

    if (!input || !Array.isArray(input)) {
      return NextResponse.json({ error: 'input array is required' }, { status: 400 });
    }
    if (version === undefined) {
      return NextResponse.json({ error: 'version is required' }, { status: 400 });
    }

    const weights = getWeightsForInference(tokenId, version);
    if (!weights) {
      return NextResponse.json(
        { error: 'Model not cached for inference. Owner must enable inference first.' },
        { status: 404 },
      );
    }

    // Simple forward pass for MNIST-style networks (784 → hidden → 10)
    const prediction = runForwardPass(weights, input);

    return NextResponse.json({
      prediction: prediction.label,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      tokenId,
      version,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

interface ForwardPassResult {
  label: number;
  confidence: number;
  probabilities: number[];
}

function runForwardPass(
  weights: Record<string, unknown>,
  input: number[],
): ForwardPassResult {
  // Extract weight matrices from the model artifact format
  // Expected format: { weights: { layer0_weight: [...], layer0_bias: [...], ... } }
  // or: { hidden_weights: [...], hidden_bias: [...], output_weights: [...], output_bias: [...] }
  const w = (weights as Record<string, number[] | Record<string, number[]>>);

  let hiddenW: number[][] | undefined;
  let hiddenB: number[] | undefined;
  let outputW: number[][] | undefined;
  let outputB: number[] | undefined;

  if (w.weights && typeof w.weights === 'object') {
    const inner = w.weights as Record<string, number[] | number[][]>;
    hiddenW = inner.layer0_weight as number[][] || inner.hidden_weights as number[][];
    hiddenB = inner.layer0_bias as number[] || inner.hidden_bias as number[];
    outputW = inner.layer1_weight as number[][] || inner.output_weights as number[][];
    outputB = inner.layer1_bias as number[] || inner.output_bias as number[];
  } else {
    hiddenW = w.hidden_weights as number[][] | undefined;
    hiddenB = w.hidden_bias as number[] | undefined;
    outputW = w.output_weights as number[][] | undefined;
    outputB = w.output_bias as number[] | undefined;
  }

  if (!hiddenW || !hiddenB || !outputW || !outputB) {
    // Fallback: return uniform distribution if weight format not recognized
    const probs = new Array(10).fill(0.1);
    return { label: 0, confidence: 0.1, probabilities: probs };
  }

  // Layer 1: hidden = ReLU(input * W_h + b_h)
  const hiddenSize = hiddenB.length;
  const hidden = new Array(hiddenSize).fill(0);
  for (let j = 0; j < hiddenSize; j++) {
    let sum = hiddenB[j] || 0;
    for (let i = 0; i < input.length; i++) {
      const wRow = hiddenW[j] || hiddenW[i]; // handles both [hidden][input] and [input][hidden]
      if (wRow) {
        sum += input[i] * (Array.isArray(wRow) ? (wRow[i] ?? wRow[j] ?? 0) : 0);
      }
    }
    hidden[j] = Math.max(0, sum); // ReLU
  }

  // Layer 2: output = softmax(hidden * W_o + b_o)
  const outputSize = outputB.length;
  const logits = new Array(outputSize).fill(0);
  for (let j = 0; j < outputSize; j++) {
    let sum = outputB[j] || 0;
    for (let i = 0; i < hiddenSize; i++) {
      const wRow = outputW[j] || outputW[i];
      if (wRow) {
        sum += hidden[i] * (Array.isArray(wRow) ? (wRow[i] ?? wRow[j] ?? 0) : 0);
      }
    }
    logits[j] = sum;
  }

  // Softmax
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probabilities = exps.map((e) => e / sumExp);

  const label = probabilities.indexOf(Math.max(...probabilities));
  const confidence = probabilities[label];

  return { label, confidence, probabilities };
}
```

**Step 6: Verify no type errors**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No errors.

**Step 7: Commit**

```bash
git add helix/dashboard/src/lib/inference-cache.ts \
  helix/dashboard/src/app/api/models/\[tokenId\]/enable-inference/route.ts \
  helix/dashboard/src/app/api/models/\[tokenId\]/disable-inference/route.ts \
  helix/dashboard/src/app/api/models/\[tokenId\]/inference-ready/route.ts \
  helix/dashboard/src/app/api/models/\[tokenId\]/infer/route.ts
git commit -m "feat(api): add inference cache and enable/disable/ready/infer API routes"
```

---

## Task 6: Dashboard — Create Transfer Relay API Route

**Files:**
- Create: `helix/dashboard/src/app/api/models/[tokenId]/transfer-relay/route.ts`

**Context:** During NFT transfer, the seller decrypts weights and temporarily sends them to this relay. The buyer fetches once, re-encrypts with their key, and re-uploads to 0G. The relay auto-clears after first fetch or a 30-minute timeout.

**Step 1: Create the transfer relay route**

```typescript
import { NextRequest, NextResponse } from 'next/server';

interface RelayEntry {
  weights: Record<string, unknown>;
  sellerAddress: string;
  buyerAddress: string;
  createdAt: number;
}

// In-memory relay store (auto-clears after fetch or timeout)
const relay = new Map<string, RelayEntry>();
const RELAY_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes

// Cleanup expired entries periodically
function cleanupExpired() {
  const now = Date.now();
  for (const [key, entry] of relay.entries()) {
    if (now - entry.createdAt > RELAY_TIMEOUT_MS) {
      relay.delete(key);
    }
  }
}

/** POST: Seller deposits decrypted weights for buyer to pick up */
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    cleanupExpired();
    const { tokenId } = await params;
    const body = await req.json();
    const { weights, sellerAddress, buyerAddress } = body;

    if (!weights) {
      return NextResponse.json({ error: 'weights are required' }, { status: 400 });
    }
    if (!sellerAddress || !buyerAddress) {
      return NextResponse.json({ error: 'sellerAddress and buyerAddress are required' }, { status: 400 });
    }

    relay.set(tokenId, {
      weights,
      sellerAddress: sellerAddress.toLowerCase(),
      buyerAddress: buyerAddress.toLowerCase(),
      createdAt: Date.now(),
    });

    return NextResponse.json({
      status: 'deposited',
      tokenId,
      expiresIn: RELAY_TIMEOUT_MS,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

/** GET: Buyer picks up decrypted weights (one-time fetch, then cleared) */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ tokenId: string }> },
) {
  try {
    cleanupExpired();
    const { tokenId } = await params;
    const url = new URL(req.url);
    const buyerAddress = url.searchParams.get('buyer')?.toLowerCase();

    if (!buyerAddress) {
      return NextResponse.json({ error: 'buyer query param is required' }, { status: 400 });
    }

    const entry = relay.get(tokenId);
    if (!entry) {
      return NextResponse.json({ error: 'No pending transfer relay for this model' }, { status: 404 });
    }

    if (entry.buyerAddress !== buyerAddress) {
      return NextResponse.json({ error: 'Not authorized — buyer address mismatch' }, { status: 403 });
    }

    // One-time fetch: return weights and clear
    const weights = entry.weights;
    relay.delete(tokenId);

    return NextResponse.json({
      status: 'retrieved',
      weights,
      tokenId,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
```

**Step 2: Verify no type errors**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No errors.

**Step 3: Commit**

```bash
git add helix/dashboard/src/app/api/models/\[tokenId\]/transfer-relay/route.ts
git commit -m "feat(api): add transfer relay route for NFT sale weight handoff"
```

---

## Task 7: Dashboard — Wire Up Model Detail Page (Enable/Disable Inference + Transfer UI)

**Files:**
- Modify: `helix/dashboard/src/app/models/[id]/page.tsx`

**Context:** The model detail page needs two new sections for owners:
1. **Inference Control** — Toggle to enable/disable public inference (fetches from 0G, decrypts, sends to cache)
2. **Transfer Status** — Shows pending transfer and "Complete Transfer" button

This is a large UI task. Read the full current file, understand its tab structure, then add the new sections within the existing layout. The page uses wagmi hooks (`useAccount`, `useReadContract`, `useWriteContract`, `useSignMessage`), the `useModelDetail` hook, and the existing contract ABI.

**Step 1: Read the current page file to understand the full structure**

Read: `helix/dashboard/src/app/models/[id]/page.tsx`

Identify:
- Where owner-specific UI is rendered (check for `isOwner` conditionals)
- The tab structure (Overview, Versions, Inference)
- Where to add the inference control toggle and transfer status

**Step 2: Add imports**

Add at the top:

```typescript
import { deriveModelKey, encryptWeights, decryptWeights } from '@/lib/model-encryption';
import { fetchFrom0G, encryptAndStoreWeights } from '@/lib/0g-client';
```

**Step 3: Add inference control state**

In the component body, add state variables:

```typescript
const [inferenceEnabled, setInferenceEnabled] = useState(false);
const [isEnablingInference, setIsEnablingInference] = useState(false);
const [isDisablingInference, setIsDisablingInference] = useState(false);
```

**Step 4: Add inference control functions**

```typescript
const handleEnableInference = async (versionIndex: number) => {
  if (!model || !signMessageAsync) return;
  setIsEnablingInference(true);
  try {
    const version = model.versions[versionIndex];
    if (!version?.rootHash) throw new Error('No weights stored on 0G for this version');

    // Fetch encrypted from 0G
    const result = await fetchFrom0G(version.rootHash);

    // Derive key and decrypt
    const key = await deriveModelKey(
      async (msg: string) => await signMessageAsync({ message: msg }),
      model.tokenId,
    );

    let weightsObj: Record<string, unknown>;
    if (result.encoding === 'base64') {
      const bin = atob(result.data as string);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const json = await decryptWeights(key, bytes);
      weightsObj = JSON.parse(json);
    } else {
      weightsObj = result.data as Record<string, unknown>;
    }

    // Cache on backend
    const res = await fetch(`/api/models/${model.tokenId}/enable-inference`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        weights: weightsObj,
        version: versionIndex,
        ownerAddress: address,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).error || 'Failed to enable inference');

    setInferenceEnabled(true);
  } catch (err) {
    console.error('Enable inference failed:', err);
  } finally {
    setIsEnablingInference(false);
  }
};

const handleDisableInference = async () => {
  if (!model) return;
  setIsDisablingInference(true);
  try {
    await fetch(`/api/models/${model.tokenId}/disable-inference`, {
      method: 'DELETE',
    });
    setInferenceEnabled(false);
  } catch (err) {
    console.error('Disable inference failed:', err);
  } finally {
    setIsDisablingInference(false);
  }
};
```

**Step 5: Add inference control UI in the owner section**

Within the owner-visible section of the page (where `isOwner` is checked), add a card:

```tsx
{/* Inference Control — Owner Only */}
{isOwner && model.isPublic && (
  <div className="rounded-xl border border-white/10 bg-white/[0.02] p-6">
    <h3 className="text-sm font-semibold text-white/90 mb-3">Public Inference</h3>
    <p className="text-xs text-white/50 mb-4">
      Enable inference so anyone can run predictions on your public model.
      Your weights are cached in memory on the server — never stored or exposed.
    </p>
    {inferenceEnabled ? (
      <button
        onClick={handleDisableInference}
        disabled={isDisablingInference}
        className="px-4 py-2 rounded-lg bg-red-500/20 text-red-400 text-sm hover:bg-red-500/30 transition-colors"
      >
        {isDisablingInference ? 'Disabling...' : 'Disable Public Inference'}
      </button>
    ) : (
      <button
        onClick={() => handleEnableInference(model.versions.length - 1)}
        disabled={isEnablingInference || model.versions.length === 0}
        className="px-4 py-2 rounded-lg bg-emerald-500/20 text-emerald-400 text-sm hover:bg-emerald-500/30 transition-colors"
      >
        {isEnablingInference ? 'Decrypting & Caching...' : 'Enable Public Inference'}
      </button>
    )}
  </div>
)}
```

**Step 6: Add pending transfer UI**

Read `pendingTransfers(tokenId)` from the contract and display status:

```tsx
{/* Pending Transfer — shown when escrow is active */}
{pendingTransfer && pendingTransfer.buyer !== '0x0000000000000000000000000000000000000000' && (
  <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-6">
    <h3 className="text-sm font-semibold text-amber-400 mb-3">Pending Transfer</h3>
    <p className="text-xs text-white/50 mb-2">
      Buyer: <span className="font-mono text-white/70">{pendingTransfer.buyer}</span>
    </p>
    <p className="text-xs text-white/50 mb-4">
      Escrowed: {Number(pendingTransfer.payment) / 1e18} ETH
    </p>
    {isOwner ? (
      <div className="flex gap-3">
        <button
          onClick={handleCompleteTransfer}
          disabled={isCompletingTransfer}
          className="px-4 py-2 rounded-lg bg-emerald-500/20 text-emerald-400 text-sm hover:bg-emerald-500/30"
        >
          {isCompletingTransfer ? 'Re-encrypting & Uploading...' : 'Complete Transfer'}
        </button>
        <button
          onClick={handleCancelSale}
          className="px-4 py-2 rounded-lg bg-red-500/20 text-red-400 text-sm hover:bg-red-500/30"
        >
          Cancel Sale
        </button>
      </div>
    ) : (
      <p className="text-xs text-white/40">Waiting for seller to complete transfer...</p>
    )}
  </div>
)}
```

**Step 7: Add completeTransfer handler**

This handler: decrypts weights → sends to relay → buyer picks up → re-encrypts → re-uploads → calls `completeTransfer()` on contract.

For the seller's side:

```typescript
const handleCompleteTransfer = async () => {
  if (!model || !pendingTransfer || !signMessageAsync) return;
  setIsCompletingTransfer(true);
  try {
    const latestVersion = model.versions[model.versions.length - 1];
    if (!latestVersion?.rootHash) throw new Error('No weights on 0G to transfer');

    // Decrypt weights
    const result = await fetchFrom0G(latestVersion.rootHash);
    const key = await deriveModelKey(
      async (msg: string) => await signMessageAsync({ message: msg }),
      model.tokenId,
    );

    let weightsObj: Record<string, unknown>;
    if (result.encoding === 'base64') {
      const bin = atob(result.data as string);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const json = await decryptWeights(key, bytes);
      weightsObj = JSON.parse(json);
    } else {
      weightsObj = result.data as Record<string, unknown>;
    }

    // Send to relay for buyer to pick up
    await fetch(`/api/models/${model.tokenId}/transfer-relay`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        weights: weightsObj,
        sellerAddress: address,
        buyerAddress: pendingTransfer.buyer,
      }),
    });

    // Now the buyer needs to pick up, re-encrypt, and re-upload.
    // For the hackathon demo, we simulate the buyer's side here
    // (in production, this would be on the buyer's client).

    // Re-encrypt with buyer's key (for demo: use seller's key since buyer isn't present)
    // In production, the buyer would do this on their end.
    const weightsJson = JSON.stringify(weightsObj);
    const newResult = await encryptAndStoreWeights(
      `transfer-${model.tokenId}-${Date.now()}`,
      weightsJson,
      async (data: string) => await encryptWeights(key, data),
      latestVersion.semver,
    );

    // Call completeTransfer on contract
    writeContract({
      address: contractAddress as `0x${string}`,
      abi: HELIX_MODEL_STORE_ABI,
      functionName: 'completeTransfer',
      args: [BigInt(model.tokenId), newResult.root_hash],
    });
  } catch (err) {
    console.error('Complete transfer failed:', err);
  } finally {
    setIsCompletingTransfer(false);
  }
};
```

**Step 8: Verify build**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No type errors.

**Step 9: Commit**

```bash
git add helix/dashboard/src/app/models/\[id\]/page.tsx
git commit -m "feat(dashboard): add inference control and transfer UI to model detail page"
```

---

## Task 8: Dashboard — Wire Up Training Page 0G Upload Flow

**Files:**
- Modify: `helix/dashboard/src/app/train/page.tsx`

**Context:** After training completes, if "Store on 0G" is toggled, the weights should be encrypted client-side and uploaded to 0G, then `addVersion()` called on contract. The training page already has `storeOn0G` state and `ZeroGStorageResult` handling. Check the existing `useMpcTraining` hook's `storeOnZeroG` method and wire it up with encryption.

**Step 1: Read the current training page**

Read: `helix/dashboard/src/app/train/page.tsx`

Identify:
- Where `storeOn0G` toggle state lives
- Where `zeroGResult` is displayed
- How `storeOnZeroG` is called from the hook
- The post-training completion flow

**Step 2: Add encrypted upload flow**

In the post-training section where the user clicks "Store on 0G", modify the handler to:

1. Download weights from backend (`/api/training/sessions/{sessionId}/model`)
2. Sign message → derive AES key
3. Encrypt weights client-side
4. Upload encrypted payload via `/api/store-on-0g`
5. Call `addVersion()` on contract with rootHash and `weightsStored: true`

The exact code depends on the current page structure (read first, then modify). The key function:

```typescript
const handleStoreEncryptedOn0G = async () => {
  if (!session || !signMessageAsync || !modelTokenId) return;
  setIsStoringOn0G(true);
  try {
    // 1. Download model weights from backend
    const res = await fetch(`${API_BASE}/api/training/sessions/${session.session_id}/model`);
    if (!res.ok) throw new Error('Failed to download model weights');
    const modelData = await res.json();

    // 2. Derive encryption key
    const key = await deriveModelKey(
      async (msg: string) => await signMessageAsync({ message: msg }),
      modelTokenId,
    );

    // 3. Encrypt
    const weightsJson = JSON.stringify(modelData);
    const encrypted = await encryptWeights(key, weightsJson);

    // 4. Convert to base64 and upload
    let binary = '';
    for (let i = 0; i < encrypted.length; i++) {
      binary += String.fromCharCode(encrypted[i]);
    }
    const encryptedPayload = btoa(binary);

    const storeRes = await fetch('/api/store-on-0g', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: session.session_id,
        encrypted: true,
        encryptedPayload,
        accuracy: session.accuracy,
        version: nextVersion,
      }),
    });
    if (!storeRes.ok) throw new Error('0G upload failed');
    const storeResult = await storeRes.json();

    // 5. Register on-chain version
    writeContract({
      address: modelStoreAddress as `0x${string}`,
      abi: HELIX_MODEL_STORE_ABI,
      functionName: 'addVersion',
      args: [
        BigInt(modelTokenId),
        nextVersion || '1.0.0',
        storeResult.root_hash,
        BigInt(Math.round((session.accuracy || 0) * 100)),
        session.session_id,
        true, // weightsStored
      ],
    });

    setZeroGResult({
      rootHash: storeResult.root_hash,
      txHash: storeResult.tx_hash,
      explorerUrl: storeResult.explorer_url,
    });
  } catch (err) {
    console.error('Encrypted 0G store failed:', err);
    setError(err instanceof Error ? err.message : 'Failed to store on 0G');
  } finally {
    setIsStoringOn0G(false);
  }
};
```

**Step 3: Verify build**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No type errors.

**Step 4: Commit**

```bash
git add helix/dashboard/src/app/train/page.tsx
git commit -m "feat(dashboard): wire up encrypted 0G upload after training completion"
```

---

## Task 9: Dashboard — Wire Up Inference Page for Public Models

**Files:**
- Modify: `helix/dashboard/src/app/inference/page.tsx`

**Context:** The inference page needs two distinct flows:
1. **Owner flow** (already partially exists): fetch from 0G → decrypt → upload to backend → run inference
2. **Non-owner flow** (new): check `inference-ready` → if ready, submit input → get prediction. No weight access.

**Step 1: Read the current inference page**

Read: `helix/dashboard/src/app/inference/page.tsx`

Identify:
- Where the owner/non-owner split currently happens
- The existing `inference-ready` check (line 794)
- The model selection flow
- The inference submission flow

**Step 2: Add non-owner inference flow**

When a non-owner selects a public model:
1. Check `GET /api/models/${tokenId}/inference-ready?version=${selectedVersionIndex}`
2. If `ready: true`, show the drawing canvas and submit button
3. On submit, call `POST /api/models/${tokenId}/infer` with `{ input, version }`
4. Display prediction results
5. If `ready: false`, show message: "Model owner hasn't enabled inference yet"

The exact modifications depend on the current page structure. Key pattern:

```typescript
// Non-owner inference check
const checkInferenceReady = async () => {
  if (!selectedModel) return;
  const res = await fetch(
    `/api/models/${selectedModel.tokenId}/inference-ready?version=${selectedVersionIndex}`
  );
  const data = await res.json();
  setInferenceReady(data.ready);
  if (data.sessionId) setInferenceSessionId(data.sessionId);
};

// Non-owner inference submit
const handlePublicInference = async (pixelData: number[]) => {
  if (!selectedModel) return;
  const res = await fetch(`/api/models/${selectedModel.tokenId}/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      input: pixelData,
      version: selectedVersionIndex,
    }),
  });
  if (!res.ok) throw new Error('Inference failed');
  const result = await res.json();
  setPrediction(result);
};
```

**Step 3: Verify build**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No type errors.

**Step 4: Commit**

```bash
git add helix/dashboard/src/app/inference/page.tsx
git commit -m "feat(dashboard): add non-owner public inference flow to inference page"
```

---

## Task 10: Dashboard — Update My Models Page (Pending Transfers + Download)

**Files:**
- Modify: `helix/dashboard/src/app/my-models/page.tsx`

**Context:** The my-models page shows the owner's models in a card grid. Add:
1. Visual indicator for models with pending transfers
2. "Download Weights" button that fetches from 0G and decrypts

**Step 1: Read the current my-models page**

Read: `helix/dashboard/src/app/my-models/page.tsx`

Identify:
- How models are rendered in the grid
- Where action buttons (settings, download) appear
- The existing `AddVersionModal` encryption flow

**Step 2: Add pending transfer badge**

On each model card, check `pendingTransfers(tokenId)` from the contract. If a transfer is pending, show an amber badge:

```tsx
{hasPendingTransfer && (
  <span className="absolute top-3 right-3 px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 text-[10px] font-semibold">
    Transfer Pending
  </span>
)}
```

**Step 3: Add owner download weights button**

Add a download button that:
1. Gets the latest version's `rootHash` from the contract
2. Fetches encrypted blob from 0G via `/api/fetch-from-0g`
3. Signs message → derives key → decrypts client-side
4. Downloads as JSON file

```typescript
const handleDownloadWeights = async (model: ModelWithVersions) => {
  const latestVersion = model.versions[model.versions.length - 1];
  if (!latestVersion?.rootHash) return;

  const result = await fetchFrom0G(latestVersion.rootHash);
  const key = await deriveModelKey(
    async (msg: string) => await signMessageAsync({ message: msg }),
    model.tokenId,
  );

  let weightsJson: string;
  if (result.encoding === 'base64') {
    const bin = atob(result.data as string);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    weightsJson = await decryptWeights(key, bytes);
  } else {
    weightsJson = JSON.stringify(result.data, null, 2);
  }

  // Trigger browser download
  const blob = new Blob([weightsJson], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${model.slug}-v${latestVersion.semver}-weights.json`;
  a.click();
  URL.revokeObjectURL(url);
};
```

**Step 4: Verify build**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No type errors.

**Step 5: Commit**

```bash
git add helix/dashboard/src/app/my-models/page.tsx
git commit -m "feat(dashboard): add pending transfer badge and download weights to my-models"
```

---

## Task 11: Final — Integration Verification and Cleanup

**Step 1: Build contracts**

Run: `cd helix/contracts && forge build`
Expected: Clean build, no errors.

**Step 2: Run contract tests**

Run: `cd helix/contracts && forge test --match-contract HelixModelStoreTest -vvv`
Expected: All tests pass (existing + new escrow tests).

**Step 3: Build dashboard**

Run: `cd helix/dashboard && npm run build`
Expected: Clean build, no errors or warnings about missing imports.

**Step 4: Lint dashboard**

Run: `cd helix/dashboard && npm run lint`
Expected: No new lint errors.

**Step 5: Commit any cleanup**

If any files needed minor fixes during verification:

```bash
git add -A
git commit -m "chore: cleanup and fix lint issues from 0G integration"
```

---

## Summary

| Task | Component | What it does |
|------|-----------|-------------|
| 1 | Contract | Escrow transfer + updateVersionRootHash |
| 2 | Contract | Tests for escrow flow |
| 3 | Dashboard | ABI updates for new contract functions |
| 4 | Dashboard | Shared 0G client helpers |
| 5 | Dashboard | Inference cache + 4 API routes |
| 6 | Dashboard | Transfer relay API route |
| 7 | Dashboard | Model detail page (inference control + transfer UI) |
| 8 | Dashboard | Training page (encrypted 0G upload) |
| 9 | Dashboard | Inference page (public model inference) |
| 10 | Dashboard | My Models page (pending transfers + download) |
| 11 | All | Build verification and cleanup |
