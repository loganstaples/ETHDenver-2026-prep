# ERC-721 Model Registry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the flat HelixModelStore with an ERC-721 NFT-based model registry where each model is an NFT with version history, encrypted weight storage on 0G, and owner-only access control.

**Architecture:** Each model = ERC-721 token. On-chain: model metadata (slug, name, description, creator) + version array (semver, encrypted 0G rootHash, accuracy). Off-chain: AES-256-GCM encrypted weights on 0G storage, decryption key derived from wallet signature. Dashboard reads NFTs owned by connected wallet.

**Tech Stack:** Solidity ^0.8.19 (OpenZeppelin ERC721Enumerable), Foundry, Next.js/React, wagmi/viem, Web Crypto API (AES-256-GCM)

---

### Task 1: Write the ERC-721 HelixModelStore contract

**Files:**
- Modify: `helix/contracts/src/core/HelixModelStore.sol` (complete rewrite)

**Step 1: Write the contract test**

Create `helix/contracts/test/HelixModelStore.t.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixModelStore.sol";

contract HelixModelStoreTest is Test {
    HelixModelStore store;
    address alice = makeAddr("alice");
    address bob = makeAddr("bob");

    function setUp() public {
        store = new HelixModelStore();
    }

    // ── createModel ─────────────────────────────────────────────────

    function test_createModel_mintsNFT() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("mnist-v1", "MNIST Classifier", "Handwritten digit recognition");

        assertEq(store.ownerOf(tokenId), alice);
        assertEq(store.balanceOf(alice), 1);
    }

    function test_createModel_storesMetadata() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("mnist-v1", "MNIST Classifier", "Digit recognition");

        (
            string memory slug,
            string memory name,
            string memory description,
            address creator,
            uint40 createdAt,
            bool isPublic,
            uint16 inferenceFee
        ) = store.models(tokenId);

        assertEq(slug, "mnist-v1");
        assertEq(name, "MNIST Classifier");
        assertEq(description, "Digit recognition");
        assertEq(creator, alice);
        assertGt(createdAt, 0);
        assertFalse(isPublic);
        assertEq(inferenceFee, 0);
    }

    function test_createModel_slugLookup() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("mnist-v1", "MNIST", "");

        assertEq(store.getModelBySlug("mnist-v1"), tokenId);
    }

    function test_createModel_duplicateSlugReverts() public {
        vm.prank(alice);
        store.createModel("mnist-v1", "MNIST", "");

        vm.prank(bob);
        vm.expectRevert("Slug already taken");
        store.createModel("mnist-v1", "Other", "");
    }

    function test_createModel_emptySlugReverts() public {
        vm.prank(alice);
        vm.expectRevert("Slug required");
        store.createModel("", "MNIST", "");
    }

    function test_createModel_emptyNameReverts() public {
        vm.prank(alice);
        vm.expectRevert("Name required");
        store.createModel("mnist-v1", "", "");
    }

    // ── addVersion ──────────────────────────────────────────────────

    function test_addVersion_ownerOnly() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(alice);
        uint256 vIdx = store.addVersion(tokenId, "1.0.0", "0xabc123", 9500, "sess-1", true);
        assertEq(vIdx, 0);
        assertEq(store.getVersionCount(tokenId), 1);
    }

    function test_addVersion_nonOwnerReverts() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(bob);
        vm.expectRevert("Not model owner");
        store.addVersion(tokenId, "1.0.0", "", 0, "", false);
    }

    function test_addVersion_storesData() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(alice);
        store.addVersion(tokenId, "1.0.0", "0xrootHash", 9500, "sess-abc", true);

        HelixModelStore.Version memory v = store.getVersion(tokenId, 0);
        assertEq(v.semver, "1.0.0");
        assertEq(v.rootHash, "0xrootHash");
        assertEq(v.accuracy, 9500);
        assertEq(v.sessionId, "sess-abc");
        assertTrue(v.weightsStored);
    }

    function test_addVersion_multipleVersions() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(alice);
        store.addVersion(tokenId, "1.0.0", "", 8000, "s1", false);

        vm.prank(alice);
        store.addVersion(tokenId, "1.1.0", "0xhash", 9200, "s2", true);

        assertEq(store.getVersionCount(tokenId), 2);

        HelixModelStore.Version[] memory all = store.getVersions(tokenId);
        assertEq(all.length, 2);
        assertEq(all[0].semver, "1.0.0");
        assertEq(all[1].semver, "1.1.0");
    }

    function test_addVersion_withoutWeights() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(alice);
        store.addVersion(tokenId, "1.0.0", "", 9000, "s1", false);

        HelixModelStore.Version memory v = store.getVersion(tokenId, 0);
        assertEq(v.rootHash, "");
        assertFalse(v.weightsStored);
    }

    // ── Access control (future) ─────────────────────────────────────

    function test_hasModelAccess_ownerHasAccess() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        assertTrue(store.hasModelAccess(tokenId, alice));
        assertFalse(store.hasModelAccess(tokenId, bob));
    }

    function test_grantAccess() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(alice);
        store.grantAccess(tokenId, bob);

        assertTrue(store.hasModelAccess(tokenId, bob));
    }

    function test_revokeAccess() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(alice);
        store.grantAccess(tokenId, bob);

        vm.prank(alice);
        store.revokeAccess(tokenId, bob);

        assertFalse(store.hasModelAccess(tokenId, bob));
    }

    function test_grantAccess_nonOwnerReverts() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(bob);
        vm.expectRevert("Not model owner");
        store.grantAccess(tokenId, bob);
    }

    // ── setPublic / setInferenceFee ─────────────────────────────────

    function test_setPublic() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(alice);
        store.setPublic(tokenId, true);

        (,,,,, bool isPublic,) = store.models(tokenId);
        assertTrue(isPublic);

        // Public models grant access to everyone
        assertTrue(store.hasModelAccess(tokenId, bob));
    }

    function test_setInferenceFee() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(alice);
        store.setInferenceFee(tokenId, 500); // 5%

        (,,,,,, uint16 fee) = store.models(tokenId);
        assertEq(fee, 500);
    }

    function test_setInferenceFee_maxCap() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(alice);
        vm.expectRevert("Fee exceeds 50%");
        store.setInferenceFee(tokenId, 5001);
    }

    // ── Transfer preserves creator ──────────────────────────────────

    function test_transfer_preservesCreator() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(alice);
        store.transferFrom(alice, bob, tokenId);

        assertEq(store.ownerOf(tokenId), bob);
        (,,, address creator,,,) = store.models(tokenId);
        assertEq(creator, alice); // Creator stays alice
    }

    function test_transfer_newOwnerCanAddVersions() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(alice);
        store.transferFrom(alice, bob, tokenId);

        // Bob (new owner) can add versions
        vm.prank(bob);
        store.addVersion(tokenId, "2.0.0", "0xnew", 9700, "s-new", true);

        assertEq(store.getVersionCount(tokenId), 1);
    }

    // ── Enumeration ─────────────────────────────────────────────────

    function test_enumeration() public {
        vm.startPrank(alice);
        store.createModel("m1", "Model 1", "");
        store.createModel("m2", "Model 2", "");
        store.createModel("m3", "Model 3", "");
        vm.stopPrank();

        assertEq(store.balanceOf(alice), 3);
        assertEq(store.totalSupply(), 3);
    }

    // ── Events ──────────────────────────────────────────────────────

    function test_createModel_emitsEvent() public {
        vm.prank(alice);
        vm.expectEmit(true, true, false, true);
        emit HelixModelStore.ModelCreated(0, alice, "mnist-v1", "MNIST");
        store.createModel("mnist-v1", "MNIST", "");
    }

    function test_addVersion_emitsEvent() public {
        vm.prank(alice);
        uint256 tokenId = store.createModel("m1", "Model", "");

        vm.prank(alice);
        vm.expectEmit(true, true, false, true);
        emit HelixModelStore.VersionAdded(tokenId, 0, "1.0.0", "0xhash");
        store.addVersion(tokenId, "1.0.0", "0xhash", 9500, "s1", true);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd helix/contracts && forge test --match-contract HelixModelStoreTest -vvv`
Expected: Compilation failure (contract doesn't match)

**Step 3: Write the contract implementation**

Rewrite `helix/contracts/src/core/HelixModelStore.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";

/// @title HelixModelStore
/// @notice ERC-721 model registry. Each model is an NFT with version history
///         and links to encrypted weights on 0G Storage.
contract HelixModelStore is ERC721Enumerable {
    struct Model {
        string slug;           // user-chosen identifier (globally unique)
        string name;           // display name
        string description;
        address creator;       // original creator (immutable, survives transfer)
        uint40 createdAt;
        bool isPublic;         // when true, anyone has access (for paid inference)
        uint16 inferenceFee;   // basis points (500 = 5%), max 5000 (50%)
    }

    struct Version {
        string semver;         // "1.0.0"
        string rootHash;       // 0G Storage root hash (encrypted), "" if not stored
        uint96 accuracy;       // scaled by 1e4 (9500 = 95.00%)
        uint40 timestamp;
        string sessionId;      // training session reference
        bool weightsStored;    // whether weights were uploaded to 0G
    }

    uint256 private _nextTokenId;

    mapping(uint256 => Model) public models;
    mapping(uint256 => Version[]) private _versions;
    mapping(bytes32 => uint256) private _slugToToken;
    mapping(bytes32 => bool) private _slugTaken;

    /// @dev tokenId => (address => hasAccess)
    mapping(uint256 => mapping(address => bool)) public accessGranted;

    // ── Events ──────────────────────────────────────────────────────

    event ModelCreated(
        uint256 indexed tokenId,
        address indexed creator,
        string slug,
        string name
    );

    event VersionAdded(
        uint256 indexed tokenId,
        uint256 indexed versionIndex,
        string semver,
        string rootHash
    );

    event ModelPublicityChanged(uint256 indexed tokenId, bool isPublic);
    event InferenceFeeChanged(uint256 indexed tokenId, uint16 feeBps);
    event AccessChanged(uint256 indexed tokenId, address indexed account, bool granted);

    // ── Modifiers ───────────────────────────────────────────────────

    modifier onlyModelOwner(uint256 tokenId) {
        require(ownerOf(tokenId) == msg.sender, "Not model owner");
        _;
    }

    // ── Constructor ─────────────────────────────────────────────────

    constructor() ERC721("Helix Model", "HMODEL") {}

    // ── Write functions ─────────────────────────────────────────────

    /// @notice Create a new model (mints an NFT to the caller)
    /// @param slug   Globally unique identifier chosen by the user
    /// @param name   Human-readable display name
    /// @param description Short description
    /// @return tokenId The NFT token ID
    function createModel(
        string calldata slug,
        string calldata name,
        string calldata description
    ) external returns (uint256 tokenId) {
        require(bytes(slug).length > 0, "Slug required");
        require(bytes(name).length > 0, "Name required");

        bytes32 slugHash = keccak256(abi.encodePacked(slug));
        require(!_slugTaken[slugHash], "Slug already taken");

        tokenId = _nextTokenId++;
        _safeMint(msg.sender, tokenId);

        models[tokenId] = Model({
            slug: slug,
            name: name,
            description: description,
            creator: msg.sender,
            createdAt: uint40(block.timestamp),
            isPublic: false,
            inferenceFee: 0
        });

        _slugTaken[slugHash] = true;
        _slugToToken[slugHash] = tokenId;

        emit ModelCreated(tokenId, msg.sender, slug, name);
    }

    /// @notice Add a new version to an existing model
    /// @param tokenId       Model NFT token ID
    /// @param semver        Semantic version string
    /// @param rootHash      0G Storage root hash (encrypted weights), "" if not stored
    /// @param accuracy      Accuracy scaled by 1e4 (9500 = 95.00%)
    /// @param sessionId     Training session reference
    /// @param weightsStored Whether weights were uploaded to 0G
    /// @return versionIndex Index of the new version
    function addVersion(
        uint256 tokenId,
        string calldata semver,
        string calldata rootHash,
        uint96 accuracy,
        string calldata sessionId,
        bool weightsStored
    ) external onlyModelOwner(tokenId) returns (uint256 versionIndex) {
        versionIndex = _versions[tokenId].length;
        _versions[tokenId].push(Version({
            semver: semver,
            rootHash: rootHash,
            accuracy: accuracy,
            timestamp: uint40(block.timestamp),
            sessionId: sessionId,
            weightsStored: weightsStored
        }));

        emit VersionAdded(tokenId, versionIndex, semver, rootHash);
    }

    /// @notice Set whether the model is publicly accessible (for paid inference)
    function setPublic(uint256 tokenId, bool _isPublic) external onlyModelOwner(tokenId) {
        models[tokenId].isPublic = _isPublic;
        emit ModelPublicityChanged(tokenId, _isPublic);
    }

    /// @notice Set the inference fee in basis points (max 5000 = 50%)
    function setInferenceFee(uint256 tokenId, uint16 feeBps) external onlyModelOwner(tokenId) {
        require(feeBps <= 5000, "Fee exceeds 50%");
        models[tokenId].inferenceFee = feeBps;
        emit InferenceFeeChanged(tokenId, feeBps);
    }

    /// @notice Grant access to a specific address (for future paid access)
    function grantAccess(uint256 tokenId, address account) external onlyModelOwner(tokenId) {
        accessGranted[tokenId][account] = true;
        emit AccessChanged(tokenId, account, true);
    }

    /// @notice Revoke access from a specific address
    function revokeAccess(uint256 tokenId, address account) external onlyModelOwner(tokenId) {
        accessGranted[tokenId][account] = false;
        emit AccessChanged(tokenId, account, false);
    }

    // ── Read functions ──────────────────────────────────────────────

    /// @notice Check if an address has access to a model
    /// @return true if owner, explicitly granted, or model is public
    function hasModelAccess(uint256 tokenId, address account) external view returns (bool) {
        if (ownerOf(tokenId) == account) return true;
        if (models[tokenId].isPublic) return true;
        return accessGranted[tokenId][account];
    }

    /// @notice Get the number of versions for a model
    function getVersionCount(uint256 tokenId) external view returns (uint256) {
        return _versions[tokenId].length;
    }

    /// @notice Get a specific version
    function getVersion(uint256 tokenId, uint256 index) external view returns (Version memory) {
        require(index < _versions[tokenId].length, "Invalid version index");
        return _versions[tokenId][index];
    }

    /// @notice Get all versions for a model
    function getVersions(uint256 tokenId) external view returns (Version[] memory) {
        return _versions[tokenId];
    }

    /// @notice Look up a model's token ID by its slug
    function getModelBySlug(string calldata slug) external view returns (uint256) {
        bytes32 slugHash = keccak256(abi.encodePacked(slug));
        require(_slugTaken[slugHash], "Slug not found");
        return _slugToToken[slugHash];
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cd helix/contracts && forge test --match-contract HelixModelStoreTest -vvv`
Expected: All tests pass

**Step 5: Commit**

```bash
git add helix/contracts/src/core/HelixModelStore.sol helix/contracts/test/HelixModelStore.t.sol
git commit -m "feat: ERC-721 model registry with version history and access control"
```

---

### Task 2: Update deploy script

**Files:**
- Modify: `helix/contracts/script/DeployModelStore.s.sol`

**Step 1: Update the deploy script to use new constructor**

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/core/HelixModelStore.sol";

contract DeployModelStore is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envOr(
            "PRIVATE_KEY",
            uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)
        );

        vm.startBroadcast(deployerPrivateKey);

        HelixModelStore store = new HelixModelStore();
        console.log("HelixModelStore (ERC-721) deployed at:", address(store));
        console.log("  Name:", store.name());
        console.log("  Symbol:", store.symbol());

        vm.stopBroadcast();
    }
}
```

**Step 2: Verify deploy script compiles**

Run: `cd helix/contracts && forge build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add helix/contracts/script/DeployModelStore.s.sol
git commit -m "feat: update deploy script for ERC-721 HelixModelStore"
```

---

### Task 3: Update dashboard contract ABI and types

**Files:**
- Modify: `helix/dashboard/src/lib/contracts.ts`

**Step 1: Replace `HELIX_MODEL_STORE_ABI` and types**

Replace the ABI constant and add new types. The new ABI must match the ERC-721 contract's functions:

- `createModel(string,string,string) → uint256`
- `addVersion(uint256,string,string,uint96,string,bool) → uint256`
- `models(uint256) → (string,string,string,address,uint40,bool,uint16)`
- `getVersions(uint256) → Version[]`
- `getVersionCount(uint256) → uint256`
- `getVersion(uint256,uint256) → Version`
- `getModelBySlug(string) → uint256`
- `hasModelAccess(uint256,address) → bool`
- `setPublic(uint256,bool)`
- `setInferenceFee(uint256,uint16)`
- `grantAccess(uint256,address)`
- `revokeAccess(uint256,address)`
- ERC721Enumerable: `balanceOf(address)`, `tokenOfOwnerByIndex(address,uint256)`, `ownerOf(uint256)`
- Events: `ModelCreated`, `VersionAdded`

Add TypeScript types:

```typescript
export interface OnChainModel {
    tokenId: bigint;
    slug: string;
    name: string;
    description: string;
    creator: string;
    createdAt: number;
    isPublic: boolean;
    inferenceFee: number; // basis points
}

export interface OnChainVersion {
    semver: string;
    rootHash: string;
    accuracy: number;     // already divided by 1e4
    timestamp: number;
    sessionId: string;
    weightsStored: boolean;
}
```

Remove old `OnChainModelEntry` type.

**Step 2: Verify build**

Run: `cd helix/dashboard && npm run build`
Expected: May have type errors in useModelRegistry.ts and my-models/page.tsx (expected, fixed in next tasks)

**Step 3: Commit**

```bash
git add helix/dashboard/src/lib/contracts.ts
git commit -m "feat: update ABI and types for ERC-721 model store"
```

---

### Task 4: Add weight encryption utilities

**Files:**
- Create: `helix/dashboard/src/lib/model-encryption.ts`

**Step 1: Write encryption/decryption module**

Uses Web Crypto API (AES-256-GCM). Key derivation: ask wallet to sign a deterministic message, SHA-256 the signature → AES key.

```typescript
/**
 * Model weight encryption for 0G Storage privacy.
 *
 * Flow:
 * 1. deriveModelKey(signMessage, tokenId) → CryptoKey
 *    - Asks wallet to sign "helix-model-key-{tokenId}"
 *    - SHA-256 of signature → 32-byte AES-GCM key
 * 2. encryptWeights(key, weightsJson) → { ciphertext: ArrayBuffer, iv: Uint8Array }
 * 3. decryptWeights(key, ciphertext, iv) → weightsJson string
 *
 * The ciphertext + iv are combined into a single blob before uploading to 0G.
 * Format: [12-byte IV][ciphertext]
 */

const SIGN_PREFIX = 'helix-model-key-';

export async function deriveModelKey(
    signMessage: (message: string) => Promise<string>,
    tokenId: number | bigint,
): Promise<CryptoKey> {
    const message = `${SIGN_PREFIX}${tokenId}`;
    const signature = await signMessage(message);

    // SHA-256 the signature to get 32 bytes
    const encoder = new TextEncoder();
    const sigBytes = encoder.encode(signature);
    const hashBuffer = await crypto.subtle.digest('SHA-256', sigBytes);

    return crypto.subtle.importKey(
        'raw',
        hashBuffer,
        { name: 'AES-GCM' },
        false,
        ['encrypt', 'decrypt'],
    );
}

export async function encryptWeights(
    key: CryptoKey,
    weightsJson: string,
): Promise<Uint8Array> {
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const encoder = new TextEncoder();
    const data = encoder.encode(weightsJson);

    const ciphertext = await crypto.subtle.encrypt(
        { name: 'AES-GCM', iv },
        key,
        data,
    );

    // Combine: [12-byte IV][ciphertext]
    const combined = new Uint8Array(12 + ciphertext.byteLength);
    combined.set(iv, 0);
    combined.set(new Uint8Array(ciphertext), 12);
    return combined;
}

export async function decryptWeights(
    key: CryptoKey,
    combined: Uint8Array,
): Promise<string> {
    const iv = combined.slice(0, 12);
    const ciphertext = combined.slice(12);

    const plaintext = await crypto.subtle.decrypt(
        { name: 'AES-GCM', iv },
        key,
        ciphertext,
    );

    const decoder = new TextDecoder();
    return decoder.decode(plaintext);
}

/** Convert a Uint8Array to hex string for display */
export function bytesToHex(bytes: Uint8Array): string {
    return Array.from(bytes)
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
}

/** Convert hex string back to Uint8Array */
export function hexToBytes(hex: string): Uint8Array {
    const bytes = new Uint8Array(hex.length / 2);
    for (let i = 0; i < hex.length; i += 2) {
        bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
    }
    return bytes;
}
```

**Step 2: Commit**

```bash
git add helix/dashboard/src/lib/model-encryption.ts
git commit -m "feat: AES-256-GCM weight encryption for 0G storage privacy"
```

---

### Task 5: Update `/api/store-on-0g` to accept encrypted payloads

**Files:**
- Modify: `helix/dashboard/src/app/api/store-on-0g/route.ts`

**Step 1: Add `encrypted` flag support**

The API should accept an optional `encrypted: true` + `encryptedPayload` (base64-encoded) field. When encrypted, it writes the raw encrypted bytes to the temp file instead of JSON.

When not encrypted, behavior is unchanged (backwards compatible).

```typescript
// Add to the body destructuring:
const { session_id, weights, accuracy, version, encrypted, encryptedPayload } = body;

// Before building the model artifact:
if (encrypted && encryptedPayload) {
    // Write encrypted binary directly
    const buffer = Buffer.from(encryptedPayload, 'base64');
    await writeFile(tmpPath, buffer);
} else {
    // Original JSON flow
    const modelArtifact = { ... };
    await writeFile(tmpPath, JSON.stringify(modelArtifact));
}
```

**Step 2: Verify the API still works for non-encrypted uploads**

Run: `cd helix/dashboard && npm run build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add helix/dashboard/src/app/api/store-on-0g/route.ts
git commit -m "feat: support encrypted weight uploads in store-on-0g API"
```

---

### Task 6: Rewrite `useModelRegistry` hook for ERC-721

**Files:**
- Modify: `helix/dashboard/src/hooks/useModelRegistry.ts`

**Step 1: Rewrite the hook**

The new hook should:
- Use `balanceOf` + `tokenOfOwnerByIndex` (ERC721Enumerable) to list user's models
- For each token: read `models(tokenId)` and `getVersions(tokenId)`
- Expose `createModel(slug, name, description)` write function
- Expose `addVersion(tokenId, semver, rootHash, accuracy, sessionId, weightsStored)` write function
- Return typed `OnChainModel[]` with nested `OnChainVersion[]`

Key structure:

```typescript
export interface ModelWithVersions {
    tokenId: number;
    slug: string;
    name: string;
    description: string;
    creator: string;
    createdAt: number;
    isPublic: boolean;
    inferenceFee: number;
    versions: OnChainVersion[];
}

export interface UseModelRegistryReturn {
    address: string | undefined;
    isConnected: boolean;
    isContractDeployed: boolean;
    models: ModelWithVersions[];
    isLoading: boolean;
    createModel: (params: { slug: string; name: string; description: string }) => void;
    addVersion: (params: { tokenId: number; semver: string; rootHash: string; accuracy: number; sessionId: string; weightsStored: boolean }) => void;
    isWritePending: boolean;
    isConfirming: boolean;
    writeError: Error | null;
    isSuccess: boolean;
    refetch: () => void;
}
```

Use wagmi's `useReadContracts` (multicall) to batch-read all model data in one RPC call. Read `balanceOf(address)` first, then for each index call `tokenOfOwnerByIndex` and `models(tokenId)` and `getVersions(tokenId)`.

**Step 2: Verify build**

Run: `cd helix/dashboard && npm run build`
Expected: Type errors in my-models/page.tsx (old API shape), fixed in Task 7

**Step 3: Commit**

```bash
git add helix/dashboard/src/hooks/useModelRegistry.ts
git commit -m "feat: rewrite useModelRegistry hook for ERC-721 contract"
```

---

### Task 7: Update `my-models` page for multi-model NFT registry

**Files:**
- Modify: `helix/dashboard/src/app/my-models/page.tsx`

**Step 1: Redesign the page**

Key changes from current single-model page to multi-model:
- **Model cards**: one card per NFT, showing model name, slug, creator, version count, best accuracy
- **Create Model modal**: slug + name + description form → calls `createModel`
- **Upload/Add Version modal**: select target model, version string, optional weights file → encrypts with wallet signature → uploads to 0G → calls `addVersion`
- **Version history**: nested under each model card (expandable)
- Remove `localStorage` dependency — everything comes from the contract
- Keep the `TrainingHistoryEntry` type for backwards-compat with train page, but merge into on-chain data

The page should still show a "Connect Wallet" notice when not connected, and "Contract Not Deployed" when the store isn't deployed.

Training history from localStorage is used as a migration bridge: if there are localStorage entries without on-chain models, show a "Migrate" button to create the model + version on-chain.

**Step 2: Verify build**

Run: `cd helix/dashboard && npm run build`
Expected: Build succeeds

**Step 3: Verify in browser**

Run: `cd helix/dashboard && npm run dev`
Open http://localhost:3000/my-models, verify:
- Empty state shows "Create Model" CTA
- Connect wallet, create a model, verify NFT minted
- Upload weights (encrypted), verify 0G upload succeeds
- Version appears in the model's history

**Step 4: Commit**

```bash
git add helix/dashboard/src/app/my-models/page.tsx
git commit -m "feat: multi-model NFT registry UI with encrypted weight uploads"
```

---

### Task 8: Wire training completion to model version recording

**Files:**
- Modify: `helix/dashboard/src/app/train/page.tsx`

**Step 1: After training completes, prompt user to save version**

When training finishes (and user clicks "Store on 0G"), the train page should:
1. Ask which model to add this version to (dropdown of user's NFTs), or create a new model
2. Encrypt weights with model-specific key
3. Upload encrypted weights to 0G
4. Call `addVersion` on the contract

This replaces the current localStorage-only flow. Keep localStorage as a fallback for when wallet isn't connected.

**Step 2: Verify build**

Run: `cd helix/dashboard && npm run build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add helix/dashboard/src/app/train/page.tsx
git commit -m "feat: wire training completion to on-chain model version recording"
```

---

### Task 9: Run full test suite and verify no regressions

**Step 1: Run contract tests**

Run: `cd helix/contracts && forge test -vvv`
Expected: All pass (including existing tests for other contracts)

**Step 2: Run dashboard build**

Run: `cd helix/dashboard && npm run build`
Expected: Clean build

**Step 3: Run dashboard lint**

Run: `cd helix/dashboard && npm run lint`
Expected: Clean or only pre-existing warnings

**Step 4: Final commit if any fixups needed**

```bash
git add -A && git commit -m "fix: address test/lint issues from model registry integration"
```
