# ERC-721 Model Registry with Weight Privacy

## Overview

Replace `HelixModelStore.sol` with an ERC-721-based model registry. Each model is an NFT. The token holder owns the model, controls access to weights, and can transfer/sell the model via standard NFT mechanics.

## Weight Privacy

0G storage is content-addressed and public. Privacy is enforced via client-side encryption:

1. Dashboard generates a random AES-256-GCM key per model
2. Weights encrypted before upload to 0G
3. On-chain rootHash points to ciphertext — useless without the key
4. Key derived deterministically from wallet signature: `personal_sign("helix-model-key-{tokenId}")` → SHA-256 → AES key
5. Only the wallet owner can produce the signature → only they can decrypt

## Smart Contract

```solidity
contract HelixModelStore is ERC721Enumerable {
    struct Model {
        string slug;           // user-chosen ID ("mnist-v1")
        string name;           // display name
        string description;
        address creator;       // original creator (immutable after transfer)
        uint40 createdAt;
        bool isPublic;         // future: paid inference by non-owners
        uint16 inferenceFee;   // basis points (500 = 5%), future use
    }

    struct Version {
        string semver;
        string rootHash;       // 0G root hash (encrypted weights), "" if not stored
        uint96 accuracy;       // scaled 1e4
        uint40 timestamp;
        string sessionId;
        bool weightsStored;
    }

    mapping(uint256 => Model) public models;
    mapping(uint256 => Version[]) public versions;
    mapping(bytes32 => uint256) public slugToToken;
    mapping(uint256 => mapping(address => bool)) public accessGranted;
}
```

### Functions

- `createModel(slug, name, description)` → mint NFT, return tokenId
- `addVersion(tokenId, semver, rootHash, accuracy, sessionId, weightsStored)` → owner only
- `setPublic(tokenId, isPublic)` → owner only, future
- `setInferenceFee(tokenId, basisPoints)` → owner only, future, max 5000 (50%)
- `grantAccess(tokenId, address)` / `revokeAccess(tokenId, address)` → owner only, future
- `hasModelAccess(tokenId, address)` → view, returns true if owner or granted

### Reads

- `getModel(tokenId)` → Model struct
- `getVersionCount(tokenId)` → uint256
- `getVersion(tokenId, index)` → Version struct
- `getVersions(tokenId)` → Version[] (all)
- `getModelBySlug(slug)` → tokenId
- Standard ERC721Enumerable: `tokenOfOwnerByIndex`, `balanceOf`, etc.

## Dashboard Changes

1. **`useModelRegistry` hook**: read from new ERC-721 contract
2. **`my-models` page**: list user's NFTs, show versions per model, create model + add versions
3. **Encryption layer**: AES-256-GCM encrypt/decrypt via wallet-derived key
4. **`/api/store-on-0g`**: accept optional encryption, return rootHash of encrypted data
5. **Deploy script**: update `DeployModelStore.s.sol`

## Now vs Future

| Now | Future |
|-----|--------|
| ERC-721 model + version structs | Inference fee collection (basis points) |
| Weight encryption before 0G upload | Public model marketplace |
| Owner-only decryption | `grantAccess` for paid users |
| Dashboard integration | Buy/sell with key handoff |
| Deploy script | ERC-2981 royalties |
