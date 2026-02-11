# helix-node/src/data/ — Technical Review

## Overview

The `data/` module handles verified data loading for training: ensuring training data is available, its integrity is verified against on-chain commitments, and batches are loaded with Merkle proofs for verifiability. Total: ~2,500 lines across 4 files.

## Architecture

### Module Tree

```
data/
├── mod.rs                (~30 lines)   Re-exports
├── verified_loader.rs    (854 lines)   Merkle-verified batch loading
├── availability.rs       (825 lines)   Multi-source data availability checking
└── commitment_check.rs   (799 lines)   On-chain commitment verification + caching
```

### Key Types

| Type | File | Purpose |
|------|------|---------|
| `VerifiedDataLoader` | verified_loader.rs | Loads training batches with Merkle proofs |
| `MerkleTree` | verified_loader.rs | SHA-256 Merkle tree for data commitment |
| `DataAvailabilityChecker` | availability.rs | Multi-source health monitoring |
| `CommitmentVerifier` | commitment_check.rs | On-chain commitment verification + cache |

### Data Flow

```
1. Data source registered → availability check (ping/health)
2. Training data loaded → Merkle tree computed over batches
3. Root commitment verified against on-chain value
4. Per-batch Merkle proofs generated for individual verification
5. Trainer receives verified batch + proof
```

## Per-Module Analysis

### `verified_loader.rs` — Verified Data Loading (854 lines)

**What it does**: Loads training data with cryptographic integrity guarantees. Builds a SHA-256 Merkle tree over training batches, generates per-batch inclusion proofs, and verifies proofs before returning data to the trainer.

**Strengths**:
- **Correct Merkle tree implementation** (`verified_loader.rs:100-250`): Leaf hashing uses domain separation (0x00 prefix for leaves, 0x01 for internal nodes) preventing second-preimage attacks
- **Per-batch proofs** (`verified_loader.rs:300-400`): Each training batch has an inclusion proof that can be independently verified
- **Lazy proof generation** (`verified_loader.rs:380-420`): Proofs computed on demand, not upfront for entire dataset
- **Configurable batch size** (`verified_loader.rs:50-80`): Batch size adjustable for memory/compute tradeoffs

**Weaknesses**:
- **Merkle tree rebuilt on every load** (`verified_loader.rs:120-200`): No caching of the Merkle tree across calls.
  - **Impact**: O(n) tree construction on every batch fetch, even when data hasn't changed
  - **Fix**: Cache the tree and invalidate on data change. Store tree alongside data.
- **No streaming for large datasets** (`verified_loader.rs:120-200`): Entire dataset loaded into memory to build Merkle tree.
  - **Impact**: OOM for datasets larger than available memory
  - **Fix**: Stream-build the Merkle tree (compute leaf hashes as data is read from disk, build tree bottom-up without holding all data in memory)
- **Proof verification is O(log n) per batch** (`verified_loader.rs:420-500`): Correct asymptotically, but each verification involves log(n) SHA-256 calls.
  - **Impact**: For large datasets, per-batch verification adds latency to training loop
  - **Fix**: Batch-verify multiple proofs against the same root (share internal nodes). Or verify once at the start of a round, not per batch.

**Tests**: 9 tests covering tree construction, proof generation, proof verification, domain separation, edge cases (single batch, power-of-two, non-power-of-two). **Well tested.**

### `availability.rs` — Data Availability Checking (825 lines)

**What it does**: Monitors data source availability across multiple providers (local, IPFS, S3, HTTP). Tracks health status, supports redundancy levels (Single, Replicated, HighlyAvailable), and provides failover recommendations.

**Strengths**:
- **Multi-source monitoring** (`availability.rs:100-250`): Supports concurrent health checks across providers
- **Redundancy levels** (`availability.rs:50-90`): `Single` (1 source), `Replicated` (2+), `HighlyAvailable` (3+ across providers)
- **Health history tracking** (`availability.rs:300-400`): Rolling window of health check results for trend analysis
- **Failover recommendations** (`availability.rs:450-550`): Suggests alternative sources when primary fails

**Weaknesses**:
- **Health checks are simulated** (`availability.rs:200-250`): For remote sources (IPFS, S3, HTTP), health checks use simulated results rather than actual network calls.
  - **Impact**: Availability checker reports health without actually checking — misleading status
  - **Fix**: Implement real HTTP HEAD/GET requests for remote health checks. For IPFS, use `ipfs swarm peers` API. For S3, use `HEAD bucket`. Behind feature flags to avoid requiring credentials in tests.
- **No data re-replication** (`availability.rs:450-550`): Detects when redundancy drops below target but doesn't trigger re-replication.
  - **Impact**: Lost replica stays lost, redundancy degrades over time
  - **Fix**: On redundancy drop, trigger background copy to another provider (advisory for now, automatic for production)
- **No periodic health checks** (`availability.rs:200-250`): Health is checked on demand, not on a schedule.
  - **Impact**: Stale health data between checks
  - **Fix**: Add background health check loop with configurable interval (e.g. every 60 seconds)

**Tests**: 5 tests covering multi-source registration, health tracking, redundancy levels. **Adequate.**

### `commitment_check.rs` — Commitment Verification (799 lines)

**What it does**: Verifies that local training data matches the on-chain commitment. Computes SHA-256 hash of local data, compares against commitment stored on-chain (or from local cache). Includes caching to avoid redundant on-chain lookups.

**Strengths**:
- **Commitment caching** (`commitment_check.rs:200-300`): LRU cache for on-chain commitment lookups, avoids repeated RPC calls
- **Multiple verification modes** (`commitment_check.rs:80-130`): Full (re-hash all data), Cached (use cached commitment), Quick (hash first/last blocks only)
- **On-chain lookup integration** (`commitment_check.rs:300-400`): Designed to query smart contract for registered dataset commitments

**Weaknesses**:
- **On-chain lookup is stubbed** (`commitment_check.rs:320-380`): The `fetch_on_chain_commitment()` method returns a placeholder or cached value, doesn't actually call `SCClient`.
  - **Impact**: Commitment verification never actually verifies against the blockchain — it only checks against locally cached values, which could be wrong
  - **Fix**: Wire to `SCClient` or contract call for actual on-chain commitment retrieval. For demo, hard-code the expected commitment.
- **Quick mode is insecure** (`commitment_check.rs:120-130`): Only hashes first and last blocks, skipping the middle.
  - **Impact**: Modified data in the middle passes Quick verification
  - **Fix**: Quick mode should sample random blocks, not just endpoints. Or remove Quick mode and use Cached mode for speed.
- **Cache invalidation is time-based only** (`commitment_check.rs:250-280`): Cache entries expire after TTL, not on actual on-chain state change.
  - **Impact**: Stale commitment in cache could pass verification for data that should fail (e.g. after model re-registration)
  - **Fix**: Invalidate cache on new round start or on-chain event notification

**Tests**: 8 tests covering full/cached/quick verification, cache behavior, hash computation. **Good coverage.**

## Strengths Summary

1. **Correct Merkle tree**: Domain-separated leaf/internal hashing prevents second-preimage attacks — this is frequently implemented wrong in prototypes.
2. **Per-batch proofs**: Individual batch verification enables fine-grained integrity checking.
3. **Multi-source availability**: Redundancy-aware health monitoring across providers.
4. **Commitment caching**: LRU cache reduces on-chain RPC load.

## Weaknesses Summary (Prioritized)

### Critical

1. **On-chain commitment lookup is stubbed** (`commitment_check.rs:320-380`): Never actually verifies against blockchain.
   - **Fix**: Wire `SCClient.get_model_state()` for real on-chain commitment retrieval. Or for demo, hard-code commitment hash.

### High Priority

2. **Health checks are simulated** (`availability.rs:200-250`): Reports health without checking.
   - **Fix**: Real HTTP/IPFS health checks behind feature flags.

3. **Merkle tree rebuilt every time** (`verified_loader.rs:120-200`): Wasteful for unchanged data.
   - **Fix**: Cache the tree with data-change invalidation.

### Nice to Have

4. **Quick verification mode is insecure** (`commitment_check.rs:120-130`): Only checks endpoints.
5. **No streaming for large datasets** (`verified_loader.rs:120-200`): OOM for large datasets.
6. **No periodic health checks** (`availability.rs:200-250`): Stale health data.

## Testing Assessment

| Module | Tests | Coverage | Assessment |
|--------|-------|----------|------------|
| verified_loader.rs | 9 | High | Tree, proofs, verification, edge cases |
| availability.rs | 5 | Medium | Registration, health, redundancy |
| commitment_check.rs | 8 | Medium | All modes, cache, hashing |

**Total**: 22 tests. Good coverage for the module size.

**Missing tests**:
- Large dataset handling (memory limits)
- Concurrent data loading
- Cache eviction behavior
- Merkle proof verification with tampered data
- Availability failover end-to-end
- On-chain commitment retrieval (integration test with mock contract)

## Demo Readiness

| Feature | Status | Notes |
|---------|--------|-------|
| Merkle-verified loading | Ready | Correct tree + proofs |
| Data availability check | Partial | Simulated health checks |
| Commitment verification | Partial | Stub on-chain lookup |
| Batch loading | Ready | Configurable batch sizes |
| Multi-source support | Partial | Local works, remote simulated |

**Demo target**: Training data loaded with verifiable integrity. **Ready for demo with local data sources.**

## Summary

### Health Score: **C+** (60/100)

The data module has a **correctly implemented Merkle tree** with domain-separated hashing — the cryptographic core is sound. Per-batch inclusion proofs and multi-mode commitment verification show thoughtful design. However, the module is undermined by two significant stubs: **on-chain commitment lookup never calls the blockchain** and **data availability health checks are simulated**. This means the verification chain has a gap: data integrity is checked against a cached/placeholder commitment, not the actual on-chain value. For demo purposes, local data loading with Merkle proofs works correctly. For production, the on-chain integration must be completed.
