# data/ - Technical Review

## Overview

The `data` module provides comprehensive data management infrastructure for HELIX's decentralized ML training. It handles everything from dataset loading and Merkle commitments to federated sharding and multi-source data fetching. This module is critical for proving that training used specific, committed data.

## Role in helix-core

This module enables verifiable data handling:
- **Commitment**: Cryptographic proof of dataset contents via Merkle trees
- **Membership**: Prove specific samples were used in training batches
- **Provenance**: Track data origin and transformations
- **Distribution**: Shard data across workers for federated learning
- **Fetching**: Retrieve data from decentralized sources (IPFS, Filecoin, S3)

## Module Structure

```
data/
├── mod.rs                  # Re-exports all data types
├── dataset.rs              # Sample, Batch, InMemoryDataset
├── merkle.rs               # Production Merkle tree implementation
├── commitment.rs           # Dataset and batch commitments
├── membership_proof.rs     # Sample membership proofs for batches
├── provenance.rs           # Data origin and transformation tracking
├── sharding.rs             # Federated learning data distribution
├── shuffling.rs            # Deterministic verifiable shuffling
├── streaming.rs            # Streaming batch verification
├── registry.rs             # Model registry
├── dataset_registry.rs     # Dataset commitment registry
├── model.rs                # Model serialization
├── ipfs.rs                 # Legacy IPFS client
└── sources/
    ├── mod.rs              # Multi-source data fetching
    ├── ipfs.rs             # IPFS data source
    ├── filecoin.rs         # Filecoin data source
    └── s3.rs               # S3 data source
```

## Detailed Analysis

### merkle.rs

**Purpose**: Production-grade Merkle tree for cryptographic commitment to datasets.

**Key Types**:
```rust
struct Hash([u8; 32]);              // SHA-256 hash
struct MerkleTree { nodes, leaves, root, height }
struct MerkleProof { leaf_index, leaf_hash, path, root }
struct MultiProof { leaf_indices, proof_nodes, root }  // Deduplicated
```

**Key Features**:

1. **Domain Separation** (lines 141-154):
   ```rust
   fn hash_leaf(data) -> prefix 0x00 + SHA256(data)
   fn hash_nodes(l, r) -> prefix 0x01 + SHA256(l || r)
   ```
   Prevents second-preimage attacks.

2. **Streaming Construction**:
   ```rust
   struct StreamingMerkleBuilder { ... }
   // Processes leaves in chunks, O(1) memory overhead
   ```

3. **Multi-Proof Optimization**:
   - Deduplicates shared siblings
   - Single proof for multiple leaves
   - Significant size reduction for batch proofs

4. **Parallel Construction**:
   ```rust
   struct ParallelMerkleBuilder { ... }
   // Uses rayon for multi-threaded tree building
   ```

**Verification**:
```rust
impl MerkleProof {
    fn verify<H: MerkleHasher>(&self, hasher: &H) -> bool {
        // Walk path from leaf to root
        // Compare computed root to expected
    }
}
```

**Complexity**:
| Operation | Time | Space |
|-----------|------|-------|
| Build tree | O(n) | O(n) |
| Generate proof | O(log n) | O(log n) |
| Verify proof | O(log n) | O(1) |
| Multi-proof verify | O(k log n) | O(k + log n) |

**Strengths**:
- Domain separation is security-critical and correctly implemented
- Streaming builder handles arbitrarily large datasets
- `BinarySerializable` implementation for on-chain storage

**Weaknesses**:
- Default builds entire tree in memory (should default to streaming)
- No support for tree updates (append-only)

### commitment.rs

**Purpose**: Higher-level commitment abstractions on top of Merkle trees.

**Key Types**:
```rust
struct SampleCommitment { sample_hash, index, dataset_root }
struct BatchCommitment { batch_hash, sample_commitments, dataset_root }
struct DatasetCommitment { root_hash, sample_count, metadata }

struct CommitmentManager {
    // Manages dataset → commitment mappings
    // Generates batch commitments on demand
}
```

**Workflow**:
1. Load dataset → compute Merkle tree → store `DatasetCommitment`
2. For each batch → generate `BatchCommitment` with sample proofs
3. Verifier checks batch membership against dataset root

**Strengths**:
- Clean separation of concerns
- Efficient batch commitment generation

### membership_proof.rs

**Purpose**: Prove that specific samples were used in training batches.

**Key Types**:
```rust
struct SampleMembershipProof {
    sample_index: usize,
    sample_hash: Hash,
    merkle_proof: MerkleProof,
}

struct BatchMembershipProof {
    batch_index: usize,
    sample_proofs: Vec<SampleMembershipProof>,
}

struct AggregatedBatchProof {
    multi_proof: MultiProof,
    batch_indices: Vec<usize>,
}
```

**Verification Stats**:
```rust
struct VerificationStats {
    proofs_verified: usize,
    proofs_failed: usize,
    verification_time_ms: u64,
}
```

**Strengths**:
- Aggregated proofs for efficiency
- Statistics tracking for monitoring

### sharding.rs

**Purpose**: Partition data across workers for federated learning.

**Sharding Strategies**:
```rust
enum ShardingStrategy {
    Random { seed },          // Uniform random
    RoundRobin,               // Sequential assignment
    HashBased,                // Hash(sample_id) % num_shards
    IID { seed },             // Stratified by label (IID distribution)
    NonIID { classes_per_shard, seed },  // Label skew
    Dirichlet { alpha, seed },           // Concentration-based imbalance
}
```

**Dirichlet Distribution**: Used to simulate realistic federated scenarios where workers have non-uniform label distributions. Lower α = more imbalanced.

**Worker Management**:
```rust
struct WorkerInfo {
    id: WorkerId,
    memory_bytes, cpu_cores, has_gpu, gpu_memory_bytes,
    bandwidth_bps, zone, status, last_heartbeat, processing_rate,
}

enum WorkerStatus { Active, Busy, Paused, Failed, Draining }
```

**Shard Registry**:
```rust
struct ShardRegistry {
    shards: HashMap<ShardId, ShardMetadata>,
    workers: HashMap<WorkerId, WorkerInfo>,
    assignments: HashMap<ShardId, Vec<WorkerId>>,  // For replication
    progress: HashMap<ShardId, ShardProgress>,
}
```

**Features**:
- Locality-aware assignment (same zone preference)
- Automatic rebalancing on worker failure
- Progress tracking with checkpoints
- Content hash verification

**Strengths**:
- Multiple strategies for different research scenarios
- Worker capabilities modeling (GPU, memory)
- Fault tolerance through replication

**Weaknesses**:
- No actual distributed coordination (uses local HashMap)
- Rebalancing algorithm is naive

### provenance.rs

**Purpose**: Track data origin and transformation chain.

**Key Types**:
```rust
struct ProvenanceRecord {
    id: ProvenanceId,
    origin: DataOrigin,
    transformations: Vec<DataTransformation>,
    attestations: Vec<Attestation>,
    timestamps: ProvenanceTimestamps,
}

enum DataOrigin {
    Synthetic { generator, seed, parameters },
    PublicDataset { name, url, license, citation },
    PrivateDataset { owner_id, description },
    Federation { contributing_parties, aggregation_method },
}

enum TransformationType {
    Normalization { method, parameters },
    Augmentation { technique, probability },
    Filtering { criteria },
    Aggregation { method, sources },
    PrivacyPreserving { technique, epsilon },
}
```

**Attestations**:
```rust
struct Attestation {
    attester: String,
    attestation_type: AttestationType,
    signature: Vec<u8>,
    timestamp: u64,
    metadata: HashMap<String, String>,
}

enum AttestationType {
    DataIntegrity,
    LicenseCompliance,
    PrivacyCompliance,
    QualityAssurance,
    ComputationVerification,
}
```

**Strengths**:
- Comprehensive origin tracking
- Supports privacy-preserving transformations (ε-DP)
- Attestation chain for audit trail

**Weaknesses**:
- Signatures not cryptographically verified (placeholder)
- No integration with on-chain attestations

### shuffling.rs

**Purpose**: Deterministic, verifiable shuffling for training order.

**Key Components**:
```rust
struct DeterministicShuffler {
    seed: ShuffleSeed,
    commitment: SeedCommitment,
}

struct ShuffleSeed([u8; 32]);
struct SeedCommitment(Hash);  // Commit to seed before revealing

struct ShuffleVerifier {
    // Verify that shuffle was performed correctly given seed
}
```

**Commit-Reveal Protocol**:
1. Shuffler commits to seed hash
2. Training proceeds with shuffled order
3. Seed revealed for verification
4. Verifier recomputes shuffle and checks

**Strengths**:
- Prevents shuffle manipulation after seeing data
- Deterministic replay for verification

### sources/

**Purpose**: Fetch data from decentralized storage.

**Supported Sources**:
```rust
trait DataSource {
    async fn fetch(&self, id: &str, options: FetchOptions) -> DataSourceResult<DataChunk>;
    async fn exists(&self, id: &str) -> bool;
    async fn metadata(&self, id: &str) -> DataSourceResult<ResourceMetadata>;
}

trait WritableDataSource: DataSource {
    async fn upload(&self, data: &[u8], options: UploadOptions) -> DataSourceResult<String>;
    async fn delete(&self, id: &str) -> DataSourceResult<()>;
}
```

**Implementations**:
| Source | Read | Write | Notes |
|--------|------|-------|-------|
| `IpfsDataSource` | ✅ | ✅ | Via HTTP gateway |
| `FilecoinDataSource` | ✅ | ✅ | Via Lotus API |
| `S3DataSource` | ✅ | ✅ | Standard S3 compatible |

**Multi-Source Fetcher**:
```rust
struct MultiSourceFetcher {
    sources: Vec<Box<dyn DataSource>>,
    fallback_behavior: FallbackBehavior,
    cache: DataCache,
}

enum FallbackBehavior {
    Sequential,    // Try sources in order
    Parallel,      // Race all sources
    Fastest,       // Use historically fastest
}
```

**Caching**:
```rust
struct DataCache {
    // LRU cache with configurable size
    // TTL support for stale data
}
```

**Strengths**:
- Async/await for concurrent fetching
- Fallback strategies for reliability
- Built-in caching

**Weaknesses**:
- IPFS and Filecoin implementations are placeholder (HTTP only)
- No content verification on fetch (should verify against expected hash)

## Strengths Summary

1. **Security-First Merkle Trees**: Domain separation prevents attacks.

2. **Flexible Sharding**: Multiple strategies for research and production.

3. **Complete Provenance**: Full audit trail from origin through transformations.

4. **Multi-Source Abstraction**: Unified interface for decentralized storage.

5. **Streaming Support**: Handles large datasets without memory issues.

## Weaknesses Summary

1. **Placeholder Implementations**: IPFS/Filecoin lack native protocol support.

2. **No Distributed Coordination**: ShardRegistry is in-memory only.

3. **Missing Content Verification**: Fetched data should verify against commitment.

4. **Legacy Code**: `ipfs.rs` duplicates `sources/ipfs.rs`.

## Demo Impact

For the ETHDenver demo:

**Ready**:
- Merkle tree commitment and verification
- Batch membership proofs
- Dataset shuffling
- Basic sharding

**Needs Work**:
| Feature | Current State | Needed |
|---------|---------------|--------|
| Data sources | Mock/placeholder | Real IPFS fetch for demo |
| Provenance | Types exist | Attestation display in UI |
| Progress tracking | In-memory | Persist across demo restarts |

## Recommendations

1. **Verify fetched content**: Hash check against expected commitment.

2. **Remove legacy ipfs.rs**: Redirect to sources/ipfs.rs.

3. **Add integration tests**: End-to-end sharding → training → verification.

4. **Implement native IPFS**: Use `ipfs-api` crate for real protocol support.

5. **Add shard streaming**: Large shards should stream, not load fully.
