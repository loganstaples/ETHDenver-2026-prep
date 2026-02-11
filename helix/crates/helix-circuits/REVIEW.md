# helix-circuits -- Code Review

**Reviewer**: Claude Opus 4.6 automated audit
**Date**: 2026-02-11 (updated; original 2026-02-10)
**Scope**: Every file in `crates/helix-circuits/` (~45 Rust files, ~30,700 LOC, 329+ tests)
**Health Score**: **C+ (68%)** -- improved from 62% after copy constraint fixes, dead code removal, and PI[7] constraint addition

---

## 1. Overview

`helix-circuits` is the ZK circuit layer for HELIX -- it defines Halo2 (PSE fork, KZG on BN254) circuits that prove ML training steps are performed correctly. It is the most architecturally ambitious crate in the workspace, containing:

- **Approximate arithmetic** circuits with tracked error bounds and copy constraints
- **Lookup tables** for non-linear activations (ReLU, GELU, sigmoid, softmax)
- **ML-specific circuits**: linear layers, attention, layer norm, full transformers, gradient verification, proof aggregation
- **IVC** (Incrementally Verifiable Computation) via Nova-style folding
- **EVM verifier** format specification for on-chain proof verification
- **Quantization** circuits (INT4/INT8) with calibration
- **Caching** infrastructure with LRU/LFU eviction
- **Parameter management** (SRS, proving/verification keys)

### Role in HELIX

This crate sits between `helix-core` (types, tensors) and `helix-prover` (proof generation). It defines the constraint systems that the prover instantiates, and the proof format that the Solidity verifier accepts. It is the cryptographic heart of the protocol.

---

## 2. Architecture

### Module Tree

```
helix-circuits/
├── src/
│   ├── lib.rs                  (~140 lines -- re-exports)
│   ├── benchmark.rs            (383 lines -- MockProver benchmarking)
│   ├── cache/
│   │   ├── mod.rs              (406 lines -- composite cache facade with TTL, LFU)
│   │   └── structure_cache.rs  (592 lines -- structure/witness/table/key caches)
│   ├── ivc.rs                  (2098 lines -- Nova-style IVC/folding)
│   ├── tests.rs                (743 lines -- integration tests)
│   ├── approximate/
│   │   ├── mod.rs              (83 lines)
│   │   ├── activation.rs       (134 lines -- ReLU chip, single-region, copy constraints)
│   │   ├── bounded_add.rs      (91 lines -- error-tracked addition, copy constraints)
│   │   ├── bounded_mul.rs      (126 lines -- error-tracked multiplication, 8 copy constraints)
│   │   ├── bounded_matmul.rs   (245 lines -- error-tracked matmul, cross-region copies)
│   │   ├── error_accumulation.rs (612 lines -- error budget tracking)
│   │   └── quantization.rs     (1068 lines -- quantized ops + lookup tables)
│   ├── commitment/
│   │   ├── mod.rs              (2 lines)
│   │   ├── model_commit.rs     (141 lines -- SHA-256 model hash)
│   │   └── state_transition.rs (76 lines -- state hash verification)
│   ├── gadgets/
│   │   ├── mod.rs              (15 lines)
│   │   ├── arithmetic.rs       (83 lines -- add/mul gates, enable_equality)
│   │   ├── lookup.rs           (685 lines -- generic plookup)
│   │   ├── poseidon.rs         (659 lines -- Poseidon hash, in-circuit)
│   │   └── range.rs            (80 lines -- range check via lookup)
│   ├── lookup/
│   │   ├── mod.rs              (83 lines)
│   │   ├── relu.rs             (710 lines -- ReLU/LeakyReLU/ReLU6/PReLU)
│   │   ├── gelu.rs             (698 lines -- GELU/FastGELU/derivative)
│   │   ├── softmax.rs          (893 lines -- sigmoid/tanh/softmax exp)
│   │   └── table.rs            (992 lines -- PlookupTable infrastructure)
│   ├── ml/
│   │   ├── mod.rs              (19 lines)
│   │   ├── aggregation.rs      (724 lines -- gradient aggregation)
│   │   ├── attention.rs        (1116 lines -- multi-head attention circuit)
│   │   ├── batch.rs            (295 lines -- batch proving)
│   │   ├── config.rs           (781 lines -- transformer configs)
│   │   ├── embedding.rs        (736 lines -- embedding/output layers)
│   │   ├── gradient.rs         (541 lines -- gradient verification)
│   │   ├── layer_norm.rs       (780 lines -- layer normalization)
│   │   ├── linear_layer.rs     (496 lines -- linear layer circuit)
│   │   ├── positional.rs       (812 lines -- positional encoding)
│   │   ├── proof_aggregation.rs (794 lines -- SHPLONK aggregation)
│   │   ├── softmax.rs          (484 lines -- softmax circuit)
│   │   ├── training_step_v2.rs (2089 lines -- THE main training circuit)
│   │   └── transformer.rs      (1975 lines -- full transformer block)
│   ├── params/
│   │   ├── mod.rs              (130 lines)
│   │   ├── keys.rs             (1287 lines -- proving/verification keys)
│   │   └── setup.rs            (1101 lines -- SRS/powers of tau)
│   ├── quantization/
│   │   ├── mod.rs              (92 lines)
│   │   ├── calibration.rs      (785 lines -- calibration verification)
│   │   ├── int4.rs             (799 lines -- INT4 quantization circuit)
│   │   └── int8.rs             (1003 lines -- INT8 quantization circuit)
│   └── verifier/
│       ├── mod.rs              (21 lines)
│       ├── evm.rs              (1511 lines -- EVM proof format)
│       ├── format_spec.rs      (856 lines -- proof serialization)
│       ├── native.rs           (283 lines -- native Rust verifier)
│       └── transcript.rs       (414 lines -- transcript utilities)
└── benches/
    └── circuit_benchmarks.rs   (767 lines -- criterion benchmarks)
```

### Key Types and Traits

| Type | Module | Purpose |
|------|--------|---------|
| `MLTrainingStepV2Circuit` | ml/training_step_v2 | **Primary circuit** -- proves one gradient step |
| `MLTrainingStepV2Witness` | ml/training_step_v2 | Witness containing weights, gradients, hashes |
| `IVCStepCircuit` | ivc | Single IVC step with Poseidon state transitions |
| `IVCFoldingCircuit` | ivc | Nova-style folding of two accumulators |
| `IVCMultiStepCircuit` | ivc | N sequential steps in one proof |
| `TransformerCircuit` | ml/transformer | Full transformer block verification |
| `EvmProof` | verifier/evm | Serialized proof for Solidity verifier |
| `EvmPublicInputsArray` | verifier/evm | 8 public inputs for contract |
| `SHPLONKAggregationCircuit` | ml/proof_aggregation | Proof aggregation circuit |
| `HelixSRS` | params/setup | Structured Reference String wrapper |
| `HelixProvingKey` | params/keys | Extended proving key with metadata |

### Data Flow

```
Input Weights + Training Data
    -> compute_witness_v2() [native Rust computation]
    -> MLTrainingStepV2Witness [all intermediate values]
    -> MLTrainingStepV2Circuit::synthesize() [Halo2 constraint assignment]
    -> proof bytes (via helix-prover)
    -> serialize_proof_for_evm() [format for Solidity]
    -> HelixCoordinatorV2.submitProof() [on-chain]
```

### Dependencies

- **External**: `halo2_proofs` (PSE fork, pinned to rev `198e9ae3`), `halo2curves 0.7.0`, `sha2`, `sha3`, `hex`, `subtle`, `rand_core`
- **Internal**: `helix-core` (types, tensors, error tracking)

---

## 3. Per-Module Analysis

### 3.1 `ml/training_step_v2.rs` (2089 lines) -- **CRITICAL PATH**

**Purpose**: The main circuit that proves a single training step (forward pass, loss, backward pass, weight update) for a 2-layer MLP.

**Key Components**:
- `compute_witness_v2()`: Computes all intermediate values outside the circuit
- `compute_state_hash_v2()`: Poseidon hash of weights, split into lo/hi 128-bit halves
- `MLTrainingStepV2Circuit`: The `Circuit<Fr>` implementation
- 8 public inputs: `[oldHashLo, oldHashHi, newHashLo, newHashHi, loss, errorBound, stepNumber, errorChecksum]`

**Algorithm**: Forward (matmul -> ReLU -> matmul -> loss), backward (gradient via chain rule), weight update (SGD), state hash verification.

**Strengths**:
- Real constraint system that actually verifies training computations
- ReLU via lookup table (efficient)
- Freivalds matmul verification (probabilistic, reduces constraints)
- Proper public input binding with `constrain_instance`
- Error bound tracking through every operation
- Error checksum (PI[7]) now constrained via 3 in-circuit Poseidon hashes

**Weaknesses**:
- **MEDIUM**: PI[5] (error_bound) and PI[6] (step_number) are still witness-only -- bound to instance column but no in-circuit computation constrains their correctness. A malicious prover can claim any error bound or step number.
- **MEDIUM**: ReLU negative detection uses `bytes[31] >= 0x19` which is a heuristic. Values with MSB in [0x19, 0x2F] are misclassified. In practice, training weights are small so this rarely triggers.
- Only supports 2-layer MLP (d_in, d_hid, d_out). No conv layers, no multi-layer support.

### 3.2 `ivc.rs` (2098 lines) -- **IVC/Folding**

**Purpose**: Nova-style IVC enabling multi-step proof compression.

**Key Components**:
- `IVCAccumulator`: Tracks state, error terms, witness/error vectors, Poseidon commitments
- `fold_accumulators()`: Linear combination Z' = Z1 + r*Z2 with cross-terms
- `IVCStepCircuit`: Verifies one step + optional fold, constrains state via Poseidon
- `IVCFoldingCircuit`: Verifies fold of two accumulators (witness/error commitment verification)
- `IVCMultiStepCircuit`: N steps in one proof

**Strengths**:
- Actually implements Nova-style folding with cross-term computation
- In-circuit Poseidon hash for state transitions (synthesized, not just declared)
- Element-wise witness/error vector folding verified in the folding circuit
- Real KZG proofs tested (`test_folding_circuit_real_proof`, `test_multi_step_five_step_real_proof`)
- 22 comprehensive tests including soundness tests (reject wrong state, wrong challenge)

**Weaknesses**:
- Cross-term computation is PLONKish-adapted (not standard Nova R1CS form). Acceptable architectural choice.
- `MAX_WITNESS_SIZE = 64` limits folding to 64-element witness vectors.
- `commit_vector()` uses sequential Poseidon pairs -- O(n) hashes. Merkle tree would be O(log n).
- Poseidon round constants are non-standard (SHA-256 derived, not Grain LFSR).

### 3.3 `gadgets/` (1,522 lines) -- **Circuit Primitives**

Now contains only 4 active, production-quality gadgets:

- `arithmetic.rs` (83 lines): `ArithmeticChip` with add/mul gates and `enable_equality` on all 3 advice columns. The most-imported gadget in the crate.
- `poseidon.rs` (659 lines): Full Poseidon implementation (width 3, rate 2, 8 full + 57 partial rounds, x^5 S-box). Both native and in-circuit synthesis. ~764 rows per hash. Used by IVC, training_step_v2, and proof_aggregation.
- `range.rs` (80 lines): Lookup-based range check constraining values to [0, RANGE). Correct `complex_selector` usage.
- `lookup.rs` (685 lines): Generic two-column lookup, ReLU table, exp table with padded loading.

Dead gadgets (freivalds, comparison, swap, builder) were deleted in production hardening Stage 3. Module is now clean with zero dead code.

### 3.4 `approximate/` (2,359 lines) -- **Error-Bounded Arithmetic**

**Purpose**: Circuits that track numerical error through arithmetic operations.

**Key Components**:
- `BoundedAddChip`: Single-region with copy constraint binding err_c across arithmetic and range check
- `BoundedMulChip`: Single-region with 8 copy constraints (val_a, val_b, err_a, err_b, term1, term2, term3, sum1)
- `BoundedMatMulChip`: Iterative dot product with cross-region copy constraints for running values
- `ReLUChip` (activation.rs): Sound decomposition (x + neg = y, y * neg = 0, range_check(y), range_check(neg)) with 6 copy constraints
- `ErrorAccumulationChip`: Tracks error budget through computation graph
- `QuantizationChip`: INT8/INT4 quantization with lookup tables

**Strengths**:
- **Copy constraints now properly bind shared values** (fixed from the original multi-region vulnerability). Each chip uses a single region with explicit `constrain_equal` calls.
- Proper error algebra: addition errors add, multiplication errors follow product rule with second-order term
- Range checks enforce error bounds within budget
- Good test coverage (positive and negative tests)

**Remaining Weaknesses**:
- `error_accumulation.rs` uses separate regions for mul error propagation without copy constraints (unlike the fixed bounded_mul.rs)
- ReLU and Div error in `ErrorAccumulationCircuit` are unconstrained (witness-only)
- Multiplication error formula uses raw field elements, not absolute values (unsound for negative values in a prime field)

### 3.5 `verifier/` (3,085 lines) -- **EVM Proof Format**

**Purpose**: EVM-compatible proof format and verification utilities.

**Strengths**:
- Correct SHPLONK layout: 3 advice commits + W + W' = 5 G1 points = 320 bytes
- Proper compressed G1 decompression for PSE halo2curves encoding
- Keccak256 transcript matching Solidity challenge derivation
- ~45 tests including pipeline integration tests

**Weaknesses**:
- **HIGH**: Generated Solidity verifier (`SolidityGenerator`) implements simplified KZG pairing check, not real SHPLONK verification. Cannot verify actual Halo2 proofs. Structural placeholder.
- **MEDIUM**: `SCALAR_FIELD_ORDER` and `BASE_FIELD_PRIME` constants are swapped in format_spec.rs (currently unused, but dangerous if referenced).
- **MEDIUM**: `native.rs::encode_commitment` truncates 128-bit values to 64 bits.

### 3.6 `cache/` (998 lines) -- **Circuit Caching**

Cache module ambiguity (cache.rs vs cache/mod.rs) has been **RESOLVED** -- deleted `cache.rs`, kept `cache/mod.rs` + `structure_cache.rs`. The active implementation provides:

- `CircuitCache`: Composite facade wrapping StructureCache + WitnessCache + TableCache
- 4 eviction policies: LRU, LFU, FIFO, Random
- TTL support and memory-aware caching
- Global singleton via `OnceLock`

### 3.7 `ml/transformer.rs` (1975 lines) -- **Full Transformer Block**

**Strengths**: Complete implementation with LayerNorm, MultiHeadAttention, Residual, FFN. 21 tests.

**Critical Weakness**: Verification methods use self-equality checks (compare output to itself) rather than actual constraint verification. `verify_layer_norm`, `verify_attention`, and `verify_ffn` prove almost nothing beyond residual connection structure.

### 3.8 `commitment/` (219 lines)

- `model_commit.rs`: Native SHA-256 hash with lo/hi split matching contract format. Working.
- `state_transition.rs`: Weight update via bounded arithmetic gadgets. Working.
- `gradient_commit.rs`: **Deleted** (was 16-line stub).

**Critical weakness**: `verify_path()` always returns `Ok(())` -- no Merkle proof verification. No in-circuit SHA-256 constraints.

### 3.9 `lookup/` (3,376 lines) -- **Activation Lookup Tables**

Rich library of activation function tables (ReLU, GELU, sigmoid, tanh, softmax) with error bound tracking, derivative tables for backward pass, and batch deduplication.

**Weakness**: `compute_field` in relu.rs uses `bytes[31] & 0x80 != 0` for negativity check (always false for BN254). Only affects native witness computation, not in-circuit lookup correctness.

### 3.10 `params/` (2,518 lines) -- **SRS and Key Management**

Real SRS generation via `ParamsKZG::setup()`, two-tier caching (memory + disk), real keygen via `RealKeyBundle::generate_real()`. Optimization module (1,030 lines of simulated optimization) has been **deleted**.

**Weakness**: `HelixSRS` and `HelixProvingKey`/`HelixVerificationKey` are metadata-only wrappers, not actual cryptographic data containers. Confusing dual API with `RealKeyBundle`.

### 3.11 `quantization/` (2,679 lines) -- **INT8/INT4 Circuits**

One of the strongest modules. Complete gate implementations for quantize/dequantize/multiply/add/requantize. Lookup-based range checks. Mixed-precision INT4xINT8 support. Three calibration methods (MinMax, Histogram, Entropy). Well-tested.

---

## 4. Strengths

### S1: Comprehensive ML Circuit Library
Circuits for every major transformer operation: attention, FFN, layer norm, embeddings, positional encoding, softmax, GELU. Impressive scope for a hackathon project. **Ref**: `ml/transformer.rs`, `ml/attention.rs`

### S2: Real IVC with Verified Folding
Goes beyond scaffolding -- folds witness/error vectors element-wise, computes cross-terms, verifies commitments via in-circuit Poseidon. Tested with real KZG proofs. **Ref**: `ivc.rs:227-289` (fold), `ivc.rs:825-1148` (folding circuit)

### S3: EVM Proof Format Specification
Complete specification of proof-to-Solidity mapping with serialization roundtrips, point-on-curve validation, and hash pair computation matching the contract. **Ref**: `verifier/format_spec.rs`, `verifier/evm.rs`

### S4: Error Bound Algebra with Copy Constraints
Tracking numerical error through arithmetic is novel for ZK-ML. Copy constraints now properly prevent a malicious prover from bypassing error bounds. **Ref**: `approximate/bounded_mul.rs:108-117` (8 copy constraints)

### S5: Extensive Test Coverage
329+ test functions, including real KZG proof tests (not just MockProver). Soundness tests verify wrong inputs are rejected. 5 adversarial PI tests. **Ref**: `ivc.rs:1509-1531`, `tests.rs:210-250`

### S6: SHPLONK Aggregation with Fiat-Shamir
Real aggregation circuit (not sequential loop) with PI chaining, Poseidon-based Fiat-Shamir challenge, and RLC commitment. 11 tests including negative tests. **Ref**: `ml/proof_aggregation.rs`

### S7: Clean Dead Code Removal
~5,700 lines of dead code removed: optimization/ (2,573), profiling/ (3,028), non-functional gadgets (freivalds, comparison, swap, builder), gradient_commit stub, duplicate cache.rs. Module is lean.

---

## 5. Weaknesses

### W1: HIGH -- Transformer Verification Is Self-Equality Facade
**Location**: `ml/transformer.rs:414-652` (verify_layer_norm, verify_attention, verify_ffn)
**Impact**: The transformer circuit accepts ANY witness values for layer norm, most attention positions, and most FFN positions. It proves only residual connection structure.
**Fix**: Reuse `LayerNormChip` for verify_layer_norm, `SoftmaxChip` for attention weights, and GELU lookup for FFN activation. Remove `.min(2)`/`.min(4)` bounds to verify all positions.

### W2: HIGH -- Generated Solidity Verifier Is a Placeholder
**Location**: `verifier/evm.rs:424-487` (generate_full_verify_body)
**Impact**: Cannot verify actual Halo2 SHPLONK proofs. The pairing equation is a simplified single-polynomial check.
**Fix**: Use `pse/halo2-solidity-verifier` to generate real verifier from proving key. Or use the working `Halo2Verifier.sol` already in the contracts/ directory.

### W3: MEDIUM -- PI[5] Error Bound and PI[6] Step Number Are Witness-Only
**Location**: `ml/training_step_v2.rs:1012-1014` (verify_error_bound call site)
**Impact**: A prover can claim an arbitrarily low error bound or any step number. PI[7] error checksum is now constrained via Poseidon, but PI[5] and PI[6] are not.
**Fix**: Link PI[5] to the accumulated `ErrorTracker.total_error`. For PI[6], enforce sequential ordering through the aggregation circuit.

### W4: MEDIUM -- Commitment Module Has No In-Circuit Hash Constraints
**Location**: `commitment/model_commit.rs` (entire file)
**Impact**: Model weight commitment is computed natively and passed as a public input, but nothing in the circuit proves the hash matches the actual weights. `verify_path()` always returns `Ok(())`.
**Fix**: Implement in-circuit Poseidon hashing for weight commitment (Poseidon gadget already exists), or clearly document that weight binding is trusted-prover-only.

### W5: MEDIUM -- Embedding Merkle Path Verification Is a Stub
**Location**: `ml/embedding.rs:245-309`
**Impact**: `s_hash` gate checks `parent == parent` (self-equality). Prover can supply any Merkle path.
**Fix**: Use `poseidon_hash_two()` gadget to compute `hash(left, right)` in-circuit and constrain against expected parent.

### W6: MEDIUM -- Error Accumulation Uses Separate Regions Without Copy Constraints
**Location**: `approximate/error_accumulation.rs:174-237`
**Impact**: Unlike the fixed bounded_mul.rs, the error accumulation circuit still uses separate regions for mul error propagation without cross-region copy constraints. ReLU/Div error is unconstrained.
**Fix**: Rewrite to single-region pattern matching bounded_mul.rs, or add explicit `constrain_equal` calls.

### W7: LOW -- Poseidon Constants Non-Standard
**Location**: `gadgets/poseidon.rs:93-130`
**Impact**: Hash outputs differ from any standard Poseidon implementation (Grain LFSR). Cannot interop with other ZK systems.
**Fix**: Replace with standard BN254 Poseidon constants for interoperability.

### W8: LOW -- ReLU `compute_field` Negativity Check Is Always False
**Location**: `lookup/relu.rs:94`
**Impact**: Uses `bytes[31] & 0x80 != 0` which is always false for BN254 Fr. Only affects native witness computation; the lookup table itself is correct.
**Fix**: Compare against `(p-1)/2` for proper sign detection.

---

## 6. Prioritized Recommendations

### Critical (All Previously Identified Critical Issues RESOLVED)

1. ~~Fix cache module ambiguity~~ DONE -- deleted cache.rs
2. ~~Constrain error checksum in circuit~~ DONE -- 3 in-circuit Poseidon hashes
3. ~~Remove non-functional gadgets~~ DONE -- deleted freivalds, comparison, swap, builder
4. ~~Audit dead code~~ DONE -- removed optimization/, profiling/, stubs
5. ~~Add copy constraints to approximate gadgets~~ DONE -- single-region rewrites

### High Priority (Remaining)

6. **Fix transformer verification** -- Replace self-equality checks with actual computation verification. Reuse existing LayerNormChip, SoftmaxChip, GELU lookup. (Effort: Medium, Impact: Large)

7. **Link PI[5] error bound to computed value** -- Add `constrain_equal` binding PI[5] to the ErrorTracker's accumulated total. (Effort: Low, Impact: Medium)

8. **Replace or document SolidityGenerator** -- Either integrate `halo2-solidity-verifier` or clearly mark the generated contract as a structural placeholder. (Effort: Low, Impact: Clarity)

### Nice-to-Have

9. Use standard Poseidon constants for interoperability
10. Add in-circuit Poseidon for weight commitment (replaces native SHA-256)
11. Fix embedding Merkle path verification
12. Fix error_accumulation.rs copy constraint gap
13. Add property-based tests for circuit soundness

---

## 7. Improvement Ideas

### 7.1 Optimizations (Expected Impact: 2-5x proving speedup)
- **Parallel witness generation**: `compute_witness_v2()` is sequential. Matmul and hash operations can be parallelized with Rayon. (Complexity: Medium)
- **Dynamic K selection**: Many circuits use k=14 (16384 rows) but only fill ~5000. Dynamic K would reduce proving time. (Complexity: Low)
- **Batch Poseidon hashing**: Multiple Poseidon hashes in IVC can share round constants and be batched. (Complexity: Medium)

### 7.2 New Features
- **Convolutional layer circuit**: Only MLP layers are supported. Conv2d would enable CNN training verification. (Complexity: High)
- **Adam/AdamW optimizer**: Only SGD is implemented. Adam requires moment tracking in the witness. (Complexity: Medium)
- **Recursive proof composition**: Use IVC folding to compress multi-step proofs into a single constant-size proof. (Complexity: Very High)

### 7.3 Integration Opportunities
- **helix-avm integration**: Connect circuit generation to the AVM executor for end-to-end automated proving. (Complexity: High)
- **EVM gas benchmarking**: Deploy to local Anvil and measure `submitProof` gas. (Complexity: Low)

---

## 8. Testing Assessment

### Coverage Summary

| Module | Tests | Coverage | Quality |
|--------|-------|----------|---------|
| ivc.rs | 22 | Excellent | Real KZG proofs, soundness tests |
| ml/training_step_v2 | 9 | Good | Real SHPLONK proof, PI adversarial |
| ml/transformer | 21 | Moderate | High count, but circuit checks are self-equality |
| ml/proof_aggregation | 11 | Excellent | Broken chain, wrong RLC, wrong loss |
| ml/batch | 7 | Good | 1-instance and 2-instance MockProver |
| ml/attention | 4 | Adequate | MockProver, structural only |
| ml/embedding | 8 | Good | Range checks, Merkle paths, OOV |
| ml/config | 11 | Good | Builder, presets, estimation |
| approximate/ | ~18 | Good | Positive + negative tests |
| verifier/ | ~45 | Good | Format roundtrips, pipeline tests |
| lookup/ | ~21 | Good | Table correctness, MockProver |
| quantization/ | ~16 | Good | Circuit satisfaction |
| gadgets/ | ~15 | Good | Poseidon circuit + determinism |
| params/ | ~25 | Good | Key serialization |
| cache/ | ~21 | Good | Put/get, eviction, stats |

### Missing Tests

1. **Transformer actual constraint verification** -- Current tests pass trivially due to self-equality
2. **PI[5] error bound forgery** -- No test demonstrating arbitrary error bound acceptance
3. **Embedding Merkle forgery** -- No test exploiting the self-equality Merkle path
4. **Freivalds wrong-matmul rejection** -- Test with intentionally incorrect matrix product
5. **Fuzz/property-based testing** -- No proptest/quickcheck anywhere in the crate

---

## 9. Demo Readiness

### ETHDenver Targets

| Target | Status | Notes |
|--------|--------|-------|
| **< 500ms proof generation** | Not Yet | ~2-3s per proof in release (k=14). Would need k optimization or hardware acceleration. |
| **~30x overhead** | Likely | IVC real proof tests show ~1-2s per step. `--bench` flag on helix-demo measures this. |
| **90-second total demo** | Ready | helix-demo targets 20 steps x 3 workers = 60 proofs. Parallelizable. |
| **Working adversarial demo** | Ready | PI[7] constrained. 5 adversarial PI tests pass. |
| **On-chain verification** | Partial | EVM format module solid. `Halo2Verifier.sol` in contracts works. PoseidonHasher.sol deployed. |

### Remaining Demo Blockers

1. Benchmark real proving time in release mode
2. Verify contract-side Poseidon checksum matches circuit output
3. Document PI[5]/PI[6] as prover-asserted values in demo context

---

## 10. Summary

### Health Score: **C+ (68%)**

### What Changed Since Last Review (2026-02-10 -> 2026-02-11)

| Change | Impact |
|--------|--------|
| Copy constraints fixed in bounded_add, bounded_mul, bounded_matmul, activation | Soundness: C- -> B for approximate/ |
| Error checksum PI[7] constrained via 3 in-circuit Poseidon hashes | Soundness: Critical gap closed |
| Cache module ambiguity resolved (deleted cache.rs) | Compilation: Fixed |
| Dead code removed (~5,700 lines) | Maintainability: Significant improvement |
| Non-functional gadgets deleted (freivalds, comparison, swap, builder) | Code quality: Clean |
| optimization.rs and profiling/ deleted | Code quality: Clean |
| gradient_commit.rs deleted | Code quality: Removed stub |

### Assessment

helix-circuits is architecturally ambitious and increasingly sound. The core training circuit (`training_step_v2.rs`) and IVC module (`ivc.rs`) are the strongest components with real KZG proof tests. The approximate arithmetic gadgets now have proper copy constraints preventing bypass attacks. Error checksum (PI[7]) is constrained via in-circuit Poseidon hashing.

The remaining gaps are concentrated in: (1) the transformer circuit's self-equality facade, (2) the commitment module's lack of in-circuit hashing, (3) PI[5]/PI[6] being prover-asserted, and (4) the generated Solidity verifier being a placeholder. For the ETHDenver demo, items (3) and (4) are acceptable with documentation, but (1) and (2) mean the transformer and embedding circuits do not provide the security guarantees their API suggests.

### Key Metrics

| Metric | Value |
|--------|-------|
| Lines of Code | ~30,700 (was ~40,000; ~9,300 lines removed) |
| Test Functions | 329+ |
| Compilation Status | **PASSES** |
| Estimated Test Coverage | ~65% |
| Documentation Quality | Moderate (doc comments, architecture docs) |
| Dead Code | Minimal (all known dead code removed) |
| Critical Security Issues | 0 (PI[7] constrained, copy constraints fixed) |
| High Security Issues | 2 (transformer facade, SolidityGenerator placeholder) |
| Code Quality | B (well-structured, consolidated) |
