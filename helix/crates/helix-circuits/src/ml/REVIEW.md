# ml/ Module Review

**Module**: `helix-circuits/src/ml/`
**Reviewed**: 2026-02-11 (updated; original 2026-02-10)
**Files**: 14 files, ~10,642 lines of code, 93 tests

---

## 1. Overview

The `ml/` module is the core of helix-circuits: it contains every Halo2 circuit responsible
for proving ML training computations in zero knowledge. The module proves a complete
2-layer MLP training step (forward pass, loss, backward pass, weight update), along with
supporting circuits for transformer blocks, attention, embeddings, softmax, layer
normalization, gradient verification, positional encodings, batch proving, and proof
aggregation.

All circuits target the PSE Halo2 fork (KZG on BN254). The critical contract interface
exposes 8 public inputs per training step:
`[old_hash_lo, old_hash_hi, new_hash_lo, new_hash_hi, loss, error_bound, step_number, error_checksum]`.

---

## 2. Architecture

```
mod.rs                     Module root; re-exports config types
    |
    +-- training_step_v2   THE critical-path circuit: full 2-layer MLP training step
    |       |
    |       +-- batch      Wraps N training steps into a single proof (shared config)
    |       +-- proof_aggregation   SHPLONK-style aggregation (PI chaining, Fiat-Shamir RLC)
    |
    +-- transformer        Transformer block circuit (attention + FFN + LayerNorm + residual)
    |       |
    |       +-- attention  Standalone multi-head attention circuit
    |       +-- linear_layer  Bounded linear layer (y = Wx + b + error tracking)
    |       +-- layer_norm Layer normalization / RMSNorm circuit
    |       +-- softmax    Standalone softmax with exp lookup tables
    |       +-- gradient   Gradient verification (ReLU backward, weight update gates)
    |
    +-- embedding          Embedding lookup with Merkle tree commitment
    +-- positional         Sinusoidal, learned, and RoPE positional encodings
    +-- aggregation        Gradient aggregation for federated learning (Byzantine tolerance)
    +-- config             TransformerConfig, presets, parameter/constraint estimation
```

**Data flow for on-chain verification:**
1. `training_step_v2` produces a proof with 8 public inputs.
2. `batch` combines N such proofs into one (shared SRS, concatenated PIs).
3. `proof_aggregation` chains N proofs with PI continuity (step[i].new_hash == step[i+1].old_hash),
   Fiat-Shamir RLC, error/loss accumulation.
4. The resulting proof is submitted on-chain to `Halo2Verifier.sol`.

---

## 3. Per-File Analysis

### training_step_v2.rs (2090 lines, 9 tests)

The most important file in the entire codebase. Proves a complete training step for a
2-layer MLP: forward pass (matmul + ReLU), MSE loss, backward pass (dW2, dW1, dh, relu mask),
and SGD weight update. All 8 public inputs are bound via `constrain_instance`.

**Key types:** `MLTrainingStepV2Config`, `MLTrainingStepV2Circuit`, `MLTrainingStepV2Witness`,
`ErrorTracker`, `CircuitMetrics`.

**What works well:**
- Freivalds randomized verification for matmul (lines 1233-1312): O(n^2) instead of O(n^3).
- Complete forward+backward+update pipeline with 10 constrained stages (lines 857-1021).
- Poseidon-based state hash for weight commitment (`compute_state_hash_v2`, line 1782).
- Real KZG proof generation test (lines 1959-2023) with SHPLONK verification.
- EVM-compatible proof/PI format traits (`ToEvmProof`, `ToEvmPublicInputs`, lines 1027-1083).
- Error tracking via `ErrorTracker` with per-operation error accounting.
- ReLU negative detection correctly fixed (line 1629): `bytes[31] >= 0x19` for BN254 MSB check.
- **Error checksum PI[7] now constrained** via 3 in-circuit Poseidon hashes in `verify_error_checksum()`.

**Remaining issues:** See Section 6 (Weaknesses), items W1 and W2.

---

### transformer.rs (1975 lines, 21 tests)

Full transformer block circuit: multi-head attention, feed-forward network (with GELU),
two layer normalization stages, and residual connections.

**Key types:** `TransformerBlockCircuit`, `TransformerBlockChip`, `TransformerBlockWitness`,
`TransformerCircuit` (multi-block wrapper).

**What works well:**
- Residual connection verification is correct (lines 528-566): constraints `input + main = output`.
- Good test count: 21 tests including witness generation, config creation, multi-block,
  performance benchmarks (500K/1M/2M parameters).

**Critical issues:** See Section 6 (Weaknesses), item W3.

---

### attention.rs (1117 lines, 4 tests)

Standalone multi-head attention circuit with Q/K/V projection, scaled dot-product attention,
and output projection.

**What works well:**
- Full dot product verification in `verify_projection` (lines 352-414).
- Proper score scaling verification.
- Attention weight sum constraint (weights per position must sum to SOFTMAX_SCALE).

**Issue:** Softmax is simplified to uniform distribution in witness generation
(line ~890: `1.0 / seq_len as f64` for all weights). The attention circuit never actually
verifies real softmax computation.

---

### linear_layer.rs (497 lines, 3 tests)

Bounded linear layer: `y = Wx + b` with error propagation.

**What works well:**
- Uses `BoundedMatMulChip` and `ArithmeticChip` from the gadgets module.
- Proper error propagation formula. Test for invalid output rejection.

**No critical issues.** One of the more solid files.

---

### layer_norm.rs (781 lines, 5 tests)

Layer normalization circuit with mean, variance, inverse sqrt, and affine transform.

**What works well:**
- Proper gate structure: separate selectors for mean, variance, inv_sqrt, normalize, affine.
- Supports both standard LayerNorm and RMSNorm.

**Issue:** `field_to_f64` helper only reads the first 8 bytes of the 32-byte field element
representation. Correct only for small values (< 2^64).

---

### softmax.rs (485 lines, 2 tests)

Standalone softmax circuit using exp lookup tables with max-subtraction for stability.

**Issue:** Only 2 tests, and neither runs `MockProver::run` on `SoftmaxCircuit`. Circuit
constraints are never actually tested.

---

### gradient.rs (542 lines, 3 tests)

Gradient verification circuit for backpropagation.

**Critical issue:** The `relu_grad_mask` gate (line 112) has constraint
`local * forward - local * upstream`, which factors to `local * (forward - upstream)`.
This does NOT correctly implement ReLU backward. However, `training_step_v2.rs` implements
its own ReLU backward via `assign_mul` with relu_mask (line 963), so the critical path
is not affected.

---

### embedding.rs (737 lines, 8 tests)

Embedding lookup circuit with Merkle tree commitment.

**Issue:** Merkle path verification (lines 245-309) assigns parent = witness-provided value
and checks `parent == parent` (self-equality). It does NOT compute the hash in-circuit.
A malicious prover can provide any Merkle path leading to the expected root.

---

### positional.rs (813 lines, 7 tests)

Sinusoidal, learned, and RoPE positional encodings.
**No critical issues.** Well-structured with clear separation of concerns.

---

### aggregation.rs (725 lines, 5 tests)

Gradient aggregation for federated/distributed learning with Byzantine tolerance.
**No critical issues.** Weight sum enforcement, outlier detection, stake-weighted aggregation.

---

### batch.rs (296 lines, 7 tests)

Batch proving: wraps N `MLTrainingStepV2Circuit` instances into a single proof.
**No critical issues.** Clean, focused design. MAX_BATCH_SIZE = 32.

---

### proof_aggregation.rs (795 lines, 11 tests)

SHPLONK-style proof aggregation circuit. Real Halo2 circuit proving structural correctness
of aggregating N training step proofs. PI chaining, Fiat-Shamir RLC, error/loss accumulation.
**No critical issues.** One of the strongest files in the module. 11 tests with negative testing.

---

### config.rs (782 lines, 11 tests)

Configuration types for transformer models. Builder pattern, presets, estimation.
**No critical issues.** Pure configuration code, well-tested.

---

## 4. Critical Path

The critical path for on-chain proof submission is:

```
training_step_v2.rs::MLTrainingStepV2Circuit
    -> synthesize_instance() [lines 815-1024]
        1. Bind 8 PIs via constrain_instance        [lines 835-855]
        2. Forward L1: matmul (Freivalds) + bias + ReLU  [lines 857-882]
        3. Forward L2: matmul (Freivalds) + bias         [lines 884-907]
        4. Loss: MSE computation                         [lines 909-928]
        5. Backward: dy, dW2, db2, dh                    [lines 930-959]
        6. Backward: ReLU mask, dh_pre                   [lines 961-965]
        7. Backward: dW1, db1                            [lines 967-980]
        8. Weight updates: W_new = W_old - lr * dW       [lines 982-1010]
        9. Error bound verification                      [lines 1012-1014]
       10. Error checksum verification (Poseidon)        [lines 1016-1021]
```

**Public Inputs (8 elements):**

| Index | Name | Constrained By |
|-------|------|---------------|
| 0-1 | old_hash_lo/hi | constrain_instance + Poseidon state hash |
| 2-3 | new_hash_lo/hi | constrain_instance + Poseidon state hash |
| 4 | loss | constrain_instance + loss gates (lines 909-928) |
| 5 | error_bound | constrain_instance only (**witness-only, no range check**) |
| 6 | step_number | constrain_instance only (**witness-only, no ordering check**) |
| 7 | error_checksum | constrain_instance + **3 in-circuit Poseidon hashes (FIXED)** |

PIs 0-4 and 7 are constrained by the circuit's arithmetic/hash gates. PIs 5-6 are
**prover-chosen values** -- the circuit binds them to the instance column but never
verifies their correctness through constraints.

---

## 5. Strengths

**S1. Freivalds Randomized Verification**
`training_step_v2.rs:1233-1312` -- O(n^2) matrix multiplication verification.

**S2. Complete Training Pipeline in a Single Circuit**
`training_step_v2.rs:815-1024` -- Forward-backward-update fully constrained.

**S3. Real KZG Proof Generation with SHPLONK**
`training_step_v2.rs:1959-2023` -- End-to-end proof generation and verification.

**S4. SHPLONK Aggregation Circuit with Fiat-Shamir**
`proof_aggregation.rs:1-795` -- Real aggregation with PI chaining and RLC.

**S5. Error Checksum Now Constrained (FIXED)**
`training_step_v2.rs:1339-1360` -- PI[7] verified via 3 in-circuit Poseidon hashes.

**S6. BN254 Negative Number Detection (FIXED)**
`training_step_v2.rs:1624-1629` -- ReLU correctly uses `bytes[31] >= 0x19`.

**S7. Byzantine-Tolerant Gradient Aggregation**
`aggregation.rs` -- Outlier detection, weight sum constraint, stake weighting.

---

## 6. Weaknesses

### W1. PI[5] Error Bound and PI[6] Step Number Are Witness-Only [MEDIUM]

**Location:** `training_step_v2.rs:1315-1329` and `training_step_v2.rs:1012-1014`

**What happens:** The error bound and step number are assigned to advice cells and bound
to the instance column, but no circuit constraint verifies their correctness.

**Impact:** A prover can claim an arbitrarily low error bound or any step number.

**Suggested fix:** Link PI[5] to the accumulated ErrorTracker total. For PI[6], enforce
sequential ordering through the aggregation circuit.

---

### W2. ~~PI[7] Error Checksum Is Unconstrained~~ FIXED

PI[7] is now constrained via `verify_error_checksum()` which synthesizes 3 in-circuit
Poseidon hashes: `Poseidon(total_error, step_number)`, `Poseidon(model_id, error_budget)`,
and `Poseidon(h1, h2)`. Cost: ~2,292 extra rows. The circuit output is constrained to
equal PI[7].

---

### W3. Transformer Verification Is Mostly Self-Equality [HIGH]

**Location:** `transformer.rs:414-652`

**What happens:**
- `verify_layer_norm` compares output to itself (self-equality gate)
- `verify_attention` only checks `.min(2)` positions; Q projection uses self-equality
- `verify_ffn` samples `.min(2)` positions with self-equality on activation

**Impact:** The transformer circuit accepts ANY witness values for layer norm, most
attention positions, and most FFN positions.

**Suggested fix:** Reuse `LayerNormChip` from `layer_norm.rs`, integrate `SoftmaxChip`
from `softmax.rs`, use GELU lookup for FFN activation, remove `.min()` bounds.

---

### W4. gradient.rs ReLU Backward Gate Is Incorrect [MEDIUM]

**Location:** `gradient.rs:103-113`

**What happens:** Constraint `local * (forward - upstream)` does not correctly implement
ReLU backward. However, `training_step_v2.rs` has its own correct implementation.

---

### W5. Embedding Merkle Path Verification Is a Stub [MEDIUM]

**Location:** `embedding.rs:245-309`

**What happens:** `s_hash` gate checks `parent == parent` (self-equality). No in-circuit
hash computation.

**Suggested fix:** Use `poseidon_hash_two()` gadget from `gadgets/poseidon.rs`.

---

### W6. Softmax in Attention Uses Uniform Distribution [LOW]

**Location:** `attention.rs:~890`

**Impact:** Attention weights set to `1.0 / seq_len` regardless of Q*K^T scores.

---

### W7. Softmax Circuit Lacks MockProver Test [LOW]

**Location:** `softmax.rs` test module

**Impact:** Circuit constraints are never actually tested.

---

## 7. Testing Assessment

| File | Tests | MockProver | Real Proof | Negative Tests | Coverage |
|------|-------|-----------|------------|----------------|----------|
| training_step_v2.rs | 9 | Yes | Yes (SHPLONK) | Yes (tampered PI) | Good |
| transformer.rs | 21 | Yes | No | No | High count, low depth |
| attention.rs | 4 | Yes | No | No | Adequate |
| linear_layer.rs | 3 | Yes | No | Yes (invalid output) | Good |
| layer_norm.rs | 5 | Yes | No | No | Adequate |
| softmax.rs | 2 | **No** | No | No | **Poor** |
| gradient.rs | 3 | Yes | No | No | Adequate |
| embedding.rs | 8 | Yes | No | Yes (OOV) | Good |
| positional.rs | 7 | Yes | No | No | Good |
| aggregation.rs | 5 | Yes | No | Yes (weight sum) | Good |
| batch.rs | 7 | Yes (1+2 inst) | No | Yes (empty, oversized) | Good |
| proof_aggregation.rs | 11 | Yes | No | Yes (chain/RLC/loss) | **Excellent** |
| config.rs | 11 | N/A | N/A | Yes (invalid config) | Good |
| **Total** | **93** | | | | |

---

## 8. Health Score

**Grade: C+**

**Justification:**

The critical path (`training_step_v2.rs`) is genuinely functional with real KZG proofs,
Freivalds verification, and in-circuit Poseidon checksum. The aggregation and batch circuits
are solid. Supporting infrastructure (config, positional, linear_layer) is clean.

Improvements since last review:
- PI[7] error checksum is now constrained (was critical gap)
- ReLU negative detection fixed (`bytes[31] >= 0x19`)

Remaining issues preventing a higher grade:
1. **PI[5-6] are still prover-chosen (W1).** Error bound and step number unconstrained.
2. **Transformer circuit is a verification facade (W3).** Self-equality checks prove nothing.
3. **Embedding Merkle path is a stub (W5).** Hash verification is self-equality.

| Component | Sub-Grade | Notes |
|-----------|-----------|-------|
| training_step_v2.rs (core) | B | Strong constraints, PI[7] fixed, PI[5-6] weak |
| batch.rs + proof_aggregation.rs | A- | Proper aggregation, good tests |
| transformer.rs | D | Self-equality facade |
| attention.rs + softmax.rs | C | Structural but not semantic verification |
| linear_layer.rs + layer_norm.rs | B | Correct gates, reasonable tests |
| embedding.rs | C- | Good structure, stub Merkle hash |
| gradient.rs | D+ | Incorrect ReLU gate, not used in critical path |
| positional.rs + config.rs | B+ | Clean, well-tested |
| Test suite overall | B- | 93 tests, good negative tests, gaps in softmax/transformer |
