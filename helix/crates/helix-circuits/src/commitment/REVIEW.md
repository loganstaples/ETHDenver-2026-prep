# commitment/ -- Code Review

## Overview

The `commitment/` module provides circuits for committing to model weights, gradient updates, and state transitions during verifiable ML training. It uses SHA-256 Merkle tree commitments to bind weight arrays to compact hashes, enabling the ZK circuit to prove that a training step transformed weights from one committed state to another. The module is sparse: `gradient_commit.rs` is a near-empty placeholder, `model_commit.rs` has native SHA-256 but no in-circuit verification, and `state_transition.rs` is a thin wrapper around gadgets.

**Files:** 4 (mod.rs, gradient_commit.rs, model_commit.rs, state_transition.rs)
**Total lines:** ~238

---

## Per-File Analysis

### mod.rs (3 lines)

Declares three submodules:
```rust
pub mod gradient_commit;
pub mod model_commit;
pub mod state_transition;
```

### gradient_commit.rs (16 lines)

**This file is an empty placeholder.** It contains only two type aliases:

```rust
pub type GradientCommitChip<F> = ModelCommitChip<F>;
pub type GradientCommitConfig<F> = ModelCommitConfig<F>;
```

There is no gradient-specific commit logic. Gradient commitment is delegated entirely to `ModelCommitChip`, which itself only performs native (non-circuit) SHA-256 hashing.

### model_commit.rs (142 lines)

Implements model weight commitment:

| Type | Purpose |
|------|---------|
| `ModelCommitChip<F>` | Chip for model weight commitment |
| `ModelCommitConfig<F>` | Configuration (phantom data only) |

Key functions:
- `compute_model_commitment(weights: &[f64]) -> (Fr, Fr)` (lines 80-110): Converts weights to bytes, SHA-256 hashes them, splits the 256-bit hash into two 128-bit halves (lo, hi), and returns as `(Fr, Fr)`. This matches the contract's `_hashPair(lo, hi)` format.
- `compute_merkle_root(leaf_hashes: &[[u8; 32]]) -> [u8; 32]` (lines 115-140): Standard binary Merkle tree computation: pairs adjacent leaves, SHA-256 hashes each pair, recurses until one root remains.
- `verify_path(leaf, path, index, root)` (lines 60-70): **ALWAYS RETURNS Ok(())**. This is a no-op placeholder.

The `configure` method (lines 30-40) creates advice and instance columns but defines no gates. The `synthesize` method is not implemented on any circuit type.

### state_transition.rs (77 lines)

Implements the weight update formula `M_new = M_old - lr * G`:

| Type | Purpose |
|------|---------|
| `StateTransitionChip<F>` | Chip using BoundedAddChip and BoundedMulChip |

Key method: `assign_transition(old_w, lr, grad)` (lines 30-65):
1. `update = lr * grad` (via BoundedMulChip)
2. `neg_update = update * (-1)` (via BoundedMulChip)
3. `new_w = old_w + neg_update` (via BoundedAddChip)

This correctly uses the bounded arithmetic gadgets which track error margins through each operation. The chip delegates to `gadgets::bounded_ops` for the actual constraint generation.

---

## Strengths

1. **Correct hash format for contract compatibility** (model_commit.rs:90-100): The `compute_model_commitment` function splits the SHA-256 hash into lo/hi 128-bit halves matching the Solidity contract's `_hashPair(lo, hi)` reconstruction. This ensures Rust-side commitments match on-chain verification.

2. **Standard Merkle tree** (model_commit.rs:115-140): `compute_merkle_root` implements correct binary Merkle tree construction with proper pair hashing. Handles odd-length layers by promoting the unpaired leaf.

3. **Error-tracking state transition** (state_transition.rs:30-65): The weight update uses `BoundedMulChip` and `BoundedAddChip`, which propagate error bounds through the computation. This ties into HELIX's error budget system.

4. **Type alias design for gradient commit** (gradient_commit.rs:5-6): While the file is nearly empty, aliasing `GradientCommitChip = ModelCommitChip` is a reasonable approach since both weight and gradient commitments use the same hash structure. The separate module allows future divergence.

---

## Weaknesses

### W1: verify_path is a no-op -- no Merkle proof verification
- **Location**: model_commit.rs:60-70
- **Code**: `pub fn verify_path(...) -> Result<(), ...> { Ok(()) }`
- **Impact**: CRITICAL -- There is no Merkle path verification. Any claimed leaf-to-root proof is accepted unconditionally. This means the circuit cannot verify that a specific weight belongs to a committed model.
- **Fix**: Implement standard Merkle path verification:
  ```rust
  pub fn verify_path(leaf: [u8; 32], path: &[[u8; 32]], index: usize, root: [u8; 32]) -> Result<(), CommitError> {
      let mut current = leaf;
      let mut idx = index;
      for sibling in path {
          current = if idx % 2 == 0 {
              sha256_pair(&current, sibling)
          } else {
              sha256_pair(sibling, &current)
          };
          idx /= 2;
      }
      if current == root { Ok(()) } else { Err(CommitError::InvalidMerkleProof) }
  }
  ```

### W2: No in-circuit SHA-256 constraints
- **Location**: model_commit.rs:30-40 (configure), entire file
- **Impact**: CRITICAL -- `compute_model_commitment` and `compute_merkle_root` are native Rust functions, not circuit operations. There are no halo2 constraints proving the hash was computed correctly. The commitment is computed outside the circuit and passed as a public input, but nothing in the circuit constrains that the public input matches the actual weights.
- **Fix**: Integrate a halo2-compatible SHA-256 gadget (e.g., from `halo2_gadgets::sha256`) to constrain hash computation in-circuit. This is a major implementation effort (~500-1000 lines) but is essential for soundness.

### W3: gradient_commit.rs is a placeholder with no gradient-specific logic
- **Location**: gradient_commit.rs (entire file, 16 lines)
- **Impact**: MEDIUM -- Gradient commitment has unique requirements (e.g., committing to gradient magnitude bounds, verifying gradient aggregation) that are not addressed by reusing `ModelCommitChip`.
- **Fix**: At minimum, add a `compute_gradient_commitment` function that commits to gradient values with an error bound. For full implementation, add a `GradientBoundsChip` that constrains `|gradient| <= max_gradient` within the commitment.

### W4: State transition has no overflow/underflow protection
- **Location**: state_transition.rs:45-55
- **Impact**: MEDIUM -- The update `new_w = old_w - lr * grad` can produce values outside the representable range of the quantization scheme. There is no clipping or range check after the update.
- **Fix**: Add a range check after the state transition: `assert!(new_w >= MIN_WEIGHT && new_w <= MAX_WEIGHT)` using a lookup table or decomposition constraint.

### W5: compute_model_commitment uses f64-to-bytes conversion that loses information
- **Location**: model_commit.rs:85-90
- **Impact**: LOW -- Weights are converted to bytes via `f64::to_le_bytes()`. This preserves the exact IEEE 754 representation, which is correct. However, if the prover uses a different floating-point representation (e.g., quantized INT8 values stored as f64), the commitment will differ from a commitment computed directly on INT8 bytes.
- **Fix**: Document the expected input format clearly, or add overloaded methods: `commit_f64_weights()`, `commit_int8_weights()`, `commit_quantized_weights()`.

---

## Health Score: D+

**Rationale**: This module is the weakest in helix-circuits. The two critical issues -- no-op Merkle verification (W1) and no in-circuit hash constraints (W2) -- mean that the commitment system provides no cryptographic guarantees within the ZK proof. The SHA-256 hashes are computed natively and passed as public inputs, but nothing in the circuit proves these hashes are correct. The state transition chip (state_transition.rs) is the only piece with real circuit constraints, and it works correctly via delegation to the bounded arithmetic gadgets. For production, this module needs a major implementation effort to add in-circuit hashing.
