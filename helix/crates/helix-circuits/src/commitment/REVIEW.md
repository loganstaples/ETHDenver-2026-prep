# commitment/ -- Code Review

**Module**: `helix-circuits/src/commitment/`
**Reviewed**: 2026-02-11 (updated; original 2026-02-10)
**Files**: mod.rs (2 lines), model_commit.rs (141 lines), state_transition.rs (76 lines)
**Total lines**: ~219

---

## 1. Overview

The `commitment/` module provides circuits for committing to model weights and state transitions during verifiable ML training. It uses SHA-256 Merkle tree commitments to bind weight arrays to compact hashes, enabling the ZK circuit to prove that a training step transformed weights from one committed state to another.

The module is sparse but focused: `model_commit.rs` has native SHA-256 hashing and Merkle tree construction, and `state_transition.rs` wraps the weight update formula using bounded arithmetic gadgets. The previous placeholder `gradient_commit.rs` (16-line type alias stub) has been **deleted**.

---

## 2. Per-File Analysis

### mod.rs (2 lines)

Declares two submodules:
```rust
pub mod model_commit;
pub mod state_transition;
```

### model_commit.rs (141 lines)

Implements model weight commitment:

| Type | Purpose |
|------|---------|
| `ModelCommitChip<F>` | Chip for model weight commitment |
| `ModelCommitConfig<F>` | Configuration (phantom data only) |

Key functions:
- `compute_model_commitment(weights: &[f64]) -> (Fr, Fr)` (lines 80-110): Converts weights to bytes, SHA-256 hashes them, splits the 256-bit hash into two 128-bit halves (lo, hi), and returns as `(Fr, Fr)`. Matches the contract's `_hashPair(lo, hi)` format.
- `compute_merkle_root(leaf_hashes: &[[u8; 32]]) -> [u8; 32]` (lines 115-140): Standard binary Merkle tree computation.
- `verify_path(leaf, path, index, root)` (lines 60-70): **ALWAYS RETURNS `Ok(())`**. This is a no-op placeholder.

The `configure` method creates advice and instance columns but defines no gates. No `synthesize` method is implemented.

### state_transition.rs (76 lines)

Implements the weight update formula `M_new = M_old - lr * G`:

| Type | Purpose |
|------|---------|
| `StateTransitionChip<F>` | Chip using BoundedAddChip and BoundedMulChip |

Key method: `assign_transition(old_w, lr, grad)` (lines 30-65):
1. `update = lr * grad` (via BoundedMulChip)
2. `neg_update = update * (-1)` (via BoundedMulChip)
3. `new_w = old_w + neg_update` (via BoundedAddChip)

This correctly uses the bounded arithmetic gadgets which now have proper copy constraints.

---

## 3. Strengths

1. **Correct hash format for contract compatibility** (model_commit.rs:90-100): The `compute_model_commitment` function splits SHA-256 into lo/hi 128-bit halves matching the Solidity contract's `_hashPair(lo, hi)` reconstruction.

2. **Standard Merkle tree** (model_commit.rs:115-140): `compute_merkle_root` implements correct binary Merkle tree construction with proper pair hashing and odd-length layer handling.

3. **Error-tracking state transition** (state_transition.rs:30-65): Weight update uses `BoundedMulChip` and `BoundedAddChip` (now with copy constraints), tying into HELIX's error budget system.

4. **Clean module** -- the 16-line stub `gradient_commit.rs` has been deleted.

---

## 4. Weaknesses

### W1: verify_path is a no-op -- no Merkle proof verification (CRITICAL)
- **Location**: model_commit.rs:60-70
- **Code**: `pub fn verify_path(...) -> Result<(), ...> { Ok(()) }`
- **Impact**: Any claimed leaf-to-root proof is accepted unconditionally. The circuit cannot verify that a specific weight belongs to a committed model.
- **Fix**: Implement standard Merkle path verification by hashing each sibling pair and comparing to root. The `poseidon_hash_two` gadget from `gadgets/poseidon.rs` could be used for in-circuit verification.

### W2: No in-circuit SHA-256 constraints (CRITICAL)
- **Location**: model_commit.rs (entire file)
- **Impact**: `compute_model_commitment` and `compute_merkle_root` are native Rust functions, not circuit operations. There are no halo2 constraints proving the hash was computed correctly. The commitment is computed outside the circuit and passed as a public input, but nothing in the circuit constrains that the public input matches the actual weights.
- **Fix**: Integrate in-circuit Poseidon hashing for weight commitment. Poseidon is much cheaper in Halo2 (~764 rows per hash vs ~28K for SHA-256). The gadget already exists in `gadgets/poseidon.rs`.

### W3: State transition has no overflow/underflow protection (MEDIUM)
- **Location**: state_transition.rs:45-55
- **Impact**: The update `new_w = old_w - lr * grad` can produce values outside the representable range. There is no range check after the update.
- **Fix**: Add a range check on the updated weight value.

### W4: compute_model_commitment commits f64 bytes, not circuit-compatible values (LOW)
- **Location**: model_commit.rs:85-90
- **Impact**: Weights are committed via `f64::to_le_bytes()`. If the circuit uses a different representation (e.g., quantized INT8 values or field elements), the commitment will differ. The commitment scheme must match the in-circuit representation.
- **Fix**: Document the expected input format, or add overloaded methods for different representations.

---

## 5. Health Score: D+

**Rationale**: This module is the weakest in helix-circuits. The two critical issues -- no-op Merkle verification (W1) and no in-circuit hash constraints (W2) -- mean the commitment system provides no cryptographic guarantees within the ZK proof. The SHA-256 hashes are computed natively and passed as public inputs, but nothing in the circuit proves these hashes are correct. The state transition chip is the only piece with real circuit constraints, and it works correctly via delegation to the bounded arithmetic gadgets. For production, this module needs in-circuit Poseidon hashing for weight commitment (feasible: the Poseidon gadget is already available in the same crate).
