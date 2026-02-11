# HELIX Circuits Security Model

## Public Input Constraint Summary

The circuit exposes 8 public inputs (PIs) that form the contract-circuit interface:

| Index | Name | Constraint Type | How Constrained |
|-------|------|----------------|-----------------|
| PI[0] | old_hash_lo | **Direct** | Instance-bound via `constrain_instance` |
| PI[1] | old_hash_hi | **Direct** | Instance-bound via `constrain_instance` |
| PI[2] | new_hash_lo | **Direct** | Instance-bound via `constrain_instance` |
| PI[3] | new_hash_hi | **Direct** | Instance-bound via `constrain_instance` |
| PI[4] | loss | **Direct** | Instance-bound; computed in-circuit from forward pass |
| PI[5] | total_error | **Direct** | In-circuit accumulation via `s_error_acc` gate; `constrain_equal` binds accumulated result to PI[5] instance cell |
| PI[6] | step_number | **Indirect** | Committed via PI[7] Poseidon hash; changing PI[6] changes checksum |
| PI[7] | error_checksum | **In-circuit** | 3 Poseidon hashes constrain: `Poseidon(Poseidon(total_error, step_number), Poseidon(model_id, error_budget))` |

All 8 PIs are verified by adversarial MockProver tests that confirm tampering with any single PI causes proof rejection.

## Transformer Verification

The transformer circuit (`ml/transformer.rs`) verifies three components with real arithmetic constraints:

- **Layer Normalization**: Constrains `(x - mean) * inv_std * scale_inv` computation, then `gamma * normalized + beta = output` via `s_sub`, `s_layer_norm`, `s_mul`, `s_add`, `s_eq` gates.
- **Multi-Head Attention**: Q projection verified via `s_linear` gate for ALL positions. Attention weight sum check (sum == 256) and output verification for all positions. No `.min()` caps.
- **FFN**: Both linear layers verified via `s_linear` for ALL `seq_len * d_ff` positions. GELU activation constrained via `activated * (hidden - activated) = 0`. Output verified via `s_eq`.

Soundness tests verify wrong layer norm, FFN hidden, FFN output, and attention weights are rejected by MockProver.

## Freivalds Verification

The circuit uses Freivalds' algorithm for matrix-vector multiplication verification. In the current architecture (2-layer MLP), all matmuls are matrix-vector (n=1), which makes the Freivalds check a **deterministic equality verification** (not probabilistic). The random challenge vector `r` is only used for the general matrix-matrix case (n > 1).

The Freivalds equality gate (`s_freivalds`) constrains `advice[0] == advice[1]`, verifying that the prover's claimed matmul output matches the in-circuit recomputation.

## Error Bound Algebra

HELIX tracks numerical error through every operation:
- Each arithmetic operation accumulates a base error term
- Error propagation follows standard floating-point analysis rules
- `total_error` (PI[5]) is computed in-circuit by accumulating per-operation error terms via the `s_error_acc` gate, then constrained equal to the PI[5] instance cell
- The error checksum (PI[7]) cryptographically commits to the error state

The error bound system directly constrains PI[5] to the in-circuit accumulated error. A malicious prover cannot claim lower error without violating the `s_error_acc` accumulation or the `s_eq` final check.

## Copy Constraints

All bounded arithmetic gadgets use single-region synthesis with explicit `constrain_equal` calls:
- `bounded_add`: value add + error add + range check in one region, err_c copy-constrained
- `bounded_mul`: 8-row single region with cross-row copy constraints for shared operands
- `bounded_matmul`: per-iteration regions with `AssignedCell` tracking and cross-iteration linking
- `activation` (ReLU): 7-row region with copy constraints for val_y, neg, err_diff, err_y
- `error_accumulation` (mul): 5-row single region with 6 copy constraints binding err_a, err_b, term1, term2, term3, partial_sum across rows

## Contract-Circuit Alignment

The error checksum uses Poseidon hashing in both the circuit (Rust) and contracts (Solidity):
- **Circuit**: `gadgets/poseidon.rs` with SHA-256-derived round constants
- **Contract**: `PoseidonHasher.sol` with identical round constant derivation
- Parameters: width=3, rate=2, 8 full + 57 partial rounds, x^5 S-box
- MDS matrix: `[[2,1,1],[1,2,1],[1,1,2]]`
- Round constants: `SHA256("HELIX_POSEIDON_RC_V1" || round_le64 || i_le64)` with `repr[31] &= 0x1F`

## Known Limitations

1. **Merkle path verification is native-only.** The `s_hash` gate in `embedding.rs` checks `parent == expected` (self-equality), not `hash(left, right) == parent`. Real Merkle integrity relies on native SHA-256 verification outside the circuit. For in-circuit verification, replace with Poseidon hash gadget.

2. **IVC/folding has real math but no compression.** The IVC module implements Nova-style folding with cross-terms, element-wise vector operations, and in-circuit Poseidon commitments. 22 tests pass with real KZG proofs. However, it does not implement actual multi-step proof compression to constant size.

3. **Lookup table range.** ReLU lookup tables cover [-128, 128) by default. Model weights must be quantized to stay within this range for proof generation to succeed.

4. **No in-circuit matmul constraint.** The Freivalds gate checks equality between prover-assigned values. The actual matmul computation is done natively (in the witness), not as in-circuit constraints. Soundness relies on the prover committing to weights via state hash PIs.

5. **Poseidon gas cost.** On-chain Poseidon verification via `PoseidonHasher.sol` computes 65 rounds of SHA-256-derived constants on the fly, consuming ~8M gas. For production, round constants should be precomputed and stored.

6. **SolidityGenerator is simplified.** The generated Solidity verifier implements a simplified KZG pairing check, not full SHPLONK. Use the handwritten `Halo2Verifier.sol` in `contracts/src/verification/` for production on-chain verification.

## Proof Format

- **Commitment scheme**: KZG on BN254 (PSE halo2 fork)
- **Opening scheme**: SHPLONK (2 fixed opening proof points: H, H')
- **Transcript**: Blake2b (for proof generation), Keccak256 (for EVM verification)
- **G1 encoding**: 32-byte compressed with PSE `CompressedFlagConfig::TwoSpare`
