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
| PI[5] | total_error | **Indirect** | Committed via PI[7] Poseidon hash; changing PI[5] changes checksum |
| PI[6] | step_number | **Indirect** | Committed via PI[7] Poseidon hash; changing PI[6] changes checksum |
| PI[7] | error_checksum | **In-circuit** | 3 Poseidon hashes constrain: `Poseidon(Poseidon(total_error, step_number), Poseidon(model_id, error_budget))` |

All 8 PIs are verified by adversarial MockProver tests that confirm tampering with any single PI causes proof rejection.

## Freivalds Verification

The circuit uses Freivalds' algorithm for matrix-vector multiplication verification. In the current architecture (2-layer MLP), all matmuls are matrix-vector (n=1), which makes the Freivalds check a **deterministic equality verification** (not probabilistic). The random challenge vector `r` is only used for the general matrix-matrix case (n > 1).

The Freivalds equality gate (`s_freivalds`) constrains `advice[0] == advice[1]`, verifying that the prover's claimed matmul output matches the in-circuit recomputation.

## Error Bound Algebra

HELIX tracks numerical error through every operation:
- Each arithmetic operation accumulates a base error term
- Error propagation follows standard floating-point analysis rules
- `total_error` (PI[5]) represents the worst-case accumulated error
- The error checksum (PI[7]) cryptographically commits to the error state

The error bound system is honest-verifier: it trusts the prover's error computation but binds it to the proof via Poseidon hashing. A malicious prover cannot claim lower error without invalidating the checksum.

## Copy Constraints

All bounded arithmetic gadgets use single-region synthesis with explicit `constrain_equal` calls:
- `bounded_add`: value add + error add + range check in one region, err_c copy-constrained
- `bounded_mul`: 8-row single region with cross-row copy constraints for shared operands
- `bounded_matmul`: per-iteration regions with `AssignedCell` tracking and cross-iteration linking
- `activation` (ReLU): 7-row region with copy constraints for val_y, neg, err_diff, err_y

## Contract-Circuit Alignment

The error checksum uses Poseidon hashing in both the circuit (Rust) and contracts (Solidity):
- **Circuit**: `gadgets/poseidon.rs` with SHA-256-derived round constants
- **Contract**: `PoseidonHasher.sol` with identical round constant derivation
- Parameters: width=3, rate=2, 8 full + 57 partial rounds, x^5 S-box
- MDS matrix: `[[2,1,1],[1,2,1],[1,1,2]]`
- Round constants: `SHA256("HELIX_POSEIDON_RC_V1" || round_le64 || i_le64)` with `repr[31] &= 0x1F`

## Known Limitations

1. **Merkle path verification is not implemented in-circuit.** `ModelCommitChip::verify_path()` will panic if called. Native SHA-256 Merkle verification is used out-of-circuit.

2. **IVC/folding is scaffolding.** The IVC module provides the data structures and folding math but does not implement real multi-step proof compression.

3. **Lookup table range.** ReLU lookup tables cover [-128, 128) by default. Model weights must be quantized to stay within this range for proof generation to succeed.

4. **No in-circuit matmul constraint.** The Freivalds gate checks equality between prover-assigned values. The actual matmul computation is done natively (in the witness), not as in-circuit constraints. Soundness relies on the prover committing to weights via state hash PIs.

5. **Poseidon gas cost.** On-chain Poseidon verification via `PoseidonHasher.sol` computes 65 rounds of SHA-256-derived constants on the fly, consuming ~8M gas. For production, round constants should be precomputed and stored.

## Proof Format

- **Commitment scheme**: KZG on BN254 (PSE halo2 fork)
- **Opening scheme**: SHPLONK (2 fixed opening proof points: H, H')
- **Transcript**: Blake2b (for proof generation), Keccak256 (for EVM verification)
- **G1 encoding**: 32-byte compressed with PSE `CompressedFlagConfig::TwoSpare`
