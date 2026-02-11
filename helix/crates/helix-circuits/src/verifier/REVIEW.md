# Verifier Module Review

**Module**: `helix-circuits/src/verifier/`
**Reviewed**: 2026-02-11 (updated; original 2026-02-10)
**Scope**: EVM-compatible proof serialization and verification for Halo2 (PSE fork, KZG/SHPLONK on BN254)

---

## 1. Overview

This module bridges the Rust Halo2 prover and the on-chain Solidity verifier (`Halo2Verifier.sol`). Its responsibilities are:

- **Proof serialization**: Convert Halo2's compressed transcript format (32-byte G1 points) into the 320-byte big-endian EVM format expected by the Solidity verifier.
- **Public input encoding**: Encode the 8 Fr public inputs from `MLTrainingStepV2Circuit` as big-endian uint256 values for Solidity.
- **Solidity code generation**: Produce a verifier contract with embedded VK (structural placeholder -- see W1).
- **Keccak256 transcript**: EVM-compatible Fiat-Shamir transcript using keccak256.
- **Native verification**: Lightweight Rust-side verification wrapper.

The module is critical infrastructure: any byte-ordering mistake or encoding error silently produces invalid proofs that the contract rejects.

---

## 2. Architecture

### SHPLONK Proof Layout (EVM Format)

```
Offset  Size   Content
------  ----   -------
0       192    3 advice commitment points (C0, C1, C2), each 64 bytes (x: u256 BE, y: u256 BE)
192     128    2 opening proof points (W, W'), each 64 bytes (x: u256 BE, y: u256 BE)
------  ----
Total   320    MIN_PROOF_SIZE
```

SHPLONK always writes exactly 2 opening proof points (H, H'), unlike GWC which writes a variable number per rotation set.

### Public Inputs Layout (8 Fr elements = 256 bytes)

| Index | Content | Description |
|-------|---------|-------------|
| 0 | `old_state_hash_lo` | Lower 128 bits of Poseidon(old_weights) |
| 1 | `old_state_hash_hi` | Upper 128 bits of Poseidon(old_weights) |
| 2 | `new_state_hash_lo` | Lower 128 bits of Poseidon(new_weights) |
| 3 | `new_state_hash_hi` | Upper 128 bits of Poseidon(new_weights) |
| 4 | `loss` | Quantized training loss |
| 5 | `error_bound` | Accumulated error bound |
| 6 | `step_number` | Training step counter |
| 7 | `error_checksum` | Poseidon-based commitment to error state |

### Module Dependency Graph

```
mod.rs (re-exports)
  +-- format_spec.rs   (constants, byte encoding, serialize_proof_for_evm)
  +-- evm.rs           (SolidityGenerator, EvmProof, EvmProofBuilder, EvmPublicInputsArray)
  +-- transcript.rs    (Keccak256Write, Keccak256Read, hash_to_fr)
  +-- native.rs        (NativeVerifier, SerializedProof, ProofMetadata)
```

---

## 3. Per-File Analysis

### 3.1 `mod.rs` (22 lines)

Module declarations and public re-exports. Clean and correct.

---

### 3.2 `evm.rs` (~1511 lines)

**Purpose**: Solidity verifier code generation and EVM proof types.

**Key Types**:
- `VkData` -- Verification key data for embedding in Solidity
- `KzgSrs` -- Wrapper for the s*G2 point from KZG setup
- `SolidityGenerator` -- Builder for Solidity verifier contracts
- `EvmProof` -- 320-byte proof wrapper with extraction methods
- `EvmProofBuilder` -- Builder enforcing exactly 3 advice commits + W/W'
- `EvmPublicInputsArray` -- 8-element Fr array with semantic accessors

**Correctness**: The `SolidityGenerator` generates a simplified KZG pairing check that does **not** match the actual SHPLONK verification protocol. See W1.

---

### 3.3 `format_spec.rs` (~856 lines)

**Purpose**: EVM byte format specification, proof serialization, field element encoding.

**Key Functions**:
- `serialize_proof_for_evm()`: Converts Halo2 compressed transcript to EVM format. Correctly reads advice from front and opening proofs from back.
- `read_halo2_compressed_g1()`: Handles PSE's 32-byte `TwoSpare` compressed encoding.
- `fr_to_evm_bytes` / `fq_to_evm_bytes` / `g1_to_evm_bytes`: LE-to-BE conversions.
- `validate_proof_format()`: Validates all 5 G1 points are on BN254 curve.

**Bug**: `SCALAR_FIELD_ORDER` and `BASE_FIELD_PRIME` constants are swapped (see W2). Currently unused.

---

### 3.4 `transcript.rs` (~415 lines)

**Purpose**: Keccak256-based Fiat-Shamir transcript for EVM-compatible challenge derivation.

**Key Types**: `Keccak256Write<W>` (prover-side), `Keccak256Read<R>` (verifier-side).

**Correctness**: Challenge derivation verified against Solidity pattern in tests. `hash_to_fr` correctly implements `uint256(hash) % R` via Horner's method.

---

### 3.5 `native.rs` (~283 lines)

**Purpose**: Native Rust-side proof verification wrapper.

**Correctness Issues**:
- `encode_commitment` truncates 128-bit values to 64 bits (see W3)
- `encode_bytes` only reads first 8 of 31 bytes per chunk (see W4)
- These issues are confined to native.rs and do not affect the EVM serialization path

---

## 4. Strengths

1. **Correct compressed G1 handling** (`format_spec.rs:625-657`): Uses `GroupEncoding::from_bytes` for PSE's `TwoSpare` encoding.

2. **SHPLONK-aware serialization** (`format_spec.rs:566-618`): Correctly extracts H and H' from the last 64 bytes of the transcript.

3. **Robust error handling** (`format_spec.rs:312-369`): `ProofFormatError` with hex-encoded coordinates for debugging.

4. **Correct Horner's method for hash-to-Fr** (`transcript.rs:257-277`): Field arithmetic for `uint256(hash) % R` reduction.

5. **Defensive identity point handling**: Both serialization and decompression check for all-zeros.

6. **~45 tests**: Byte encoding roundtrips, proof structure validation, transcript challenge alignment, pipeline integration.

---

## 5. Weaknesses

### W1 (HIGH): Generated Solidity verifier does not implement real SHPLONK verification

**Location**: `evm.rs:424-487`

**Impact**: The generated contract cannot verify actual Halo2 SHPLONK proofs. The pairing equation is a simplified single-polynomial check. A real SHPLONK verifier needs multiple evaluation points, linearization polynomial, and multi-open reduction.

**Note**: The `Halo2Verifier.sol` contract in `helix/contracts/src/verification/` is a separate, working verifier that does handle real proofs. The `SolidityGenerator` here is a structural placeholder.

**Suggested Fix**: Use `pse/halo2-solidity-verifier` to generate the actual verifier contract, or clearly document that `SolidityGenerator` is a placeholder and point users to the real contract.

### W2 (MEDIUM): SCALAR_FIELD_ORDER and BASE_FIELD_PRIME constants are swapped

**Location**: `format_spec.rs:117-131`

**Impact**: Currently zero (neither constant is referenced in logic). But any future validation code would use wrong bounds.

**Suggested Fix**: Swap the byte arrays.

### W3 (MEDIUM): encode_commitment truncates 128-bit to 64-bit

**Location**: `native.rs:171-187`

**Impact**: `Fr::from(low as u64)` discards upper 64 bits. Not on critical path.

**Suggested Fix**: Use `Fr::from_u128(low)`.

### W4 (LOW): encode_bytes reads only 8 of 31 bytes per chunk

**Location**: `native.rs:158-168`

**Impact**: Loses 74% of input data. Not on critical path.

### W5 (LOW): Keccak256 transcript doesn't implement Halo2 TranscriptWrite trait

**Location**: `transcript.rs:36-150`

**Impact**: Cannot use as drop-in replacement in `create_proof`/`verify_proof`.

### W6 (LOW): EvmProofFormat is dead code

**Location**: `format_spec.rs:133-141`

**Impact**: Declared but never used. Odd type signature.

---

## 6. Testing Assessment

| File | Tests | Coverage |
|------|-------|----------|
| evm.rs | ~24 | Generator output, VK data, EvmProof roundtrip, builder, PI encoding, pipeline |
| format_spec.rs | 11 | Fr roundtrip, structure parse, validation, hash pair, serialize |
| transcript.rs | 6 | Determinism, sequential challenges, prover/verifier agreement |
| native.rs | 4 | SerializedProof roundtrip, metadata, encoding, verifier |

**What is well-tested**: Byte encoding roundtrips, proof structure, Keccak256 challenges.
**What is not tested**: Generated Solidity compiled/executed, `encode_commitment` truncation, field order constants, EVM execution.

---

## 7. Health Score

**Grade: C+**

**Rationale**: The serialization layer (format_spec.rs, EvmProof, EvmPublicInputsArray) is solid. The Keccak256 transcript correctly mirrors Solidity. However, the generated Solidity verifier is a placeholder that cannot verify real proofs (W1), the swapped constants signal insufficient validation (W2), and native.rs has data truncation bugs (W3-W4). For the demo, the serialization layer is production-quality. The `SolidityGenerator` would need replacement with `halo2-solidity-verifier` for real on-chain verification, though the existing `Halo2Verifier.sol` contract handles this separately.
