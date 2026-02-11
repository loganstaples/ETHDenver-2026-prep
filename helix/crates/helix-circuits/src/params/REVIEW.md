# params/ -- Code Review

## Overview

The `params/` module provides infrastructure for HELIX's ZK proof system parameters: Structured Reference String (SRS) generation and caching, proving/verification key management, and circuit optimization/benchmarking estimation. It is organized into three submodules: `setup.rs` (SRS ceremony and parameter profiles), `keys.rs` (proving/verification key wrappers), and `optimization.rs` (constraint estimation and proof size calculation).

**Files:** 4 (mod.rs, setup.rs, keys.rs, optimization.rs)
**Total lines:** ~3,551

---

## Per-File Analysis

### mod.rs (131 lines)

Module root with extensive re-exports from all three submodules. Clean organization with grouped re-exports by category (SRS types, key types, optimization types). Uses `BenchmarkResult as OptBenchmarkResult` alias to avoid name conflicts.

### setup.rs (1,102 lines)

Manages SRS generation and caching:

| Type | Purpose |
|------|---------|
| `HelixSRS` | Metadata-only wrapper (does NOT contain actual SRS data) |
| `SRSMetadata` | Serialization header with HLXS magic bytes and version |
| `SRSCache` | Thread-safe in-memory + disk cache (`~/.cache/helix/srs/`) |
| `ParameterProfile` | Enum: Small(k=14), Medium(k=18), Large(k=22), ExtraLarge(k=24) |
| `PowersOfTauCeremony` | Simulated ceremony with SHA-256 proofs of knowledge |
| `FinalizedCeremony` | Result after ceremony finalization |
| `CeremonyContribution` | Individual participant's contribution |
| `SRSGenerationBenchmark` | Timing benchmarks for parameter generation |

Key method: `generate_params(k) -> ParamsKZG<Bn256>` (line 280) calls `ParamsKZG::<Bn256>::setup(k, OsRng)` for real KZG parameter generation. This is the only place where actual cryptographic parameters are created.

The `SRSCache` (lines 400-550) implements a two-tier caching strategy: in-memory `HashMap<u32, Arc<ParamsKZG<Bn256>>>` protected by `RwLock`, with disk persistence to `~/.cache/helix/srs/params_k{k}.bin`. Cache hits avoid the expensive `setup()` call.

### keys.rs (1,288 lines)

Manages proving and verification keys:

| Type | Purpose |
|------|---------|
| `HelixProvingKey` | Metadata descriptor (NOT actual pk data) |
| `HelixVerificationKey` | Metadata descriptor (NOT actual vk data) |
| `ProvingKeyMetadata` | Serialization header with HLXP magic |
| `VerificationKeyMetadata` | Serialization header with HLXV magic |
| `RealKeyBundle` | Wraps actual halo2 `ProvingKey<G1Affine>` + `VerifyingKey<G1Affine>` |
| `KeyBundle` | Metadata-only bundle (pk + vk descriptors) |
| `KeyCache` | Thread-safe cache for key bundles |
| `CircuitConfig` | Circuit dimensions (rows, columns, degree) |
| `CommitmentScheme` | Enum: KZG, IPA |
| `KeygenBenchmark` | Timing for keygen operations |

Key method: `RealKeyBundle::generate_real(params, circuit)` (lines 584-619) performs actual `keygen_vk` + `keygen_pk` using halo2's API. This is the real key generation path used by the prover pipeline.

**ISSUE** (lines 709-712): `HelixVerificationKey::commitment_points()` returns an empty `Vec<Vec<u8>>`. This placeholder means no commitment point extraction is available, which would be needed for on-chain verification key registration.

### optimization.rs (1,030 lines)

Circuit analysis and proof size estimation:

| Type | Purpose |
|------|---------|
| `CircuitAnalysis` | Constraint counts and column utilization |
| `CircuitOptimizer` | Simulated optimization passes |
| `OptimizationPass` | Enum of 6 pass types (Freivalds, Batching, etc.) |
| `OptimizationResult` / `OptimizationSummary` | Optimization output |
| `ProofSizeEstimator` | KZG BN254 proof size formula |
| `ModelConfig` | Neural network dimensions (demo_small, transformer_small, medium) |
| `CircuitBenchmarkSuite` | Benchmark harness for model architectures |

Key method: `ProofSizeEstimator::estimate(config)` (lines 700-750) computes proof size as `384 + 64 * (num_advice + 2) + 32 * num_public_inputs` bytes. For a typical HELIX circuit this yields ~800-1200 bytes, which is correct for SHPLONK proofs.

**CRITICAL ISSUE**: ALL optimization passes use hardcoded estimates, not actual circuit transformation. `CircuitOptimizer::apply_pass()` (lines 400-500) returns simulated savings percentages (e.g., "Freivalds reduces matmul by 80%") without modifying any circuit. The method `optimize()` (lines 350-400) sums these estimated savings. This is purely a planning/estimation tool, not an actual optimizer.

---

## Strengths

1. **Real SRS generation** (setup.rs:280): `generate_params` calls `ParamsKZG::<Bn256>::setup(k, OsRng)` -- this produces real cryptographic parameters suitable for production proofs.

2. **Two-tier SRS caching** (setup.rs:400-550): Memory + disk caching avoids repeated ~10-second SRS generation. The disk path `~/.cache/helix/srs/params_k{k}.bin` with manifest tracking is well-designed.

3. **Real key generation** (keys.rs:584-619): `RealKeyBundle::generate_real` performs actual `keygen_vk` + `keygen_pk`. This is correctly integrated into the prover pipeline.

4. **Correct proof size formula** (optimization.rs:700-750): The SHPLONK proof size estimate `384 + 64 * (advice_cols + 2) + 32 * public_inputs` is mathematically correct for BN254 KZG with 2 opening points.

5. **Parameter profiles** (setup.rs:200-250): `ParameterProfile::Small/Medium/Large/ExtraLarge` with appropriate k values (14/18/22/24) and recommended use cases. The `constraints_capacity()` method correctly returns `2^k - blinding_rows`.

6. **Serialization format versioning** (setup.rs:100-150, keys.rs:100-150): Magic bytes (`HLXS`, `HLXP`, `HLXV`) and format versions enable forward-compatible deserialization.

---

## Weaknesses

### W1: HelixSRS and HelixProvingKey/HelixVerificationKey are metadata-only wrappers
- **Location**: setup.rs:50-100, keys.rs:50-100, keys.rs:200-250
- **Impact**: HIGH -- These types look like they hold actual cryptographic data but contain only metadata (dimensions, creation time, hashes). Callers must separately manage the real `ParamsKZG<Bn256>`, `ProvingKey<G1Affine>`, and `VerifyingKey<G1Affine>`. This creates a confusing dual API: `HelixProvingKey` (metadata) vs `RealKeyBundle` (actual keys).
- **Fix**: Either merge metadata into the real key types (make `RealKeyBundle` the primary API and remove `HelixProvingKey`/`HelixVerificationKey` as standalone types) or clearly document the distinction with naming like `ProvingKeyDescriptor` instead of `HelixProvingKey`.

### W2: commitment_points() returns empty Vec
- **Location**: keys.rs:709-712
- **Impact**: MEDIUM -- On-chain VK registration requires extracting commitment points from the verification key. This placeholder blocks that workflow.
- **Fix**: Implement using `vk.fixed_commitments()` from halo2:
  ```rust
  pub fn commitment_points(&self, vk: &VerifyingKey<G1Affine>) -> Vec<Vec<u8>> {
      vk.fixed_commitments().iter()
          .map(|c| c.to_bytes().as_ref().to_vec())
          .collect()
  }
  ```

### W3: CircuitOptimizer is entirely simulated
- **Location**: optimization.rs:350-500
- **Impact**: MEDIUM -- The optimizer reports estimated savings without performing any actual circuit transformation. Users may expect `optimize()` to produce a more efficient circuit, but it only produces a report.
- **Fix**: Either rename to `CircuitAnalyzer` / `OptimizationEstimator` to set correct expectations, or implement actual optimization passes (at minimum, Freivalds verification replacement for matrix multiplications).

### W4: PowersOfTauCeremony is a single-process simulation
- **Location**: setup.rs:600-850
- **Impact**: LOW for demo, HIGH for production -- The ceremony simulates multi-party contributions in a single process. All contributions share the same trust domain. This is fine for development but provides zero security benefit over `ParamsKZG::setup(k, OsRng)`.
- **Fix**: Document clearly that this is a simulation. For production, integrate with existing Powers of Tau ceremony outputs (e.g., Hermez or EF ceremonies) via file import.

### W5: SRS disk cache has no integrity verification
- **Location**: setup.rs:450-500
- **Impact**: MEDIUM -- Cached SRS files on disk are loaded without verifying a checksum. A corrupted or tampered file would produce invalid parameters, leading to proof generation failures or security issues.
- **Fix**: Add SHA-256 checksum to the manifest and verify before loading:
  ```rust
  let hash = sha2::Sha256::digest(&bytes);
  if hash.as_slice() != manifest.expected_hash { return Err(SetupError::CorruptedCache) }
  ```

### W6: KeyCache uses String-based keys with no namespacing
- **Location**: keys.rs:900-1000
- **Impact**: LOW -- Cache keys are circuit type strings (e.g., "ml_training_v2"). No namespace prevents collisions between different circuit versions sharing the same name.
- **Fix**: Include a config hash in the cache key: `format!("{}_{}", circuit_type, hex::encode(&config_hash[..8]))`.

---

## Health Score: B

**Rationale**: The module successfully provides real SRS generation and key management through `ParamsKZG::setup()` and `RealKeyBundle::generate_real()`. The caching infrastructure is practical and well-tested. However, the confusing dual API between metadata wrappers and real key types (W1) is a design issue, the simulated optimizer (W3) is misleadingly named, and the empty `commitment_points()` (W2) blocks on-chain VK registration. For the demo, this module works well; for production, the metadata/real-key confusion and missing integrity checks need addressing.
