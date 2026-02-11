# params/ -- Code Review

**Module**: `helix-circuits/src/params/`
**Reviewed**: 2026-02-11 (updated; original 2026-02-10)
**Files**: mod.rs (130 lines), setup.rs (1,101 lines), keys.rs (1,287 lines)
**Total lines**: ~2,518

---

## 1. Overview

The `params/` module provides infrastructure for HELIX's ZK proof system parameters: Structured Reference String (SRS) generation and caching, and proving/verification key management. It is organized into two submodules: `setup.rs` (SRS ceremony and parameter profiles) and `keys.rs` (proving/verification key wrappers).

The previous `optimization.rs` (1,030 lines of simulated optimization passes) has been **deleted** as dead code during production hardening Stage 3.

---

## 2. Per-File Analysis

### mod.rs (130 lines)

Module root with extensive re-exports from both submodules. Clean organization with grouped re-exports by category (SRS types, key types). Previously re-exported optimization types; those exports have been removed.

### setup.rs (1,101 lines)

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

The `SRSCache` (lines 400-550) implements a two-tier caching strategy: in-memory `HashMap<u32, Arc<ParamsKZG<Bn256>>>` protected by `RwLock`, with disk persistence to `~/.cache/helix/srs/params_k{k}.bin`.

### keys.rs (1,287 lines)

Manages proving and verification keys:

| Type | Purpose |
|------|---------|
| `HelixProvingKey` | Metadata descriptor (NOT actual pk data) |
| `HelixVerificationKey` | Metadata descriptor (NOT actual vk data) |
| `RealKeyBundle` | Wraps actual halo2 `ProvingKey<G1Affine>` + `VerifyingKey<G1Affine>` |
| `KeyBundle` | Metadata-only bundle (pk + vk descriptors) |
| `KeyCache` | Thread-safe cache for key bundles |
| `CircuitConfig` | Circuit dimensions (rows, columns, degree) |
| `CommitmentScheme` | Enum: KZG, IPA |
| `KeygenBenchmark` | Timing for keygen operations |

Key method: `RealKeyBundle::generate_real(params, circuit)` (lines 584-619) performs actual `keygen_vk` + `keygen_pk` using halo2's API.

---

## 3. Deleted File

### ~~optimization.rs (1,030 lines)~~ DELETED

Was a framework of simulated optimization passes (`CircuitOptimizer`, `OptimizationPass`, `ProofSizeEstimator`, `CircuitBenchmarkSuite`). All optimization passes used hardcoded estimates without actual circuit transformation. `CircuitOptimizer::apply_pass()` returned simulated savings percentages. Deleted in production hardening as it was never wired into the actual circuit pipeline.

---

## 4. Strengths

1. **Real SRS generation** (setup.rs:280): `generate_params` calls `ParamsKZG::<Bn256>::setup(k, OsRng)` -- produces real cryptographic parameters.

2. **Two-tier SRS caching** (setup.rs:400-550): Memory + disk caching avoids repeated ~10-second SRS generation. Disk path `~/.cache/helix/srs/params_k{k}.bin` with manifest tracking.

3. **Real key generation** (keys.rs:584-619): `RealKeyBundle::generate_real` performs actual `keygen_vk` + `keygen_pk`. Correctly integrated into the prover pipeline.

4. **Parameter profiles** (setup.rs:200-250): `ParameterProfile::Small/Medium/Large/ExtraLarge` with appropriate k values (14/18/22/24). `constraints_capacity()` correctly returns `2^k - blinding_rows`.

5. **Serialization format versioning** (setup.rs:100-150, keys.rs:100-150): Magic bytes (`HLXS`, `HLXP`, `HLXV`) and format versions enable forward-compatible deserialization.

6. **Clean module** -- 1,030 lines of dead optimization code removed.

---

## 5. Weaknesses

### W1: HelixSRS and HelixProvingKey/HelixVerificationKey are metadata-only wrappers (HIGH)
- **Location**: setup.rs:50-100, keys.rs:50-100
- **Impact**: These types look like they hold actual cryptographic data but contain only metadata. Callers must separately manage real `ParamsKZG`, `ProvingKey`, `VerifyingKey`. Creates confusing dual API.
- **Fix**: Merge metadata into real key types or rename to `ProvingKeyDescriptor`.

### W2: commitment_points() returns empty Vec (MEDIUM)
- **Location**: keys.rs:709-712
- **Impact**: On-chain VK registration requires extracting commitment points. Placeholder blocks this.
- **Fix**: Implement using `vk.fixed_commitments()`.

### W3: PowersOfTauCeremony is a single-process simulation (LOW for demo)
- **Location**: setup.rs:600-850
- **Impact**: Simulates multi-party contributions in a single process. Zero security benefit over `ParamsKZG::setup()`.
- **Fix**: Document clearly as simulation. For production, import from existing ceremony outputs (Hermez, EF).

### W4: SRS disk cache has no integrity verification (MEDIUM)
- **Location**: setup.rs:450-500
- **Impact**: Cached SRS files loaded without checksum verification. Corrupted or tampered file would produce invalid parameters.
- **Fix**: Add SHA-256 checksum to manifest and verify before loading.

### W5: KeyCache uses String-based keys with no namespacing (LOW)
- **Location**: keys.rs:900-1000
- **Impact**: Cache key collisions possible between different circuit versions sharing the same name.
- **Fix**: Include config hash in cache key.

---

## 6. Health Score: B

**Rationale**: The module successfully provides real SRS generation and key management through `ParamsKZG::setup()` and `RealKeyBundle::generate_real()`. The caching infrastructure is practical and well-tested. The dead optimization code (1,030 lines) has been removed, improving clarity. The confusing dual API between metadata wrappers and real key types (W1) is the main design issue. For the demo, this module works well; for production, the metadata/real-key confusion and missing integrity checks need addressing.
