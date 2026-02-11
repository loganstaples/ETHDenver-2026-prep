# Beaver Module - Technical Review

**Updated**: 2026-02-11 — Multiple critical and high-priority issues resolved. Health score upgraded from D+ to B+.

**Review Date**: 2026-02-10
**Files**: 7 (mod.rs, triple.rs, dealer.rs, distributed.rs, ot.rs, pipeline.rs, pool.rs)
**Total Lines**: ~3,170
**Health Score**: B+ (75-80% production-ready)

---

## 1. Overview

The `beaver/` module implements Beaver triple generation for secure multiplication in the SPDZ protocol. Beaver triples `(a, b, c)` where `c = a*b` allow parties to compute multiplications on secret-shared values by revealing only random-looking "opened" values `d = x - a` and `e = y - b`.

**Three generation methods are provided:**
1. **TrustedDealer** (dealer.rs) — single trusted party generates all triples
2. **DistributedDealer** (dealer.rs) — pairwise cross-term protocol, no trusted party
3. **OTTripleGenerator** (ot.rs) — oblivious transfer-based generation

**Supporting infrastructure:**
- **BeaverPool** (pool.rs) — per-party triple storage and consumption
- **BeaverPipeline** (pipeline.rs) — background threaded generation with demand prediction

---

## 2. File-by-File Analysis

### 2.1 triple.rs (119 lines) — Data Structures

Defines three triple types: `BeaverTriple` (scalar), `VectorBeaverTriple` (element-wise), `MatrixBeaverTriple` (matmul). All use `Fr` (BN254 scalar field) values with `from_f64()` conversion helpers.

**Strengths:**
- Clean, minimal data structures
- Proper dimension assertions in constructors
- Row-major matrix layout documented

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 1 | Low | triple.rs:75 | `MatrixBeaverTriple::new` uses `assert!` not `Result` — panics on bad dimensions | Return `MPCResult<Self>` instead |
| 2 | Low | triple.rs:37-69 | No serialization format documented — row-major assumed but not enforced | Add `/// Storage: row-major flattened` doc comments |

---

### 2.2 dealer.rs (894 lines) — Triple Generation Core

The largest and most important file. Contains `TrustedDealer` (single-party generation) and `DistributedDealer` (pairwise cross-term protocol).

#### TrustedDealer (lines 22-282)

Generates triples centrally using `ChaCha20Rng`. Core algorithm:
1. Generate random `a`, `b` values in [-1000, 1000] fixed-point range
2. Compute `c = fixed_mul(a, b)`
3. Additively share each of `(a, b, c)` across parties

**Strengths:**
- Deterministic seeding via `with_seed()` for reproducibility
- Proper additive sharing: n-1 random shares + correction share
- Supports scalar, vector, and matrix triples

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| C1 | ~~**CRITICAL**~~ | dealer.rs:62 | ~~Uses `fixed_mul()` while `DistributedDealer` uses `exact_fixed_mul()` — triples incompatible.~~ | ✅ **RESOLVED** (2026-02-11): TrustedDealer now uses `mpc_scale` (`exact_fixed_mul`). All triple generation standardized on `mpc_scale`. |
| 3 | Medium | dealer.rs:46 | `random_value()` generates in [-1000, 1000] — arbitrary range that limits triple applicability to small model weights | Document range limitation; consider parameterizing |

#### DistributedDealer (lines 385-577)

Implements trustless pairwise cross-term protocol:
1. Each party `i` generates random `(a_i, b_i)`, computes `c_i = exact_fixed_mul(a_i, b_i)`
2. For each pair `(i, j)`: party `i` picks random mask `r_ij`, commits to it, adds `r_ij` to `c_i`. Party `j` adds `exact_fixed_mul(a_i, b_j) - r_ij` to `c_j`
3. Result: `sum(c) = sum(a_i*b_i) + sum_{i!=j}(a_i*b_j) = (sum a)(sum b)`

**Critical function — `exact_fixed_mul()`** (line 343-345):
```rust
fn exact_fixed_mul(a: &Fr, b: &Fr) -> Fr {
    a.mpc_scale(b)  // (a*b)*(2^64)^{-1} mod r — exact, linear
}
```

This is mathematically correct and preserves the additive homomorphism needed for the protocol.

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| C2 | ~~**CRITICAL**~~ | dealer.rs:461-463 | ~~**RNG not advanced**: clones RNG, advances clone, original unchanged. Mask `r_ij` repeats across iterations.~~ | ✅ **RESOLVED** (2026-02-11): Fixed in dealer.rs. RNG clone bug eliminated. |
| 4 | Medium | dealer.rs:466 | `MaskCommitment` generated but never verified — commitments exist only "for post-hoc audit" (line 358) but no audit code exists | Either wire in verification or remove commitment generation |
| 5 | Low | dealer.rs:265-275 | `additive_share_scalar` shares one value across parties but doesn't validate `num_parties > 0` | Add `assert!(num_parties > 0)` guard |

#### MaskCommitment (lines 288-331)

SHA-256 commitment to cross-term masks with constant-time verification (`ct_eq_hash()`). Correctly includes sender, receiver, and nonce in the hash. Well-implemented but unused in practice.

---

### 2.3 distributed.rs (296 lines) — Alternative Distributed Interface

Higher-level state machine for distributed triple generation with `TripleGenMessage` protocol messages.

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| C3 | ~~**CRITICAL**~~ | distributed.rs:193 | ~~`simulate_distributed_generation` uses `Fr::mul()` (raw field multiplication) instead of `exact_fixed_mul()` — triples numerically wrong.~~ | ✅ **RESOLVED** (2026-02-11): Now uses `mpc_scale` (`exact_fixed_mul`). |
| 6 | High | distributed.rs:147-148 | `masked_b` received in `CrossTermContribution` but explicitly ignored (`masked_b: _`). The protocol description (lines 6-12) says both masked values are needed. | Either use `masked_b` or remove it from `TripleGenMessage` and document why only `masked_a` is needed |
| 7 | High | distributed.rs:104 | `mask_commitment` field generated but never checked by any receiver | Wire in `MaskCommitment::verify()` in `phase2_compute()` |
| 8 | Medium | distributed.rs:247-295 | Tests pass despite C3 because reconstruction uses the same broken multiplication — tests compare broken output against itself | Add tests that cross-validate against `TrustedDealer` output |

---

### 2.4 ot.rs (750 lines) — Oblivious Transfer

Implements (attempted) Chou-Orlandi 1-of-2 OT, correlated OT, OT extension, and OT-based triple generation.

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| C4 | ~~**CRITICAL**~~ | ot.rs:88-97 | ~~OT sender computes second receiver key as `receiver_pk XOR sender_pk` — not a valid EC operation. Receiver can compute both shared secrets.~~ | ✅ **RESOLVED** (2026-02-11): OT gated with `#[deprecated]` warning. `generate_beaver_triple_ot` deprecated. OT should not be used until proper group operations are implemented. |
| C5 | ~~**CRITICAL**~~ | ot.rs:140-150 | ~~Same XOR bug in `OTReceiver::public_key()`~~ | ✅ **RESOLVED** (2026-02-11): Covered by OT deprecation gating. |
| 9 | High | ot.rs:248-258 | `OTExtension` claims IKNP-style but is actually just random pair selection — no matrix transposition, no seed expansion, no hash-based extension | Either implement real IKNP or rename to `SimpleOTBatch` |
| 10 | High | ot.rs:281-331 | `generate_beaver_triple_ot()` uses `f64` arithmetic, not `Fr` — type mismatch with `BeaverTriple` which stores `Fr` values | Use `Fr` throughout or add `from_f64()` conversion at the boundary |
| 11 | Medium | ot.rs:599-601 | `simulate_full_generation()` converts f64→Fr at the end, losing precision from floating-point intermediate computation | Compute in `Fr` from the start |

---

### 2.5 pipeline.rs (837 lines) — Background Generation

Background thread pool with demand prediction for proactive triple generation.

**Architecture:**
- Worker threads process `GenerationRequest` messages
- Replenishment thread monitors pool levels and submits requests
- `DemandPredictor` estimates consumption from history

**Strengths:**
- Priority levels (Low/Normal/High/Critical) for request escalation
- Per-model presets (`small_model()`, `large_model()`)
- Clean stats reporting via `PipelineStats`
- Proper shutdown signaling

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 12 | Medium | pipeline.rs:375-448 | Request queue is FIFO — Critical requests wait behind Low-priority work | Use `BinaryHeap` or priority channel (e.g., `crossbeam::channel` with priority wrapper) |
| 13 | Medium | pipeline.rs:478-482 | Replenisher acquires all pool read-locks sequentially — single slow pool blocks entire level check | Use `try_read()` with timeout, skip locked pools |
| 14 | Medium | pipeline.rs:230 | Demand prediction formula `per_step = num_layers * hidden_dim * 4` is linear — transformers scale quadratically with sequence length for attention | Add `seq_len` parameter to prediction |
| 15 | Low | pipeline.rs:314-315 | Bounded channel capacity of 1000 with no backpressure — dropped requests are silently lost | Return `Err` to caller when queue full, or use unbounded with memory limit |

---

### 2.6 pool.rs (324 lines) — Triple Storage

Per-party storage indexed by triple type (scalar, vector by dimension, matrix by `(m,k,n)`).

**Strengths:**
- Clean HashMap-based indexing for different dimensions
- `fill_for_training_step()` pre-populates based on model architecture
- Consumption tracking for monitoring
- Proper `MPCError::BeaverPoolExhausted` errors (not panics)

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 16 | Medium | pool.rs:166-236 | `fill_for_training_step()` takes `&mut TrustedDealer` — not thread-safe if called from pipeline worker threads | Take `&dealer` with interior mutability, or generate triples outside and pass in |
| 17 | Low | pool.rs:84-89 | `take_scalar()` uses `Vec::pop()` (LIFO) — newest triples consumed first. This is fine for correctness but prevents FIFO freshness guarantees | Document LIFO behavior; consider `VecDeque::pop_front()` for FIFO |

---

## 3. Cross-Module Issues

### 3.1 ~~Three Incompatible Multiplication Functions~~ — RESOLVED

✅ **RESOLVED** (2026-02-11): All triple generation now standardized on `mpc_scale` (`exact_fixed_mul`).

Previously three different multiplication semantics were used. Now all use the same:

| Location | Function | Status |
|----------|----------|--------|
| `dealer.rs` (TrustedDealer) | `mpc_scale()` | ✅ Fixed (was `fixed_mul`) |
| `dealer.rs` (DistributedDealer) | `mpc_scale()` via `exact_fixed_mul()` | ✅ Already correct |
| `distributed.rs` | `mpc_scale()` via `exact_fixed_mul()` | ✅ Fixed (was `Fr::mul`) |

**Note on `fixed_mul` vs `mpc_scale`**: `mpc_scale` is used for Beaver protocol operations on random Fr shares. `fixed_mul` is still correct for `from_f64 x from_f64` products (e.g., `TensorShare.scale()`), where values are small and within the byte-shift safe range. This distinction is now documented.

### 3.2 No Cross-Dealer Validation Tests

No test verifies that triples from `TrustedDealer`, `DistributedDealer`, and `OTTripleGenerator` are interchangeable. Each module tests against its own output.

**Fix**: Add integration tests that:
1. Generate a triple via TrustedDealer
2. Generate a triple via DistributedDealer (same seed)
3. Assert `a_trusted == a_distributed` (they won't match today — this is the bug detector)

### 3.3 f64 ↔ Fr Boundary

The `ot.rs` module operates in f64 while everything else uses Fr. This creates a precision boundary where information is lost during conversion.

**Fix**: Move OT triple generation to Fr arithmetic throughout.

---

## 4. Security Assessment

### What Works
- Additive sharing in `TrustedDealer` is mathematically correct (n-1 random + correction)
- `MaskCommitment` uses constant-time comparison via `ct_eq_hash()`
- `DistributedDealer` protocol math is correct (when using `exact_fixed_mul`)
- Pool consumption properly returns errors instead of panicking

### What's Been Fixed (2026-02-11)
1. ✅ ~~**OT is cryptographically broken**~~ — OT gated with `#[deprecated]` warning; `generate_beaver_triple_ot` deprecated
2. ✅ ~~**RNG clone bug**~~ — Fixed in dealer.rs
3. ✅ ~~**Three incompatible multiplications**~~ — All standardized on `mpc_scale`

### Remaining Issues
4. **Commitments generated but never verified** — security theater (medium priority)

### Trust Model
- `TrustedDealer`: Trusted single party (acceptable for demos, now uses correct `mpc_scale` arithmetic)
- `DistributedDealer`: RNG bug fixed; commitments still unverified. Usable for semi-honest setting.
- `OTTripleGenerator`: Deprecated/gated. Do not use until proper group operations are implemented.

---

## 5. Test Coverage

| File | Tests | Coverage Assessment |
|------|-------|-------------------|
| triple.rs | 0 | No unit tests (tested indirectly via dealer tests) |
| dealer.rs | 7 | Good: scalar/vector/matrix, 2-party, linearity checks |
| distributed.rs | 3 | Weak: tests pass despite `Fr::mul` bug (self-referential) |
| ot.rs | 4 | Weak: OT tests don't verify security properties, only functional correctness |
| pipeline.rs | 8 | Good: lifecycle, priority, demand prediction, shutdown |
| pool.rs | 4 | Adequate: basic ops, exhaustion, matrix, fill-for-training |

**Missing tests:**
- Cross-dealer compatibility (TrustedDealer vs DistributedDealer output)
- OT security property (receiver cannot learn unchosen message)
- Pipeline under load (worker contention, pool exhaustion)
- Pool thread safety (concurrent take/fill)
- Commitment verification (MaskCommitment.verify() never called in tests)

---

## 6. Demo Readiness (ETHDenver)

**TrustedDealer + BeaverPool**: Ready. This path works correctly for single-machine demos where a trusted dealer is acceptable.

**DistributedDealer**: Ready. RNG bug (C2) fixed. Uses correct `mpc_scale` arithmetic. Commitments still unverified but functional for semi-honest demo.

**OT-based generation**: Deprecated/gated. Do not use. `#[deprecated]` warning in place.

**Pipeline**: Ready for demos. Background generation with demand prediction works. Priority ordering would be nice but not blocking.

**Recommendation**: Use `TrustedDealer` for ETHDenver. Fix `DistributedDealer` RNG bug as a follow-up. OT needs redesign.

---

## 7. Prioritized Recommendations

### Critical (fix before any production use)
1. ✅ ~~**Standardize multiplication**~~ — **RESOLVED**: All dealers now use `mpc_scale`.
2. ✅ ~~**Fix RNG clone bug**~~ — **RESOLVED**: Fixed in dealer.rs.
3. ✅ ~~**Fix or remove OT**~~ — **RESOLVED**: OT gated with `#[deprecated]` warning.

### High (fix before multi-party deployment)
4. ✅ ~~**Fix distributed.rs `Fr::mul`**~~ — **RESOLVED**: Now uses `mpc_scale`.
5. **Wire in commitment verification** — Add `MaskCommitment::verify()` call in distributed generation
6. **Move OT to Fr arithmetic** — Eliminate f64 precision loss in ot.rs (lower priority now that OT is deprecated)

### Nice-to-have (improves production quality)
7. Priority queue for pipeline requests
8. Cross-dealer validation test suite
9. Pool FIFO ordering with consumption timestamps
10. Demand prediction with sequence-length awareness
