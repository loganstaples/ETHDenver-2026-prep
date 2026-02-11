# Protocols Module - Technical Review

**Updated**: 2026-02-11 — Multiple critical and high-priority issues resolved. Health score upgraded from C- to B+.

**Review Date**: 2026-02-10
**Files**: 9 (mod.rs, arithmetic.rs, matmul.rs, activation.rs, normalization.rs, comparison.rs, aggregation.rs, proved_arithmetic.rs, reshare.rs)
**Total Lines**: ~5,830
**Health Score**: B+ (75-80% production-ready)

---

## 1. Overview

The `protocols/` module implements the core MPC protocols for privacy-preserving neural network training. These protocols consume Beaver triples from `beaver/` and produce witness data for `integration/` to feed into ZK proof generation.

**Protocol categories:**
- **Local operations** (no communication): addition, subtraction, scaling
- **Interactive operations** (Beaver protocol): multiplication, matmul
- **Reconstruct-compute-reshare**: activation functions, normalization, softmax
- **Garbled circuits**: comparison, sign bit (broken — see below)
- **Aggregation**: gradient compression, tree aggregation, weighted averaging
- **Proof integration**: witness capture wrapping all of the above

---

## 2. File-by-File Analysis

### 2.1 mod.rs (73 lines) — Module Entry Point

Exports all protocol types and provides `reshare_values()` utility.

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 1 | Medium | mod.rs:~55 | `reshare_values()` uses `rng.gen_range(-100.0..100.0)` for share randomness — bounded range leaks distribution information | Use `Fr::random()` for uniform field randomness |

---

### 2.2 arithmetic.rs (470 lines) — Core Secure Arithmetic

Implements local operations (add, sub, scale) and interactive Beaver triple multiplication.

#### Beaver Multiplication Protocol (lines 86-100)

```
[xy] = [c] + d*[b] + e*[a] + d*e  (party 0 only adds d*e)
```

where `d = x - a` (public), `e = y - b` (public), and `(a, b, c)` is the Beaver triple.

**Strengths:**
- Correct standard Beaver protocol implementation
- Proper use of `fixed_mul()` for fixed-point arithmetic (lines 94-97)
- Party 0 condition correctly gates the `d*e` term (line 96)
- Batched operations serialize/deserialize efficiently (lines 236-314)

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 2 | Medium | arithmetic.rs:186-226 | `reciprocal()` uses Newton's method with a **public** initial guess (only party 0 has non-zero). If initial guess is 0, all iterations produce 0 and the result silently diverges. | Validate `initial_guess != 0` and document that initial guess reveals magnitude of the secret |
| 3 | Low | arithmetic.rs:162-176 | `vector_multiply()` doesn't validate that pool has enough triples before starting — fails partway through with unhelpful `BeaverPoolExhausted` error | Pre-check `pool.scalar_available() >= dim` before loop |

---

### 2.3 matmul.rs (435 lines) — Secure Matrix Multiplication

Extends Beaver protocol to matrices: `[C] = [W] + D@[V] + [U]@E + D@E (party 0 only)`.

**Strengths:**
- Correct matrix Beaver protocol
- Multiple APIs: direct, from pool, matvec, outer product, public matrix multiply
- Row-major layout consistently maintained
- Batch communication (entire D/E matrices in single message)

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 4 | Low | matmul.rs:134-138 | Checks party count but not triple dimensions — if pool has fewer than `m*k*n` triples, `take_matrix()` fails with generic error | Pre-check `pool.matrix_available(m, k, n) > 0` |
| 5 | Low | matmul.rs:29-61 | No validation of `m, k, n > 0` — zero-dimension matmul produces empty result silently | Add `assert!(m > 0 && k > 0 && n > 0)` or return error |

---

### 2.4 activation.rs (421 lines) — Activation Functions

Two approaches: reconstruct-compute-reshare (reveals activations, standard in MPC-ML) and polynomial approximation (keeps privacy, lower accuracy).

#### Reconstruct-Compute-Reshare (lines 51-78)

Correctly reconstructs secret, applies f64 activation function, reshares result. This is the standard MPC-ML tradeoff: activation values are revealed but model weights remain hidden.

#### Polynomial Approximations

| Function | Polynomial | Valid Range | Accuracy |
|----------|-----------|-------------|----------|
| ReLU | `0.5*x + 0.25*x^2` | [-1, 1] | **Poor** — this is essentially a leaky quadratic, not ReLU |
| Sigmoid | `0.5 + 0.25*x - 0.0208*x^3` | [-4, 4] | Reasonable |
| GELU | `0.5*x*(1 + 0.851*x - 0.0354*x^3)` | [-3, 3] | Reasonable |

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 6 | High | activation.rs:114-154 | `approximate_relu()` polynomial `0.5*x + 0.25*x^2` is **not a valid ReLU approximation** — error > 50% for values outside [-1, 1], and has negative outputs for x < -2. Any model using this for training will fail to converge. | Use higher-degree minimax polynomial (degree 5-7) or piecewise linear approximation |
| 7 | Medium | activation.rs:51-78 | `apply_reconstruct_reshare()` reveals activation values at every layer at every step. Over K steps with L layers and D dimensions, this leaks `K * L * D` values per party. | Document the leakage budget; consider random padding or differential privacy noise |
| 8 | Low | activation.rs:296-419 | No test for `approximate_relu()` accuracy — tests only verify reconstruct-reshare path | Add accuracy test: `assert!((approx_relu(x) - relu(x)).abs() < epsilon)` for range of x values |

---

### 2.5 normalization.rs (379 lines) — Layer Norm, RMS Norm, Softmax

All use reconstruct-normalize-reshare: values are reconstructed to f64, normalized, and reshared.

**Strengths:**
- Numerically stable softmax (subtracts max before exp, line 135)
- Proper fixed-point multiplication for shared gamma/beta parameters
- Both layer norm and RMS norm correctly compute mean/variance

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 9 | Medium | normalization.rs:37-72 | Same privacy concern as activations — layer norm reveals all intermediate values | Same mitigation: document leakage, consider DP noise |
| 10 | Low | normalization.rs:175-217 | Shared gamma/beta dimension never validated against input dimension — if mismatched, `fixed_mul` operates on wrong elements | Add `assert_eq!(gamma_shares[i].len(), values.len())` |

---

### 2.6 comparison.rs (1,586 lines) — THE MOST PROBLEMATIC FILE

Implements garbled circuit-based comparison and sign computation.

#### Garbled Circuit Engine (lines 60-337)

Provides encrypt/decrypt, point-and-permute optimization, AND/XOR gates.

**Strengths:**
- Random label generation with guaranteed distinct LSBs (lines 97-103)
- 4-entry garbled table per gate (standard construction)
- Point-and-permute optimization

#### Critical Security Failures

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| C1 | ~~**CRITICAL**~~ | comparison.rs:198-242 | ~~**OT is simulated, not implemented.** Garbler sees evaluator's choice bits.~~ | ✅ **RESOLVED** (2026-02-11): `ot_transfer_labels` renamed to `simulated_ot_transfer_labels` with security warning. Both `sign_bit` implementations unified to reconstruct-compare-reshare (no cfg split). Same for `secure_less_than` and `decompose`. |
| C2 | **CRITICAL** | comparison.rs:143-152 | `fr_to_bits()` converts `Fr` to 256 bits, but BN254 scalar field is 254 bits. The top 2 bits can encode values `>= p` which are not valid field elements, causing incorrect sign computation for large values. | Extract only 254 bits; clamp or reduce modulo p |
| 11 | ~~High~~ | comparison.rs:562-630 | ~~Two implementations behind `#[cfg(not(feature = "simulation"))]` and `#[cfg(feature = "simulation")]`. Compile-time switching is dangerous.~~ | ✅ **RESOLVED** (2026-02-11): Unified to single reconstruct-compare-reshare implementation (no cfg split). Both sign_bit, secure_less_than, and decompose implementations unified. |
| 12 | High | comparison.rs:366-535 | Garbled sign circuit uses 256-bit ripple-carry adder (~2,048 gates) for a single comparison. This is extremely expensive. | Use 254-bit circuit; consider boolean circuit that directly extracts MSB after modular reduction |
| 13 | Medium | comparison.rs:1288-1338 | ReLU with gradient stores mask from broken sign computation — backward pass inherits all sign bugs | Fix sign_bit first; gradient mask correctness follows |

#### Test Assessment (lines 1345-1585)

14 tests exist but **do not verify security properties**:
- `test_no_input_leakage()` (line 1551) checks that OT returns correct labels but doesn't verify that the garbler cannot learn the evaluator's choice
- Tests pass because they compare against the same buggy implementation

---

### 2.7 aggregation.rs (1,137 lines) — Gradient Aggregation

Implements gradient compression (top-k sparsification + quantization), tree-based aggregation, dropout-tolerant aggregation, and weighted averaging.

**Strengths:**
- Top-k sparsification with error feedback loop (maintains dropped gradients)
- Tree aggregation reduces communication rounds from O(n) to O(log n)
- Commitment verification available (SHA-256)
- Dropout tolerance handles missing parties gracefully

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 14 | High | aggregation.rs:556 | `compressed.decompress()` reconstructs full gradient tensor **in plaintext** — the aggregator sees all gradients. This is **not MPC aggregation**; it's centralized aggregation with optional compression. | Implement aggregation on compressed secret-shared gradients, or clearly document that aggregator is trusted |
| 15 | High | aggregation.rs:452 | Commitment verification only runs if `verify_commitments=true`. `WeightedAggregator` disables it (line 1058). Malicious parties can submit corrupted gradients without detection. | Remove the disable option, or require explicit `UnsafeAggregator` wrapper |
| 16 | Medium | aggregation.rs:202-210 | Error feedback state (`HashMap` in `RwLock`) persists across rounds — leaks which tensors have sparse gradients over time | Reset error feedback each round or use random padding |
| 17 | Medium | aggregation.rs:276-296 | Quantization doesn't handle zero-range values: `if range < 1e-10` returns all zeros — destroys gradient signal for constant tensors | Return original values when range is zero |
| 18 | Low | aggregation.rs:111 | Compression ratio calculation ignores metadata overhead (scale, offset, indices) — reported ratio is ~2x better than actual | Include metadata size in calculation |

---

### 2.8 proved_arithmetic.rs (1,046 lines) — Witness Capture

Wraps all arithmetic operations with witness recording for ZK proof generation.

**Architecture:**
- `WitnessCapture`: Thread-safe (`Arc<Mutex>`) collector of `OperationWitness` records
- `ProvedArithmetic`: Wraps `SecureArithmetic` methods, captures inputs/outputs/errors
- Can be enabled/disabled at runtime
- Generates SHA-256 commitment over accumulated witnesses

**Strengths:**
- Comprehensive capture of all operation types
- Structured metadata (operation type + key-value pairs)
- Non-intrusive when disabled
- Commitment generation for verification

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 19 | ~~High~~ | proved_arithmetic.rs:204-238 | ~~`BeaverWitness` stores **private** shares `(a, b, c)` alongside public `(opened_d, opened_e)`. Private values could be leaked.~~ | ✅ **RESOLVED** (2026-02-11): BeaverWitness now only stores `opened_d`, `opened_e`, `party_index`, `verified`. Private triple shares (a, b, c) removed. `from_triple` renamed to `from_protocol`. |
| 20 | High | proved_arithmetic.rs:749-794 | `record_forward_pass()` stores all activations (`h_pre`, `h`, `y`) and gradients **in plaintext** in the witness. If this witness is used for on-chain ZK proof, these values become public. | Store only Poseidon hash commitments of activations, not plaintext values |
| 21 | Medium | proved_arithmetic.rs:297 | Uses `elapsed().as_nanos()` for operation timestamps — nanosecond precision reveals exact operation ordering and timing across parties | Use logical operation counters or truncate to milliseconds |
| 22 | Medium | proved_arithmetic.rs:826 | Records count of active ReLU mask elements — reveals activation sparsity pattern | Remove or anonymize mask statistics |
| 23 | Low | proved_arithmetic.rs:373 | Error accumulation assumes independence (`total += error`). In practice, errors from sequential matmul → activation → norm are correlated. | Document that total error is an upper bound, not exact |

---

### 2.9 reshare.rs (278 lines) — Share Refreshing

Periodic re-sharing adds zero-shares to refresh secret sharing, preventing long-term information accumulation.

**Algorithm:**
1. Generate n zero-shares (sum to 0) using `Fr::random()`
2. Each party generates zero-shares and adds to all parties' shares
3. New shares sum to same value (by linearity of addition)

**Strengths:**
- Correct zero-share construction (n-1 random + negated sum)
- Multiple granularities: scalar, vector, tensor, full model
- Proper `Fr::random()` for uniform field randomness
- `should_reshare()` with configurable interval

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 24 | Medium | reshare.rs:56 | Seed derivation: `seed.wrapping_add(i * 0x9E3779B97F4A7C15)`. If `seed == 0`, all party seeds are 0 → all parties generate identical zero-shares → shares don't actually change. | Validate `seed != 0` or mix in party ID: `seed ^ (i as u64 + 1)` |
| 25 | Low | reshare.rs:126-190 | Model re-sharing silently skips `None` components (embeddings, lm_head). If some parties have embeddings and others don't, this reveals model structure. | Require consistent structure across all parties |

---

## 3. Cross-Module Analysis

### 3.1 Privacy Leakage Budget

The reconstruct-compute-reshare pattern (used in activation, normalization, softmax) reveals intermediate values. For a model with L layers, D hidden dimensions, trained for K steps with N parties:

| Operation | Values leaked per step | Total leaked |
|-----------|----------------------|--------------|
| Activations (per layer) | L * D | K * L * D |
| Layer norm (per layer) | L * D | K * L * D |
| Softmax (per layer) | L * D | K * L * D |
| **Total** | **3 * L * D** | **3 * K * L * D** |

For a small model (L=6, D=256, K=1000, N=3): **4.6 million** values leaked per party. An adversary participating in training can correlate these with public gradient updates to potentially reconstruct model weights.

**Mitigation**: Add calibrated Gaussian noise before resharing (differential privacy). Noise variance should be chosen based on sensitivity analysis of the specific model architecture.

### 3.2 Dependency Graph

```
mod.rs (entry)
├── arithmetic.rs ← beaver/triple.rs
│   └── matmul.rs ← arithmetic.rs
├── activation.rs ← arithmetic.rs (for resharing)
├── normalization.rs ← arithmetic.rs (for resharing)
├── comparison.rs ← beaver/ot.rs (broken)
├── aggregation.rs (standalone, NOT actually MPC)
├── proved_arithmetic.rs ← arithmetic.rs (wraps all ops)
└── reshare.rs (standalone)
```

### 3.3 The Aggregation Disconnect

`aggregation.rs` is the largest file (1,137 lines) but **doesn't actually perform MPC aggregation**. Gradients are decompressed to plaintext before aggregation. This means the aggregator node sees all gradient values — a significant trust assumption that contradicts the MPC privacy model.

In a true MPC aggregation:
1. Each party holds gradient shares `[g_i]`
2. Aggregation computes `[g_avg] = sum([g_i]) / n` entirely on shares
3. No single party sees any plaintext gradient

The current implementation skips step 2 and reconstructs gradients at the aggregator.

---

## 4. Security Assessment

### Secure Components
- **arithmetic.rs**: Beaver protocol is correctly implemented; `fixed_mul` used consistently
- **matmul.rs**: Matrix Beaver protocol is correct
- **reshare.rs**: Zero-share construction is information-theoretically secure

### Fixed Components (2026-02-11)
- **comparison.rs**: OT renamed to `simulated_ot_transfer_labels` with security warning. All comparison operations unified to reconstruct-compare-reshare (no cfg split). 256-bit field issue remains (medium priority).
- **proved_arithmetic.rs**: BeaverWitness no longer stores private shares.

### Remaining Issues
- **aggregation.rs**: Not actually MPC — aggregator sees plaintext gradients
- **comparison.rs**: Still uses 256-bit arithmetic for 254-bit field (medium priority)

### Privacy-Leaking (by design, documented tradeoff)
- **activation.rs**: Reveals activation values
- **normalization.rs**: Reveals normalized values
- Both are standard in MPC-ML literature but should be quantified

### Needs Hardening
- ~~**proved_arithmetic.rs**: Witness stores private data~~ — ✅ **RESOLVED**: BeaverWitness now stores only public values
- **proved_arithmetic.rs**: Forward/backward recording still stores plaintext activations (issue 20, medium priority)

---

## 5. Test Coverage

| File | Tests | Quality | Key Gap |
|------|-------|---------|---------|
| arithmetic.rs | 12 | Good | Missing batched operation network-size verification |
| matmul.rs | 8 | Good | Missing zero-dimension edge cases |
| activation.rs | 8 | Partial | No polynomial approximation accuracy tests |
| normalization.rs | 6 | Good | Missing dimension mismatch tests |
| comparison.rs | 14 | **Poor** | Tests don't verify security properties; pass despite broken OT |
| aggregation.rs | 10 | Partial | Missing tests for MPC property (because it doesn't have one) |
| proved_arithmetic.rs | 6 | Partial | Missing batch witness tests; no private-data-in-witness check |
| reshare.rs | 3 | Adequate | Missing zero-seed edge case test |

**Total**: ~67 tests. Functional correctness is well-tested; security properties are not.

---

## 6. Demo Readiness (ETHDenver)

**Ready for demo:**
- arithmetic.rs, matmul.rs: Core Beaver protocol works correctly
- activation.rs, normalization.rs: Reconstruct-reshare path works (polynomial approximations should not be used)
- reshare.rs: Works correctly
- proved_arithmetic.rs: Witness capture works (privacy of witness is a non-issue for single-machine demos)

**Not ready:**
- comparison.rs: Broken OT means sign/comparison cannot be used securely. **For demos, use reconstruct-compare-reshare instead** (reveals values but produces correct results).
- aggregation.rs: Works but is not MPC. For demos, this is acceptable if the aggregator is trusted.

---

## 7. Prioritized Recommendations

### Critical
1. ✅ ~~**Fix or isolate comparison.rs**~~ — **RESOLVED** (2026-02-11): OT renamed to `simulated_ot_transfer_labels` with security warning. All comparison operations unified to reconstruct-compare-reshare.
2. ✅ ~~**Remove private data from BeaverWitness**~~ — **RESOLVED** (2026-02-11): BeaverWitness now only stores `opened_d`, `opened_e`, `party_index`, `verified`.

### High
3. **Fix polynomial ReLU** — Current approximation will break model training. Use minimax polynomial of degree >= 5.
4. **Document aggregation trust model** — aggregation.rs should have a prominent `/// WARNING: Aggregator sees plaintext gradients` comment on every public function.
5. **Store activation commitments, not plaintext** — proved_arithmetic.rs forward/backward recording should hash before storing.
6. **Fix 256-bit → 254-bit** in comparison.rs `fr_to_bits()`.

### Medium
7. Add privacy budget calculation and documentation for reconstruct-reshare pattern
8. Validate dimensions in normalization shared_gamma path
9. Fix aggregation quantization zero-range edge case
10. Add cross-module integration tests (arithmetic → matmul → activation → normalization pipeline)

### Nice-to-have
11. ✅ ~~Runtime dispatch for comparison production/simulation~~ — **RESOLVED**: Unified to single implementation (no cfg split).
12. Implement actual MPC gradient aggregation
13. Priority: tree aggregation for multi-party gradient compression
