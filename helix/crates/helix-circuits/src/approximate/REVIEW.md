# REVIEW: `helix-circuits/src/approximate/` -- Error Bound Algebra for ZK Circuits

**Reviewer date:** 2026-02-10
**Files reviewed:** mod.rs, bounded_add.rs, bounded_mul.rs, bounded_matmul.rs, activation.rs, error_accumulation.rs, quantization.rs
**Lines of code:** ~1,067 (excluding tests), ~1,350 total

---

## 1. Overview

This module implements HELIX's core innovation: proving that neural network computations are correct *within bounded numerical error*, rather than requiring bit-exact correctness. The key insight is that training is inherently noisy (SGD, dropout, quantization), so proving bounded correctness achieves the same security guarantees at dramatically lower proof overhead.

The module provides:
- **Bounded arithmetic gadgets** (add, mul, matmul) that constrain both values and their error terms
- **Activation verification** (ReLU with error propagation)
- **Error accumulation circuit** that proves total error stays within a budget across a sequence of operations
- **Quantization verification** for INT4/INT8 operations with lookup-based range checks

All gadgets follow the same pattern: constrain the value computation via arithmetic gates, constrain the error propagation formula via additional arithmetic gates, then range-check the output error.

---

## 2. Error Algebra

The error propagation rules implemented are standard interval arithmetic:

| Operation | Error Formula | File | Mathematically Correct? |
|-----------|---------------|------|------------------------|
| Add/Sub | `err(a +/- b) = err(a) + err(b)` | bounded_add.rs:3-4, error_accumulation.rs:24-26 | Yes -- triangle inequality |
| Mul | `err(a*b) = \|a\|*err(b) + \|b\|*err(a) + err(a)*err(b)` | bounded_mul.rs:5-6, error_accumulation.rs:28 | Yes -- first-order Taylor with second-order term |
| MatMul | Per-element: sum of multiplication error terms | bounded_matmul.rs:37-38 | Yes -- dot product is iterated multiply-add |
| ReLU | `err(relu(x)) = err(x) if x > 0, else 0` | activation.rs:3-4, error_accumulation.rs:33 | Approximately correct (see weakness W3) |
| Div | "Complex, uses simplified upper bound" | error_accumulation.rs:30-31 | Not actually implemented -- defers to pre-computed witness |
| Quantize | `\|value - quantized * scale\| <= scale / 2` | quantization.rs:310-322 | Yes -- standard rounding bound |

---

## 3. Per-File Analysis

### `mod.rs` (83 lines)

**Purpose:** Module declaration, re-exports, and documentation.

The ASCII architecture diagram and error propagation table in the doc comment are excellent for onboarding. Re-exports are clean and comprehensive.

**Note:** The doc comment lists files that do not exist (error_algebra.rs, bounded_ops.rs, verification.rs, error_budget.rs, error_bound.rs, calibration.rs). The actual files are bounded_add.rs, bounded_mul.rs, bounded_matmul.rs, activation.rs, error_accumulation.rs, quantization.rs.

---

### `bounded_add.rs` (108 lines)

**Purpose:** Verifies `value_c = value_a + value_b` and `error_c = error_a + error_b` with a range check on the output error.

**Key types:**
- `BoundedAddConfig<F, RANGE>` -- holds ArithmeticConfig + RangeConfig
- `BoundedAddChip<F, RANGE>` -- the chip with `assign()` method

**Mathematical correctness:** Correct. Addition error is the sum of input errors (triangle inequality). The range check on `err_c` ensures it stays within `[0, RANGE)`.

**Constraint structure (3 regions):**
1. `s_add` gate: `val_a + val_b = val_c`
2. `s_add` gate: `err_a + err_b = err_c`
3. `s_range` lookup: `err_c in [0, RANGE)`

---

### `bounded_mul.rs` (153 lines)

**Purpose:** Verifies `value_c = value_a * value_b` and the three-term error propagation formula, with range check.

**Key types:**
- `BoundedMulConfig<F, RANGE>`, `BoundedMulChip<F, RANGE>`

**Mathematical correctness:** The error formula `err_c = val_a*err_b + val_b*err_a + err_a*err_b` is the standard product error bound, but only when all values are non-negative. For signed values, the formula should use absolute values (see weakness W1).

**Constraint structure (6 regions):**
1. `s_mul` gate: `val_a * val_b = val_c`
2. `s_mul` gate: `val_a * err_b = term1`
3. `s_mul` gate: `val_b * err_a = term2`
4. `s_mul` gate: `err_a * err_b = term3`
5. `s_add` gate: `term1 + term2 = sum1`
6. `s_add` gate: `sum1 + term3 = err_c`
7. `s_range` lookup: `err_c in [0, RANGE)`

---

### `bounded_matmul.rs` (271 lines)

**Purpose:** Verifies a single dot-product cell of a matrix multiplication, with per-element error propagation.

**Key types:**
- `BoundedMatMulConfig<F, RANGE>`, `BoundedMatMulChip<F, RANGE>`

**Key function:** `assign_dot_product()` -- takes vectors of values and errors for one row of A and one column of B, plus the expected result value and error, and constrains everything.

**Mathematical correctness:** The per-element error is computed as `va*eb + vb*ea + ea*eb` (same as bounded_mul), then accumulated via addition. This is correct for non-negative values but has the same absolute value issue as bounded_mul.

**Constraint count:** Per dot product of length K: `K` mul gates (value), `K-1` add gates (value accumulation), `5K` gates for error terms (3 muls + 2 adds per element), `K-1` add gates for error accumulation, plus 2 add gates for final verification + 1 range check = **~9K + 2** constraints per output element.

**Design concern:** The i=0 special case (lines 105-116, 201-217) skips the accumulation add gate when running sum is zero. The logic is correct (0 + x = x is trivially true), but the asymmetry between i=0 and i>0 is fragile and under-documented.

---

### `activation.rs` (122 lines)

**Purpose:** ReLU activation gadget with error propagation.

**Key types:**
- `ReLUConfig<F, RANGE>`, `ReLUChip<F, RANGE>`

**Key function:** `assign()` -- verifies `y = max(0, x)` and propagates error correctly.

**Constraint approach (5 steps):**
1. Compute `diff = val_y - val_x` via `s_add`: `val_x + diff = val_y`
2. Enforce `val_y * diff = 0` via `s_mul` -- this means either `val_y = 0` or `val_y = val_x`
3. Compute `err_diff = err_y - err_x` via `s_add`
4. Enforce `val_y * err_diff = 0` via `s_mul` -- if `val_y != 0`, then `err_y = err_x`
5. Range check `err_y`

**Mathematical correctness:** The constraint `y * (y - x) = 0` correctly captures ReLU for **non-negative** field elements. However, in a prime field, "negative" values are large positive integers near the modulus, so `y * (y - x) = 0` does NOT actually check `x >= 0`. See weakness W3.

---

### `error_accumulation.rs` (611 lines)

**Purpose:** Proves that error bounds accumulate correctly through a *sequence* of operations, and that the total error stays within a budget.

**Key types:**
- `OpType` enum: Add, Sub, Mul, Div, ReLU, MatMulTerm
- `ErrorAccumulationChip<F, RANGE>` with `configure()`, `assign_add_error_propagation()`, `assign_mul_error_propagation()`, `assign_error_sequence()`
- `ErrorAccumulationCircuit<F, RANGE>` -- a full Circuit impl wrapping the chip
- `OperationWitness<F>`, `OperationData<F>` -- witness types for operations

**Key function:** `assign_error_sequence()` -- iterates through operations, dispatches to add/mul error propagation, accumulates a running error total, then range-checks `max_allowed_error - running_error >= 0`.

**Mathematical correctness:**
- Addition/subtraction error propagation: correct
- Multiplication error propagation: correct (uses 5-gate decomposition)
- ReLU: just trusts `op.output_err` without constraining it (lines 279-283) -- see weakness W4
- Div: same issue, just trusts the witness (lines 284-287) -- see weakness W5
- Final budget check: `remaining_budget = max - running` in range [0, RANGE) -- correct approach

**Tests (4 tests):**
- `test_add_error_accumulation` -- 2 additions, verifies via MockProver
- `test_mul_error_accumulation` -- single multiplication
- `test_error_exceeds_bound` -- budget overflow detection via range check failure
- `test_complex_sequence` -- mixed add/mul sequence

---

### `quantization.rs` (1,067 lines)

**Purpose:** Quantization verification for INT4/INT8 neural network operations. The largest and most feature-complete file.

**Key types:**
- `QuantFormat` enum (UInt4, Int4, UInt8, Int8, SymmetricInt8, AsymmetricInt8)
- `QuantParams` -- scale, zero_point, format, per-channel scales
- `QuantizationChip<F>` with gate definitions and verification methods
- `QuantizedMatMulCircuit<F>` -- full Circuit impl for INT8 matmul verification
- `QuantizedActivationTable<F>` -- lookup tables for ReLU, ReLU6, LeakyReLU, sigmoid
- `QuantErrorTracker` -- non-circuit error tracking for pre-computation
- `estimate_layer_error()` -- analytical error estimation

**Gates defined (5):**
1. `quantization`: `value = quantized * scale + error`
2. `quantized_mul`: `a * b = c`
3. `quantized_add`: `a + b = c`
4. `requantization`: `accum = output * out_scale + error`
5. Range lookups: INT8 (`[0, 256)`), INT4 (`[0, 16)`)

**Mathematical correctness:**
- Quantization gate is correct: `value = quantized * scale + error` with error bounded by range check
- The `QuantizedMatMulCircuit` correctly verifies each `a[i][k] * b[k][j]` term and accumulates
- `QuantizedActivationTable` lookup approach is sound for small bit widths
- `QuantErrorTracker::record_mul` drops the second-order `err_a * err_b` term (line 868-870), which is fine for small errors but could underestimate for large accumulated errors

**Tests (7 tests):** format bounds, quantize/dequantize roundtrip, quantized value construction, ReLU table contents, error tracker, layer error estimation, full matmul circuit via MockProver.

---

## 4. Strengths

**S1. Sound mathematical foundation.** The error propagation rules for addition and multiplication are textbook interval arithmetic. The three-term product error formula (bounded_mul.rs:5-6) correctly includes the second-order `err_a * err_b` term that many implementations drop.

**S2. Clean gadget composition.** All gadgets follow a consistent pattern: ArithmeticConfig + RangeConfig, with separate regions for value constraints, error constraints, and range checks. This makes the code auditable and each constraint independently verifiable. See bounded_add.rs:41-107 for the cleanest example.

**S3. Budget enforcement via subtraction + range check.** The `assign_error_sequence()` approach of computing `remaining_budget = max_allowed - accumulated` and range-checking it (error_accumulation.rs:322-365) is the standard and correct way to prove an inequality in a ZK circuit. This avoids the need for comparison circuits.

**S4. Comprehensive quantization support.** The quantization.rs module covers the full pipeline: INT4/INT8 range checks via lookups, quantize/dequantize verification, quantized arithmetic, requantization, activation tables (4 types), error tracking, and analytical layer error estimation. This is production-grade for quantized inference.

**S5. Good negative test coverage.** The test suite in tests.rs (lines 210-274) explicitly tests that invalid values, invalid errors, and out-of-range errors are all rejected. The error_accumulation test `test_error_exceeds_bound` (error_accumulation.rs:543-566) verifies budget enforcement.

**S6. Freivalds mentioned but not integrated.** The mod.rs doc mentions Freivalds for O(n^2) matmul verification. The `gadgets/freivalds.rs` implements this. While bounded_matmul.rs uses the naive O(K) approach per dot product, the infrastructure for the probabilistic optimization exists.

---

## 5. Weaknesses

### W1. Absolute value not enforced in multiplication error (MEDIUM)

**Location:** bounded_mul.rs:64-68, bounded_matmul.rs:122-127, error_accumulation.rs:174-176

**Problem:** The error formula `err_c = val_a * err_b + val_b * err_a + err_a * err_b` uses raw field elements for `val_a` and `val_b`, not their absolute values. In a prime field, "negative" numbers are represented as large values near the modulus. Multiplying a "negative" value by an error produces a large field element, not the expected small error bound.

For example, if `val_a` represents -3 (i.e., `p - 3` in the field), then `val_a * err_b` where `err_b = 1` gives `p - 3`, not `3`. The range check on err_c would then fail or, worse, the constraint could be satisfied by an incorrect error value that happens to land in range.

**Impact:** Error propagation through multiplication is unsound for signed/negative values. Since neural network weights and activations are frequently negative, this affects correctness of the core error tracking mechanism.

**Suggested fix:** Add absolute value computation gadgets. Decompose each value into sign bit + magnitude: `val = sign * magnitude` where `sign in {0, 1}` (0 = positive, 1 = negative) and `magnitude = val if sign=0 else -val`. Use `magnitude` in the error formula. This requires ~2 additional constraints per value (sign range check + reconstruction).

---

### W2. No copy constraints between regions (HIGH)

**Location:** All files -- bounded_add.rs:52-104, bounded_mul.rs:52-148, bounded_matmul.rs:90-267, activation.rs:87-118, error_accumulation.rs:150-237

**Problem:** Each arithmetic operation is assigned in its own `assign_region` call, but there are NO copy constraints between regions. For example, in bounded_mul.rs:

- Region "bounded mul values" assigns `val_a` at column `a`, row 0
- Region "error term 1" also assigns `val_a` at column `a`, row 0

These are different cells in different regions. The prover could assign different values to "val_a" in each region without violating any constraint. The intended value `val_a` is passed as a `Value<F>` (a witness hint), but there is no `region.constrain_equal()` call linking the cells across regions.

This means a malicious prover could:
1. Assign `val_a = 10` in the value multiplication region (so `val_c = 10 * val_b`)
2. Assign `val_a = 0` in the error term region (so `term1 = 0 * err_b = 0`)
3. Claim zero error while computing a large product

**Impact:** CRITICAL. The error bound constraints are completely bypassable. A prover can claim any error bound for any computation.

**Suggested fix:** Either (a) combine all constraint assignments into a single region so cells are shared, or (b) use `AssignedCell` return values and `region.constrain_equal()` to enforce that the same value is used across regions. The standard halo2 pattern is:

```rust
let val_a_cell = region.assign_advice(|| "val_a", col_a, 0, || val_a)?;
// ... in a later region:
let val_a_copy = region.assign_advice(|| "val_a", col_a, 0, || val_a)?;
layouter.constrain_equal(val_a_cell.cell(), val_a_copy.cell())?;
```

---

### W3. ReLU does not distinguish positive from negative in a prime field (HIGH)

**Location:** activation.rs:61-121

**Problem:** The constraint `val_y * (val_y - val_x) = 0` correctly captures "y is either 0 or x" in any field. However, it does NOT enforce that `y = 0` when `x < 0` and `y = x` when `x >= 0`. In a prime field, there is no notion of "negative" -- all elements are in `[0, p)`. A prover could set `y = x` for any input (claiming the identity function instead of ReLU) and the constraint would be satisfied.

The standard approach for ReLU in ZK circuits is to decompose `x` into positive and negative parts: `x = pos - neg` where `pos, neg >= 0` (range-checked) and `pos * neg = 0` (at most one is nonzero). Then `y = pos`.

**Impact:** HIGH. ReLU is the most common activation function. Without correctly distinguishing sign, the error propagation for ReLU (`err_y = err_x if x > 0, else 0`) is also unconstrained -- a prover can always claim `err_y = err_x` regardless of the sign of x.

**Suggested fix:** Implement the pos/neg decomposition:
```
x + offset = pos + neg_shifted  (where offset shifts to unsigned)
y = pos
range_check(pos, [0, RANGE))
range_check(neg_shifted, [0, RANGE))
pos * neg_shifted_complement = 0  (enforce mutual exclusivity)
```

---

### W4. ReLU and Div error not constrained in accumulation circuit (MEDIUM)

**Location:** error_accumulation.rs:279-287

**Problem:** For `OpType::ReLU` and `OpType::Div`, the accumulation circuit simply uses `op.output_err` as-is, with no constraint proving it follows the correct propagation rule. The witness could contain any value for the output error.

```rust
OpType::ReLU => {
    // ReLU error is same as input error (for positive inputs)
    // For negative inputs, both value and error are 0
    op.output_err  // <-- UNCONSTRAINED
}
OpType::Div => {
    // Division error is complex; assume pre-computed
    op.output_err  // <-- UNCONSTRAINED
}
```

**Impact:** A malicious prover can set ReLU/Div output errors to zero (or any small value), effectively hiding accumulated error. This undermines the entire error budget mechanism for any computation involving activations or divisions.

**Suggested fix:** For ReLU, add the same constraint as activation.rs (or call ReLUChip). For Div, implement the standard quotient error formula: `err(a/b) <= (|a|*err_b + |b|*err_a) / (|b|^2 - err_b^2)`, or at minimum constrain that `err_div >= err_a / |b|` (the dominant term for small errors).

---

### W5. Quantization error range not range-checked (LOW)

**Location:** quantization.rs:312-322

**Problem:** The quantization gate constrains `value = quantized * scale + error`, but the `error` term is not range-checked to be within `[-scale/2, scale/2]`. A prover could set `error` to any value as long as the linear equation holds. The INT8/INT4 range checks only apply to the `quantized` value, not the `error` term.

**Impact:** A prover could assign a large quantization error to one step and compensate with a negative error in another step, potentially masking larger deviations.

**Suggested fix:** Add a range check on `error + scale/2` to ensure it falls in `[0, scale)`. This requires shifting the error to unsigned and applying the existing lookup.

---

### W6. `QuantizedMatMulCircuit` early-exits on mismatch instead of constraining (LOW)

**Location:** quantization.rs:698-701

**Problem:**
```rust
if accum != expected {
    return Err(ErrorFront::Synthesis);
}
```

This check happens during witness generation (synthesize), not via a circuit constraint. The MockProver would catch it because synthesis fails, but a real prover could simply provide matching `accum` and `c[i][j]` values. The constraint on individual multiplications (via `s_quant_mul` gates) should transitively enforce the accumulator correctness, but there is no explicit add-chain constraint for the accumulation itself.

**Impact:** Low -- the individual multiplication gates do constrain each product, so if all products are correct and accumulation is done honestly, the result must match. However, without an accumulation constraint, there is a gap: the circuit verifies each `a*b` but never proves that `c[i][j] = sum(a[i][k]*b[k][j])`.

**Suggested fix:** Add `s_quant_add` constraints for each accumulation step (similar to what bounded_matmul.rs does), and remove the Rust-level assertion.

---

### W7. `QuantErrorTracker` drops second-order term in `record_mul` (LOW)

**Location:** quantization.rs:867-873

**Problem:**
```rust
pub fn record_mul(&mut self, value_bound: f64, other: &Self, other_value_bound: f64) {
    let mul_error = value_bound * other.accumulated_error
        + other_value_bound * self.accumulated_error;
    // Missing: + self.accumulated_error * other.accumulated_error
    self.accumulated_error = mul_error;
}
```

The circuit gadgets (bounded_mul.rs) correctly include the `err_a * err_b` term, but the f64 tracker used for pre-computation and error budget estimation drops it.

**Impact:** Low for small errors (the second-order term is negligible), but could cause tracker underestimates to diverge from circuit constraints for large accumulated errors, leading to witness generation failures.

**Suggested fix:** Add `+ self.accumulated_error * other.accumulated_error` to match the circuit formula.

---

### W8. No test for mixed add/mul/relu sequences (LOW)

**Location:** error_accumulation.rs tests (lines 471-611)

**Problem:** The `test_complex_sequence` test uses only Add and Mul operations. There is no test that exercises ReLU or Div in the accumulation circuit. The activation.rs file has no inline tests at all (all testing is done in the crate-level tests.rs, which tests ReLU in isolation but not in a sequence).

**Suggested fix:** Add a test combining Add + Mul + ReLU in a single `ErrorAccumulationCircuit`, and verify that ReLU error propagation interacts correctly with the running total.

---

## 6. Testing Assessment

| File | Inline Tests | External Tests (tests.rs) | Coverage Assessment |
|------|-------------|--------------------------|-------------------|
| bounded_add.rs | 0 | 3 (valid, invalid value, invalid error, out-of-range) | Good |
| bounded_mul.rs | 0 | 2 (valid, invalid value, invalid error) | Good |
| bounded_matmul.rs | 0 | 1 (valid 1x1 dot product) | Weak -- no multi-element dot product test |
| activation.rs | 0 | 1 (valid positive ReLU) | Weak -- no negative input test, no zero-crossing test |
| error_accumulation.rs | 4 | 0 | Moderate -- tests add, mul, overflow, mixed; no ReLU/Div |
| quantization.rs | 7 | 0 | Good -- format bounds, roundtrip, tables, circuit |

**Total test count:** 18 tests across the module.

**Key gaps:**
- No test for ReLU with negative input (the most important case)
- No multi-element matmul dot product test
- No test for quantization error range bounding
- No adversarial test attempting to exploit the missing copy constraints (W2)
- No test for division error propagation (marked as unimplemented)

---

## 7. Health Score

### Grade: C-

**Rationale:**

The mathematical foundations are correct, the code is well-structured and readable, and the quantization support is impressively thorough. The error propagation formulas for addition and multiplication are textbook-correct.

However, the module has two critical soundness issues that undermine the core value proposition:

1. **Missing copy constraints (W2)** means error bounds are completely bypassable by a malicious prover. This is the single most important issue in the entire module.
2. **Signed value handling (W1, W3)** means error propagation through multiplication and ReLU is incorrect for negative values, which are ubiquitous in neural networks.

These are not theoretical concerns -- they are exploitable in any adversarial setting, which is the entire point of ZK proofs.

The module is suitable as a **prototype/demo** demonstrating the bounded verification concept, but it is NOT suitable for adversarial production use without addressing W1, W2, and W3.

| Aspect | Grade | Notes |
|--------|-------|-------|
| Mathematical correctness | B+ | Formulas correct; signed value handling flawed |
| Circuit soundness | D | Copy constraint gap is critical |
| Code quality | B+ | Clean, consistent, well-documented |
| Test coverage | C | Good positive tests; weak negative/adversarial tests |
| Quantization support | A- | Thorough and well-designed |
| Integration | B | Used by state_transition, gradient, linear_layer circuits |
| Production readiness | D+ | Demo-grade; needs W1-W4 fixes for security |
