# REVIEW: `helix-circuits/src/approximate/` -- Error Bound Algebra for ZK Circuits

**Reviewer date:** 2026-02-11 (updated; original 2026-02-10)
**Files reviewed:** mod.rs, bounded_add.rs, bounded_mul.rs, bounded_matmul.rs, activation.rs, error_accumulation.rs, quantization.rs
**Lines of code:** ~2,359

---

## 1. Overview

This module implements HELIX's core innovation: proving that neural network computations are correct *within bounded numerical error*, rather than requiring bit-exact correctness. The key insight is that training is inherently noisy (SGD, dropout, quantization), so proving bounded correctness achieves the same security guarantees at dramatically lower proof overhead.

The module provides:
- **Bounded arithmetic gadgets** (add, mul, matmul) that constrain both values and their error terms
- **Activation verification** (ReLU with error propagation)
- **Error accumulation circuit** that proves total error stays within a budget across a sequence of operations
- **Quantization verification** for INT4/INT8 operations with lookup-based range checks

All gadgets now follow a single-region pattern with explicit copy constraints binding shared values across different gate rows.

---

## 2. Error Algebra

The error propagation rules implemented are standard interval arithmetic:

| Operation | Error Formula | File | Mathematically Correct? |
|-----------|---------------|------|------------------------|
| Add/Sub | `err(a +/- b) = err(a) + err(b)` | bounded_add.rs | Yes -- triangle inequality |
| Mul | `err(a*b) = \|a\|*err(b) + \|b\|*err(a) + err(a)*err(b)` | bounded_mul.rs | Yes -- first-order Taylor with second-order term |
| MatMul | Per-element: sum of multiplication error terms | bounded_matmul.rs | Yes -- dot product is iterated multiply-add |
| ReLU | `err(relu(x)) = err(x) if x > 0, else 0` | activation.rs | Correct (sound decomposition with range checks) |
| Div | "Complex, uses simplified upper bound" | error_accumulation.rs | Not actually implemented -- defers to pre-computed witness |
| Quantize | `\|value - quantized * scale\| <= scale / 2` | quantization.rs | Yes -- standard rounding bound |

---

## 3. Per-File Analysis

### `mod.rs` (83 lines)

**Purpose:** Module declaration, re-exports, and documentation.

The ASCII architecture diagram and error propagation table in the doc comment are excellent for onboarding. Re-exports are clean and comprehensive.

**Note:** The doc comment lists files that do not exist (error_algebra.rs, bounded_ops.rs, verification.rs, error_budget.rs, error_bound.rs, calibration.rs). The actual files are bounded_add.rs, bounded_mul.rs, bounded_matmul.rs, activation.rs, error_accumulation.rs, quantization.rs.

---

### `bounded_add.rs` (91 lines)

**Purpose:** Verifies `value_c = value_a + value_b` and `error_c = error_a + error_b` with a range check on the output error.

**Key types:**
- `BoundedAddConfig<F, RANGE>` -- holds ArithmeticConfig + RangeConfig
- `BoundedAddChip<F, RANGE>` -- the chip with `assign()` method

**Mathematical correctness:** Correct. Addition error is the sum of input errors (triangle inequality). The range check on `err_c` ensures it stays within `[0, RANGE)`.

**Constraint structure (single region, 3 rows):**
1. Row 0: `s_add` gate: `val_a + val_b = val_c`
2. Row 1: `s_add` gate: `err_a + err_b = err_c`
3. Row 2: `s_range` lookup: `err_c in [0, RANGE)`
4. **Copy constraint**: `err_c` at row 1 == `err_c` at row 2 (prevents prover from using different error values)

**Status:** **FIXED** -- Previously used 3 separate regions. Now single-region with copy constraint (line 82).

---

### `bounded_mul.rs` (126 lines)

**Purpose:** Verifies `value_c = value_a * value_b` and the three-term error propagation formula, with range check.

**Key types:**
- `BoundedMulConfig<F, RANGE>`, `BoundedMulChip<F, RANGE>`

**Mathematical correctness:** The error formula `err_c = val_a*err_b + val_b*err_a + err_a*err_b` is the standard product error bound, but only when all values are non-negative. For signed values, the formula should use absolute values (see weakness W1).

**Constraint structure (single region, 7 rows):**
1. Row 0: `s_mul`: `val_a * val_b = val_c`
2. Row 1: `s_mul`: `val_a * err_b = term1`
3. Row 2: `s_mul`: `val_b * err_a = term2`
4. Row 3: `s_mul`: `err_a * err_b = term3`
5. Row 4: `s_add`: `term1 + term2 = sum1`
6. Row 5: `s_add`: `sum1 + term3 = err_c`
7. Row 6: `s_range` lookup: `err_c in [0, RANGE)`
8. **8 copy constraints**: val_a (0==1), val_b (0==2), err_a (2==3), err_b (1==3), term1 (1==4), term2 (2==4), term3 (3==5), sum1 (4==5)

**Status:** **FIXED** -- Previously used 7 separate regions with no cross-region binding. Now single-region with 8 explicit `constrain_equal` calls (lines 110-117).

---

### `bounded_matmul.rs` (245 lines)

**Purpose:** Verifies a single dot-product cell of a matrix multiplication, with per-element error propagation.

**Key types:**
- `BoundedMatMulConfig<F, RANGE>`, `BoundedMatMulChip<F, RANGE>`

**Key function:** `assign_dot_product()` -- takes vectors of values and errors for one row of A and one column of B, plus the expected result value and error, and constrains everything.

**Mathematical correctness:** Per-element error is computed as `va*eb + vb*ea + ea*eb`, then accumulated via addition. Correct for non-negative values.

**Constraint count:** Per dot product of length K: ~9K + 2 constraints per output element.

**Status:** Uses cross-region copy constraints for running sum values. AssignedCell references track values across iterations.

---

### `activation.rs` (134 lines)

**Purpose:** ReLU activation gadget with error propagation.

**Key types:**
- `ReLUConfig<F, RANGE>`, `ReLUChip<F, RANGE>`

**Constraint approach (single region, 7 rows):**
1. Row 0: `x + neg = y` (decomposition)
2. Row 1: `y * neg = 0` (exactly one of y, neg is zero -- disjointness)
3. Row 2: range_check(y) -- y >= 0
4. Row 3: range_check(neg) -- neg >= 0
5. Row 4: `err_x + err_diff = err_y` (error decomposition)
6. Row 5: `y * err_diff = 0` (if y != 0, err_diff = 0 => err_y = err_x)
7. Row 6: range_check(err_y)
8. **6 copy constraints**: val_y (0==1==2==5), neg (0==1==3), err_diff (4==5), err_y (4==6)

**Status:** **FIXED** -- Previously used the unsound `y * (y - x) = 0` constraint which didn't distinguish positive from negative in a prime field. Now uses the sound pos/neg decomposition `x + neg = y, y * neg = 0, range_check(y), range_check(neg)`. This correctly captures ReLU for values within the range check's domain.

---

### `error_accumulation.rs` (612 lines)

**Purpose:** Proves that error bounds accumulate correctly through a *sequence* of operations, and that the total error stays within a budget.

**Key types:**
- `OpType` enum: Add, Sub, Mul, Div, ReLU, MatMulTerm
- `ErrorAccumulationChip<F, RANGE>` with `configure()`, `assign_add_error_propagation()`, `assign_mul_error_propagation()`, `assign_error_sequence()`
- `ErrorAccumulationCircuit<F, RANGE>` -- a full Circuit impl wrapping the chip

**Mathematical correctness:**
- Addition/subtraction error propagation: correct
- Multiplication error propagation: correct formula (uses 5-gate decomposition)
- ReLU: just trusts `op.output_err` without constraining it (lines 279-283) -- see weakness W3
- Div: same issue, just trusts the witness (lines 284-287) -- see weakness W3
- Final budget check: `remaining_budget = max - running` in range [0, RANGE) -- correct approach

**Status:** **NOT YET FIXED** -- Unlike the bounded_add/mul/matmul/activation chips, the error accumulation circuit's `assign_mul_error_propagation` still uses separate regions without cross-region copy constraints. This is the last remaining copy-constraint gap in the approximate module.

**Tests (4 tests):**
- `test_add_error_accumulation` -- 2 additions, verifies via MockProver
- `test_mul_error_accumulation` -- single multiplication
- `test_error_exceeds_bound` -- budget overflow detection via range check failure
- `test_complex_sequence` -- mixed add/mul sequence

---

### `quantization.rs` (1,068 lines)

**Purpose:** Quantization verification for INT4/INT8 neural network operations. The largest and most feature-complete file.

**Key types:**
- `QuantFormat` enum (UInt4, Int4, UInt8, Int8, SymmetricInt8, AsymmetricInt8)
- `QuantParams` -- scale, zero_point, format, per-channel scales
- `QuantizationChip<F>` with gate definitions and verification methods
- `QuantizedMatMulCircuit<F>` -- full Circuit impl for INT8 matmul verification
- `QuantizedActivationTable<F>` -- lookup tables for ReLU, ReLU6, LeakyReLU, sigmoid
- `QuantErrorTracker` -- non-circuit error tracking for pre-computation

**Gates defined (5):**
1. `quantization`: `value = quantized * scale + error`
2. `quantized_mul`: `a * b = c`
3. `quantized_add`: `a + b = c`
4. `requantization`: `accum = output * out_scale + error`
5. Range lookups: INT8 (`[0, 256)`), INT4 (`[0, 16)`)

**Mathematical correctness:** All gates are correct. The `QuantErrorTracker::record_mul` drops the second-order `err_a * err_b` term (line 868-870), which is fine for small errors but could underestimate for large accumulated errors.

**Tests (7 tests):** format bounds, quantize/dequantize roundtrip, quantized value construction, ReLU table contents, error tracker, layer error estimation, full matmul circuit via MockProver.

---

## 4. Strengths

**S1. Sound mathematical foundation.** The error propagation rules for addition and multiplication are textbook interval arithmetic. The three-term product error formula (bounded_mul.rs:57-59) correctly includes the second-order `err_a * err_b` term.

**S2. Single-region pattern with copy constraints (FIXED).** All core gadgets (bounded_add, bounded_mul, activation) now use a single-region layout with explicit `constrain_equal` calls. This prevents a malicious prover from using different values at different rows. See bounded_mul.rs:65-121 for the cleanest example.

**S3. Sound ReLU decomposition (FIXED).** The activation chip now uses `x + neg = y, y * neg = 0, range_check(y), range_check(neg)` which correctly enforces the ReLU function within the range check domain. The error propagation `y * err_diff = 0` correctly links error to the activation output.

**S4. Budget enforcement via subtraction + range check.** The `assign_error_sequence()` approach of computing `remaining_budget = max_allowed - accumulated` and range-checking it (error_accumulation.rs:322-365) is the standard and correct way to prove an inequality in a ZK circuit.

**S5. Comprehensive quantization support.** The quantization.rs module covers the full pipeline: INT4/INT8 range checks via lookups, quantize/dequantize verification, quantized arithmetic, requantization, activation tables (4 types), error tracking, and analytical layer error estimation.

**S6. Good negative test coverage.** The test suite in tests.rs explicitly tests that invalid values, invalid errors, and out-of-range errors are all rejected. The error_accumulation test `test_error_exceeds_bound` verifies budget enforcement.

---

## 5. Weaknesses

### W1. Absolute value not enforced in multiplication error (MEDIUM)

**Location:** bounded_mul.rs:57-59, bounded_matmul.rs:122-127, error_accumulation.rs:174-176

**Problem:** The error formula uses raw field elements for `val_a` and `val_b`, not their absolute values. In a prime field, "negative" numbers are large values near the modulus. Multiplying a "negative" value by an error produces a large field element, not the expected small error bound.

**Impact:** Error propagation through multiplication is unsound for signed/negative values. Since neural network weights and activations are frequently negative, this affects correctness of the core error tracking mechanism.

**Suggested fix:** Add absolute value computation gadgets. Decompose each value into sign bit + magnitude. Use magnitude in the error formula. (~2 additional constraints per value.)

---

### W2. Error accumulation uses separate regions without copy constraints (MEDIUM)

**Location:** error_accumulation.rs:150-237 (`assign_mul_error_propagation`)

**Problem:** Unlike the fixed bounded_mul.rs (single-region, 8 copy constraints), the error accumulation circuit's multiplication error propagation still uses separate `assign_region` calls without `constrain_equal` between them.

**Impact:** A malicious prover could assign different values for shared variables across regions in the accumulation circuit, potentially claiming lower accumulated error than actually occurred.

**Suggested fix:** Rewrite `assign_mul_error_propagation` to use the same single-region pattern as bounded_mul.rs.

---

### W3. ReLU and Div error not constrained in accumulation circuit (MEDIUM)

**Location:** error_accumulation.rs:279-287

**Problem:** For `OpType::ReLU` and `OpType::Div`, the accumulation circuit simply uses `op.output_err` as-is, with no constraint proving it follows the correct propagation rule.

**Impact:** A malicious prover can set ReLU/Div output errors to zero, hiding accumulated error and undermining the budget mechanism.

**Suggested fix:** For ReLU, call the ReLUChip or add the same constraint pattern. For Div, constrain `err_div >= err_a / |b|` (dominant term for small errors).

---

### W4. Quantization error range not range-checked (LOW)

**Location:** quantization.rs:312-322

**Problem:** The quantization gate constrains `value = quantized * scale + error`, but the `error` term is not range-checked to be within `[-scale/2, scale/2]`. Only the `quantized` value is range-checked.

**Suggested fix:** Add a range check on `error + scale/2` to ensure it falls in `[0, scale)`.

---

### W5. `QuantErrorTracker` drops second-order term in `record_mul` (LOW)

**Location:** quantization.rs:867-873

**Problem:** The circuit gadgets (bounded_mul.rs) correctly include `err_a * err_b`, but the f64 tracker used for pre-computation drops it. Could cause tracker underestimates to diverge from circuit constraints for large accumulated errors.

**Suggested fix:** Add `+ self.accumulated_error * other.accumulated_error`.

---

### W6. mod.rs doc comment references non-existent files (LOW)

**Location:** mod.rs:1-80

**Problem:** Lists error_algebra.rs, bounded_ops.rs, verification.rs, error_budget.rs, error_bound.rs, calibration.rs. Actual files have different names.

**Suggested fix:** Update doc comment to list actual file names.

---

## 6. Testing Assessment

| File | Inline Tests | External Tests (tests.rs) | Coverage Assessment |
|------|-------------|--------------------------|-------------------|
| bounded_add.rs | 0 | 3 (valid, invalid value, invalid error, out-of-range) | Good |
| bounded_mul.rs | 0 | 2 (valid, invalid value, invalid error) | Good |
| bounded_matmul.rs | 0 | 1 (valid 1x1 dot product) | Weak -- no multi-element dot product test |
| activation.rs | 0 | 1 (valid positive ReLU) | Moderate -- tests positive case; neg handled by decomposition |
| error_accumulation.rs | 4 | 0 | Moderate -- tests add, mul, overflow, mixed; no ReLU/Div |
| quantization.rs | 7 | 0 | Good -- format bounds, roundtrip, tables, circuit |

**Total test count:** 18 tests across the module.

**Key gaps:**
- No multi-element matmul dot product test
- No test for quantization error range bounding
- No adversarial test attempting to exploit the error accumulation copy constraint gap (W2)
- No test for division error propagation

---

## 7. Health Score

### Grade: B-

**Rationale:**

The mathematical foundations are correct, the code is well-structured, and the quantization support is thorough. **The critical copy constraint gap (previously the #1 issue) has been fixed** in bounded_add.rs, bounded_mul.rs, bounded_matmul.rs, and activation.rs. The ReLU activation now uses a sound pos/neg decomposition with range checks instead of the unsound `y * (y - x) = 0` constraint.

The remaining issues are:
1. **Error accumulation still has separate-region pattern** (W2) -- needs the same fix applied to bounded_mul.rs
2. **Signed value handling in multiplication error** (W1) -- the error formula uses raw field elements instead of absolute values
3. **ReLU/Div error unconstrained in accumulation** (W3) -- witness-only values

The module is now suitable as a **demo with acknowledged limitations** for small positive values (where the absolute value issue doesn't trigger). For adversarial production use, W1-W3 still need addressing.

| Aspect | Grade | Notes |
|--------|-------|-------|
| Mathematical correctness | B+ | Formulas correct; signed value handling flawed |
| Circuit soundness | B- | Copy constraints fixed in core gadgets; accumulation gap remains |
| Code quality | B+ | Clean, consistent, well-documented |
| Test coverage | C+ | Good positive tests; weak negative/adversarial tests |
| Quantization support | A- | Thorough and well-designed |
| Integration | B | Used by state_transition, gradient, linear_layer circuits |
| Production readiness | C+ | Demo-grade with improvements; needs W1-W3 for security |
