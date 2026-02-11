# lookup/ -- Code Review

**Module**: `helix-circuits/src/lookup/`
**Reviewed**: 2026-02-11 (updated; original 2026-02-10)
**Files**: mod.rs (83 lines), relu.rs (710 lines), gelu.rs (698 lines), table.rs (992 lines), softmax.rs (893 lines)
**Total lines**: ~3,376

---

## 1. Overview

The `lookup/` module implements Plookup-style lookup tables for ZK-circuit evaluation of ML activation functions. Instead of computing nonlinear functions (ReLU, GELU, Sigmoid, Tanh, Softmax) arithmetically inside the circuit (which would require many constraints), it precomputes function values over a quantized integer range and loads them as lookup tables. The circuit then constrains that each activation output matches a table entry, reducing per-activation cost to a single lookup constraint.

---

## 2. Per-File Analysis

### mod.rs (83 lines)

Module root. Declares submodules, re-exports public types, and defines constants:
- `DEFAULT_LOOKUP_SCALE = 256`
- `INT8_TABLE_SIZE = 256`, `INT4_TABLE_SIZE = 16`, `INT12_TABLE_SIZE = 4096`
- `LookupStats` (hit rate tracking), `LookupError` enum

### relu.rs (710 lines)

Five lookup types + one chip:
- `ReLULookup`, `LeakyReLULookup`, `ReLU6Lookup`, `PReLULookup`, `ReLUGradientLookup`
- `ReLUChip` wrapping `PlookupChip` with selector gates

**BUG** (line 94): `compute_field` checks negativity via `bytes[31] & 0x80 != 0`, which is ALWAYS FALSE for BN254. Affects only native witness computation; the lookup table entries themselves are correct.

6 tests including `test_relu_circuit` (MockProver).

### gelu.rs (698 lines)

Four lookup types + one chip:
- `GELULookup`, `FastGELULookup`, `GELUDerivativeLookup`, `SiLULookup`
- `GELUChip`

Error tracking integrated per table entry. 5 tests including `test_gelu_circuit`.

### table.rs (992 lines)

Core lookup infrastructure:
- `PlookupTable`: `Vec<(Fr, Fr)>` entries + `HashMap<[u8;8], Fr>` cache
- `PlookupChip`: Configure, load, lookup
- `MultiColumnLookup`: N-column lookups (input, output, error)
- `BatchLookupOptimizer`: Deduplicates repeated lookups
- `LookupTableBuilder`: Builder pattern for custom tables
- `PrecomputedTable`: Static pre-built tables

**Note**: HashMap cache key uses only first 8 bytes of 32-byte field representation. Collision-safe for INT8 values but only 64-bit collision resistance for arbitrary elements.

5 tests including `test_lookup_circuit`, `test_batch_optimizer`.

### softmax.rs (893 lines)

Five lookup types + one composite chip:
- `SigmoidLookup`, `TanhLookup`, `SoftmaxExpLookup`, `HardSigmoidLookup`, `HardTanhLookup`
- `SoftmaxChipFull`: Gate configuration for sub/sum/div but **no synthesize implementation**

5 tests including `test_sigmoid_circuit`, `test_softmax_chip_full`.

---

## 3. Strengths

1. **Correct table construction** (relu.rs:22-40, gelu.rs:35-60, softmax.rs:30-55): All generators produce mathematically correct (input, output) pairs with proper quantization.

2. **Full MockProver tests** (relu.rs:550-710, gelu.rs:550-698, table.rs:800-993, softmax.rs:750-893): Every lookup type has a circuit test running through MockProver.

3. **Error bound tracking** (gelu.rs:45-52, softmax.rs:48-55): Table entries record quantization error for downstream error budget accounting.

4. **Batch deduplication** (table.rs:580-650): `BatchLookupOptimizer` deduplicates repeated lookups, reducing constraint count.

5. **Gradient lookups** (relu.rs:180-220, gelu.rs:180-230): Derivative tables enable backward-pass verification in-circuit.

6. **Builder pattern** (table.rs:660-750): `LookupTableBuilder` provides clean API for custom tables.

---

## 4. Weaknesses

### W1: ReLU negative detection is wrong for BN254 (HIGH for witness, no circuit impact)
- **Location**: relu.rs:94
- **Code**: `bytes[31] & 0x80 != 0` -- always false for BN254 Fr
- **Impact**: `compute_field` never detects negative field elements. Witness computation produces wrong values. The lookup table itself correctly maps negatives to zero.
- **Fix**: Compare field element against `(p - 1) / 2`.

### W2: Table cache key truncation to 8 bytes (LOW)
- **Location**: table.rs:226-231
- **Impact**: Only 64-bit collision resistance. Safe for INT8 values, fragile for arbitrary inputs.
- **Fix**: Use full 32-byte field representation as HashMap key.

### W3: SoftmaxChipFull has no synthesize implementation (MEDIUM)
- **Location**: softmax.rs:650-780
- **Impact**: Gates for sub/sum/div are configured but never wired into circuit synthesis. Only native computation works. Individual SigmoidLookup/TanhLookup/SoftmaxExpLookup do work in-circuit.
- **Fix**: Implement synthesize, or remove gate configuration and document lookup-per-element approach.

### W4: Range limited to INT8 (127 entries per side) (MEDIUM)
- **Location**: relu.rs:22, gelu.rs:35, softmax.rs:30
- **Impact**: Values outside [-127, 127] cannot be looked up and cause proof failure. Too narrow for BF16/FP16.
- **Fix**: Support INT12 (4096 entries). The `INT12_TABLE_SIZE` constant exists but is unused.

### W5: No overflow protection in exp table entries (LOW)
- **Location**: softmax.rs:40-45
- **Impact**: `exp(x)` for x > ~709 overflows to `f64::INFINITY`. Not consistently clamped.
- **Fix**: Clamp exp outputs to maximum representable value before Fr conversion.

---

## 5. Health Score: B-

**Rationale**: The lookup infrastructure is well-designed and the core Plookup integration works correctly. Table generators produce mathematically correct entries, MockProver tests validate circuit correctness, error bounds are tracked, and batch deduplication reduces constraint counts. The ReLU `compute_field` bug (W1) is significant for witness generation but does not affect in-circuit soundness. The INT8-only range (W4) limits applicability for higher precision models. The incomplete softmax chip (W3) leaves a gap. Overall, the module is functional for INT8 quantized models and would need W1/W3/W4 fixes for broader production use.
