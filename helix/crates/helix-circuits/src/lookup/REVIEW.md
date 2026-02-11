# lookup/ -- Code Review

## Overview

The `lookup/` module implements Plookup-style lookup tables for ZK-circuit evaluation of ML activation functions. Instead of computing nonlinear functions (ReLU, GELU, Sigmoid, Tanh, Softmax) arithmetically inside the circuit (which would require many constraints), it precomputes function values over a quantized integer range and loads them as lookup tables. The circuit then constrains that each activation output matches a table entry, reducing per-activation cost to a single lookup constraint.

**Files:** 5 (mod.rs, relu.rs, gelu.rs, table.rs, softmax.rs)
**Total lines:** ~3,381

---

## Per-File Analysis

### mod.rs (84 lines)

Module root. Declares submodules, re-exports public types, and defines four constants:

- `DEFAULT_LOOKUP_SCALE = 256` -- default quantization resolution
- `INT8_TABLE_SIZE = 256` -- entries for INT8 range
- `INT4_TABLE_SIZE = 16` -- entries for INT4 range
- `INT12_TABLE_SIZE = 4096` -- entries for INT12 range

Also defines `LookupStats` (hit rate tracking) and `LookupError` enum.

### relu.rs (711 lines)

Implements five lookup types and one chip:

| Type | Purpose |
|------|---------|
| `ReLULookup` | Standard max(0, x) |
| `LeakyReLULookup` | `x if x > 0 else alpha * x` |
| `ReLU6Lookup` | `min(max(0, x), 6)` |
| `PReLULookup` | Per-channel learnable alpha |
| `ReLUGradientLookup` | `1 if x > 0 else 0` for backward pass |
| `ReLUChip` | Halo2 chip wrapping `PlookupChip` with selector gates |

Key functions:
- `relu_table_entries(range)` -- generates `[(input, output)]` pairs for `[-range/2, range/2]`
- `leaky_relu_table_entries(range, alpha)` -- same with leaky variant
- `compute_field(value: Fr) -> Fr` -- native field ReLU for witness computation

**CRITICAL BUG** (line 94): `compute_field` checks negativity via `bytes[31] & 0x80 != 0`. For BN254's scalar field, the modulus p has MSB byte `0x30`, not `0x80`. This check is ALWAYS FALSE for valid field elements, meaning `compute_field` never returns zero for negative inputs. Fortunately, the actual lookup table in `relu_table_entries` (line 22-40) constructs correct `(input, output)` pairs where negative inputs map to zero, so the in-circuit proof is correct. The `compute_field` function is used only in native/witness computation, which means witness values could be wrong and the proof would fail at verification rather than silently producing incorrect results.

Tests: 6 tests including `test_relu_circuit` that runs MockProver.

### gelu.rs (699 lines)

Implements four lookup types and one chip:

| Type | Purpose |
|------|---------|
| `GELULookup` | Standard GELU via tanh approximation |
| `FastGELULookup` | `x * sigmoid(1.702 * x)` (faster, less accurate) |
| `GELUDerivativeLookup` | Analytical GELU derivative for backward pass |
| `SiLULookup` | Swish: `x * sigmoid(x)` |
| `GELUChip` | Halo2 chip wrapping `PlookupChip` |

Key functions:
- `gelu_table_entries(range)` -- generates quantized GELU table
- `gelu_derivative_entries(range)` -- generates derivative table
- `compute_gelu(x: f64) -> f64` -- native computation using `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`

Error tracking is integrated: each table entry stores the max quantization error from rounding.

Tests: 5 tests including `test_gelu_circuit` with MockProver.

### table.rs (993 lines)

Core lookup infrastructure. This is the most important file in the module.

| Type | Purpose |
|------|---------|
| `PlookupTable` | Core table: `Vec<(Fr, Fr)>` entries + `HashMap<[u8;8], Fr>` cache |
| `PlookupChip` | Halo2 chip: configure, load table, perform lookups |
| `MultiColumnLookup` | Extends to N-column lookups (e.g., (input, output, error)) |
| `BatchLookupOptimizer` | Deduplicates repeated lookups in a batch |
| `LookupTableBuilder` | Builder pattern for custom tables |
| `PrecomputedTable` | Static, pre-built table for common functions |

Key methods on `PlookupChip`:
- `configure_simple` / `configure_multi` -- sets up lookup argument columns
- `load_table` / `load_table_padded` -- fills fixed columns during synthesis
- `lookup_single` / `lookup_multi` -- performs constrained lookups

**BUG** (lines 226-231): The `HashMap` cache key uses only the first 8 bytes of the field element's 32-byte representation (`repr[0..8]`). For random field elements this is unlikely to collide, but for small integers (which is the common case for quantized activations), many values share the same upper bytes, so the 8-byte prefix is fine. However, for security-critical scenarios, truncation to 8 bytes provides only 64 bits of collision resistance, which is below the 128-bit standard.

Tests: 5 tests including `test_lookup_circuit`, `test_batch_optimizer`.

### softmax.rs (894 lines)

Implements five lookup types and one composite chip:

| Type | Purpose |
|------|---------|
| `SigmoidLookup` | `1 / (1 + exp(-x))` |
| `TanhLookup` | `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` |
| `SoftmaxExpLookup` | `exp(x)` for softmax numerator |
| `HardSigmoidLookup` | Piecewise linear approximation |
| `HardTanhLookup` | Clipped linear approximation |
| `SoftmaxChipFull` | Composite chip attempting sub, sum, div gates for full softmax |

Key functions:
- `sigmoid_table_entries`, `tanh_table_entries`, `softmax_exp_entries` -- table generators
- `compute_native` / `compute_quantized` on `SoftmaxChipFull`

**WEAKNESS**: `SoftmaxChipFull` defines gate configuration for subtraction, summation, and division (lines 650-780) but has no `synthesize` implementation for the full softmax circuit. Only native computation is available. The individual lookup tables (SigmoidLookup, etc.) work fine as standalone lookups.

Tests: 5 tests including `test_sigmoid_circuit`, `test_softmax_chip_full`.

---

## Strengths

1. **Correct table construction** (relu.rs:22-40, gelu.rs:35-60, softmax.rs:30-55): All table generators produce mathematically correct `(input, output)` pairs. The quantization from `f64` to `Fr` via rounding is sound, and error margins are tracked per-entry.

2. **Full MockProver integration tests** (relu.rs:550-710, gelu.rs:550-698, table.rs:800-993, softmax.rs:750-893): Every lookup type has a circuit test that runs through `MockProver::run` and verifies the proof passes. This catches column misconfiguration and gate errors.

3. **Error bound tracking** (gelu.rs:45-52, softmax.rs:48-55): Table entries record quantization error, enabling downstream error budget accounting. This is a strong design pattern for approximate computing.

4. **Batch deduplication** (table.rs:580-650): `BatchLookupOptimizer` deduplicates repeated lookup inputs, reducing constraint count when the same activation value appears multiple times.

5. **Builder pattern** (table.rs:660-750): `LookupTableBuilder` provides a clean API for constructing custom tables with arbitrary mappings.

6. **Gradient lookups** (relu.rs:180-220, gelu.rs:180-230): Including derivative tables enables backward-pass verification in-circuit, not just forward pass.

---

## Weaknesses

### W1: ReLU negative detection is wrong for BN254
- **Location**: relu.rs:94
- **Code**: `bytes[31] & 0x80 != 0`
- **Impact**: HIGH -- `compute_field` never detects negative field elements. Witness computation produces wrong values. Proofs may fail during MockProver verification for negative inputs, but this silently returns the original value instead of zero.
- **Fix**: Use the standard BN254 negativity check: compare the field element against `(p - 1) / 2`. If `value > (p-1)/2`, treat as negative. Example:
  ```rust
  let half_p = Fr::from_raw([0x9e10460b6c3e7ea4, 0xcbc0b548b438e546, 0xdc2822db40c0ac2e, 0x183227397098d014]);
  if value > half_p { Fr::ZERO } else { value }
  ```

### W2: Table cache key truncation to 8 bytes
- **Location**: table.rs:226-231
- **Code**: `let key: [u8; 8] = repr[0..8].try_into().unwrap()`
- **Impact**: LOW -- For quantized INT8 values (0-255), all 8-byte prefixes are distinct. For larger tables or adversarial inputs, collisions are possible but unlikely in practice (2^64 collision resistance).
- **Fix**: Use the full 32-byte field representation as the HashMap key, or hash it to 16 bytes: `let key = blake2b_simd::Params::new().hash_length(16).hash(&repr).as_bytes().try_into()`.

### W3: SoftmaxChipFull has no synthesize implementation
- **Location**: softmax.rs:650-780
- **Impact**: MEDIUM -- The chip configures gates for sub/sum/div but never wires them into a `Circuit::synthesize` implementation. Only native computation works. The individual SigmoidLookup/TanhLookup/SoftmaxExpLookup chips do work in-circuit.
- **Fix**: Either implement the full `synthesize` method that computes `softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))` using the configured gates, or remove the gate configuration and document that softmax is only available as a lookup-per-element approach.

### W4: Range limited to INT8 (127 entries per side)
- **Location**: relu.rs:22, gelu.rs:35, softmax.rs:30
- **Impact**: MEDIUM -- All table generators use `RANGE/2` as the maximum input magnitude. For INT8 (RANGE=256), this covers [-127, 127]. Values outside this range cannot be looked up and will cause proof failure. For BF16 or FP16 precision levels, this range is far too narrow.
- **Fix**: Add configurable range parameter and support INT12 (4096 entries) or larger tables. The `INT12_TABLE_SIZE` constant is already defined in mod.rs but never used by any table generator.

### W5: No overflow protection in table entry computation
- **Location**: gelu.rs:42, softmax.rs:40-45
- **Impact**: LOW -- `exp(x)` for large positive x overflows to `f64::INFINITY`. The table generators clamp to `f64::MAX` in some places but not consistently. For the SoftmaxExpLookup, entries for `x > ~709` will be infinity.
- **Fix**: Apply log-sum-exp trick at the table level or clamp exp outputs to a maximum representable value before converting to Fr.

---

## Health Score: B-

**Rationale**: The lookup infrastructure is well-designed and the core Plookup integration works correctly. The table generators produce mathematically correct entries, and MockProver tests validate circuit correctness. However, the ReLU `compute_field` bug (W1) is a significant correctness issue for witness generation, the INT8-only range (W4) limits applicability, and the incomplete softmax chip (W3) leaves a gap in the activation function coverage. The module is functional for INT8 quantized models but would need fixes for production use or higher precision.
