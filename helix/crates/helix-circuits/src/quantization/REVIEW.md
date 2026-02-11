# quantization/ -- Code Review

## Overview

The `quantization/` module implements ZK circuits for verifying INT8 and INT4 quantized neural network operations. It covers the full quantization pipeline: parameter calibration (determining scale/zero-point), quantization/dequantization gates, quantized arithmetic (matmul, dot product, mixed-precision multiply), and range-check constraints. Each operation is implemented as a halo2 chip with `configure` and `synthesize` methods, and each circuit has MockProver integration tests.

**Files:** 4 (mod.rs, int8.rs, int4.rs, calibration.rs)
**Total lines:** ~2,683

---

## Per-File Analysis

### mod.rs (93 lines)

Module root defining constants and re-exports:

| Constant | Value | Purpose |
|----------|-------|---------|
| `INT8_MAX` | 127 | Signed 8-bit maximum |
| `INT8_MIN` | -128 | Signed 8-bit minimum |
| `INT4_MAX` | 7 | Signed 4-bit maximum |
| `INT4_MIN` | -8 | Signed 4-bit minimum |
| `INT32_ACCUM_MAX` | 2^31-1 | Accumulator max for matmul |
| `INT32_ACCUM_MIN` | -2^31 | Accumulator min for matmul |
| `DEFAULT_INT8_SCALE` | 0.00784314 | ~1/127, default symmetric scale |
| `DEFAULT_INT4_SCALE` | 0.142857 | ~1/7, default symmetric scale |

### int8.rs (1,004 lines)

Implements INT8 quantization circuits:

| Type | Purpose |
|------|---------|
| `Int8QuantConfig` | 8 advice columns + lookup table + 6 selectors |
| `Int8SymmetricParams` | Symmetric quantization: `q = round(x / scale)` |
| `Int8AsymmetricParams` | Asymmetric: `q = round(x / scale) + zero_point` |
| `PerChannelInt8Params` | Per-channel scales for conv/linear layers |
| `Int8QuantWitness` | Input values + scale + zero_point for circuit |
| `Int8QuantChip` | Chip with quantize/dequantize/mul/add/requantize gates + INT8 range lookup |
| `Int8MatMulCircuit` | Full circuit for `C = A * B` with INT8 values |
| `Int8DotProductCircuit` | Circuit for dot product of two INT8 vectors |
| `Int8QuantCircuit` | Circuit for quantize-then-dequantize roundtrip |

Key gates in `Int8QuantChip::configure` (lines 200-350):
- **quantize gate**: `s_q * (input - scale * output - zero_point) = 0`
- **dequantize gate**: `s_dq * (output - scale * input + scale * zero_point) = 0`
- **mul gate**: `s_mul * (a * b - output) = 0`
- **add gate**: `s_add * (a + b - output) = 0`
- **requantize gate**: `s_req * (input * scale_in - output * scale_out) = 0`
- **range check**: Lookup argument constraining all quantized values to [0, 255]

The `Int8MatMulCircuit` (lines 500-700) implements a full matrix multiply: for each output element `C[i][j]`, it computes the dot product `sum(A[i][k] * B[k][j])` with accumulation in INT32, then requantizes to INT8.

Tests: 5 tests including `test_int8_matmul_circuit`, `test_int8_dot_product_circuit`.

### int4.rs (800 lines)

Implements INT4 quantization circuits with packing:

| Type | Purpose |
|------|---------|
| `Int4QuantConfig` | 6 advice columns + lookup table + 5 selectors |
| `Int4WeightParams` | Includes block-wise quantization for LLM weight compression |
| `Int4QuantChip` | Chip with quantize/pack/unpack/mul/mixed_mul gates |
| `Int4PackedCircuit` | Circuit for packing two INT4 values into one byte |
| `Int4MixedPrecisionCircuit` | INT4 weights x INT8 activations with INT16 accumulator |
| `Int4QuantCircuit` | Basic quantize/dequantize circuit |

Key functions:
- `pack_int4_values(a, b) -> u8` (line 100): Packs two 4-bit values into one byte (`(a << 4) | (b & 0x0F)`)
- `unpack_int4_values(packed) -> (i8, i8)` (line 110): Reverse operation with sign extension

The `Int4MixedPrecisionCircuit` (lines 450-600) is particularly relevant for modern LLM inference, where weights are INT4 but activations remain INT8. The circuit constrains: `output = sum(w4_i * act8_i) * scale_w * scale_a`.

Block-wise quantization in `Int4WeightParams` (lines 60-90) supports the GPTQ/AWQ pattern where each block of weights shares a scale factor, reducing quantization error.

Tests: 5 tests including `test_int4_mixed_precision`, `test_int4_packing`.

### calibration.rs (786 lines)

Implements calibration methods for determining quantization parameters:

| Type | Purpose |
|------|---------|
| `MinMaxCalibration` | Track min/max of activation distribution |
| `HistogramCalibration` | Percentile-based range selection (default 99.99th) |
| `EntropyCalibration` | KL-divergence minimization between original and quantized distributions |
| `DynamicRangeVerifier` | EMA-based runtime range tracking |
| `CalibrationChip` | Halo2 chip with minmax/range_check/error_bound/scale_valid gates |
| `CalibrationCircuit` | Verifies calibration parameters are valid |

Key algorithm in `EntropyCalibration` (lines 300-400): iterates over candidate quantization ranges, computes the KL divergence between the original histogram and the quantized-then-dequantized histogram, and selects the range minimizing divergence. This matches TensorRT's calibration approach.

Standalone functions:
- `verify_calibration_bounds(scale, zero_point, min_val, max_val)` -- checks parameter consistency
- `compute_optimal_scale(values, bit_width)` -- min-max scale calculation

The `CalibrationChip` gates (lines 500-600) constrain:
- **minmax gate**: `observed_min <= value <= observed_max`
- **range_check gate**: `0 <= quantized_value <= 2^bits - 1`
- **error_bound gate**: `|dequantized - original| <= epsilon`
- **scale_valid gate**: `scale > 0 AND scale <= max_scale`

Tests: 6 tests including `test_calibration_circuit`, `test_entropy_calibration`.

---

## Strengths

1. **Complete gate implementations** (int8.rs:200-350, int4.rs:180-320): All quantization operations (quantize, dequantize, multiply, add, requantize) have proper halo2 gate definitions with selector columns. These are real circuit constraints, not stubs.

2. **Range check via lookup argument** (int8.rs:330-345, int4.rs:300-315): Quantized values are constrained to valid ranges using lookup tables rather than decomposition into bits, which is more constraint-efficient (1 lookup vs 8 range gates for INT8).

3. **Mixed-precision support** (int4.rs:450-600): `Int4MixedPrecisionCircuit` correctly handles the common INT4-weight x INT8-activation pattern with proper scale factor combination, matching modern LLM quantization practice (GPTQ, AWQ).

4. **Three calibration methods** (calibration.rs:100-400): MinMax, Histogram (percentile), and Entropy (KL-divergence) cover the standard calibration approaches used in production quantization tools (TensorRT, ONNX Runtime).

5. **Error bound gates** (calibration.rs:550-570): The calibration circuit constrains that quantization error stays within a specified epsilon, tying into HELIX's error budget system.

6. **Block-wise INT4 quantization** (int4.rs:60-90): `Int4WeightParams` supports per-block scales, matching the GPTQ/AWQ quantization scheme used in practice for LLM weight compression.

---

## Weaknesses

### W1: INT32 accumulator overflow is not constrained in-circuit
- **Location**: int8.rs:550-600 (Int8MatMulCircuit synthesize)
- **Impact**: HIGH -- The matmul accumulates INT8 x INT8 products into a field element, but there is no constraint proving the accumulator stays within INT32 range. For a dot product of length N with max values 127*127=16129, overflow occurs at N > 133,000. For typical ML layers (N < 10000) this is safe, but the circuit does not enforce it.
- **Fix**: Add a range check after accumulation: decompose the accumulator into bits and verify it fits in 32 bits, or add a lookup table for the INT32 range.

### W2: Requantize gate assumes linear scale relationship
- **Location**: int8.rs:300-310
- **Code**: `s_req * (input * scale_in - output * scale_out) = 0`
- **Impact**: MEDIUM -- This constrains `output = input * scale_in / scale_out`, which is only correct for symmetric quantization with zero_point=0. For asymmetric quantization, the correct formula is `output = (input - zp_in) * scale_in / scale_out + zp_out`.
- **Fix**: Update the requantize gate to include zero-point terms:
  ```
  s_req * ((input - zp_in) * scale_in - (output - zp_out) * scale_out) = 0
  ```

### W3: PerChannelInt8Params is defined but not used in circuits
- **Location**: int8.rs:150-170
- **Impact**: MEDIUM -- Per-channel quantization is defined as a struct with a `Vec<f64>` of per-channel scales, but neither `Int8MatMulCircuit` nor `Int8DotProductCircuit` use it. They only accept single-scale `Int8SymmetricParams`.
- **Fix**: Add a `PerChannelInt8MatMulCircuit` that applies different scales per output channel, or extend `Int8MatMulCircuit` to accept per-channel params.

### W4: EntropyCalibration histogram has fixed 2048 bins
- **Location**: calibration.rs:320
- **Impact**: LOW -- The histogram bin count is hardcoded to 2048. For distributions with very different shapes (e.g., bimodal activations in attention layers), this may be too coarse or too fine.
- **Fix**: Make bin count configurable via a parameter, defaulting to 2048.

### W5: No overflow protection in scale computation
- **Location**: calibration.rs:600-620 (compute_optimal_scale)
- **Impact**: LOW -- `scale = (max_val - min_val) / (2^bits - 1)` can produce very small scales for narrow distributions, leading to precision loss in field arithmetic. No minimum scale threshold is enforced.
- **Fix**: Add `scale = scale.max(f64::EPSILON * 100.0)` to prevent denormalized scales.

### W6: DynamicRangeVerifier EMA coefficient is hardcoded
- **Location**: calibration.rs:420
- **Impact**: LOW -- The exponential moving average coefficient for dynamic range tracking is fixed at 0.01. This means the range adapts slowly, which is appropriate for inference but not for training where distributions shift rapidly.
- **Fix**: Make the EMA coefficient configurable, with a default of 0.01 for inference and 0.1 for training.

---

## Health Score: B+

**Rationale**: This is one of the strongest modules in helix-circuits. All circuits have real halo2 gate implementations (not stubs), range checks use efficient lookup arguments, and the calibration methods match industry practice. The mixed-precision INT4xINT8 support and block-wise quantization demonstrate awareness of modern quantization techniques. The main gap is the missing INT32 accumulator range constraint (W1), which is a real soundness issue for large matrix dimensions, and the asymmetric requantize gate bug (W2). Overall, the module is well-structured and mostly production-ready.
