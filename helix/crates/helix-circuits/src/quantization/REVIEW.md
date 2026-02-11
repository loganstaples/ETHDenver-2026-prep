# quantization/ -- Code Review

**Module**: `helix-circuits/src/quantization/`
**Reviewed**: 2026-02-11 (updated; original 2026-02-10)
**Files**: mod.rs (92 lines), int8.rs (1,003 lines), int4.rs (799 lines), calibration.rs (785 lines)
**Total lines**: ~2,679

---

## 1. Overview

The `quantization/` module implements ZK circuits for verifying INT8 and INT4 quantized neural network operations. It covers the full quantization pipeline: parameter calibration (determining scale/zero-point), quantization/dequantization gates, quantized arithmetic (matmul, dot product, mixed-precision multiply), and range-check constraints. Each operation is implemented as a halo2 chip with `configure` and `synthesize` methods, and each circuit has MockProver integration tests.

---

## 2. Per-File Analysis

### mod.rs (92 lines)

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

### int8.rs (1,003 lines)

| Type | Purpose |
|------|---------|
| `Int8QuantConfig` | 8 advice columns + lookup table + 6 selectors |
| `Int8SymmetricParams` | Symmetric quantization: `q = round(x / scale)` |
| `Int8AsymmetricParams` | Asymmetric: `q = round(x / scale) + zero_point` |
| `PerChannelInt8Params` | Per-channel scales (defined but not used in circuits) |
| `Int8QuantChip` | Chip with quantize/dequantize/mul/add/requantize gates + INT8 range lookup |
| `Int8MatMulCircuit` | Full `C = A * B` with INT8 |
| `Int8DotProductCircuit` | Dot product of two INT8 vectors |
| `Int8QuantCircuit` | Quantize-then-dequantize roundtrip |

5 tests including `test_int8_matmul_circuit`, `test_int8_dot_product_circuit`.

### int4.rs (799 lines)

| Type | Purpose |
|------|---------|
| `Int4QuantConfig` | 6 advice columns + lookup table + 5 selectors |
| `Int4WeightParams` | Block-wise quantization for LLM weight compression |
| `Int4QuantChip` | Chip with quantize/pack/unpack/mul/mixed_mul gates |
| `Int4PackedCircuit` | Packing two INT4 values into one byte |
| `Int4MixedPrecisionCircuit` | INT4 weights x INT8 activations with INT16 accumulator |
| `Int4QuantCircuit` | Basic quantize/dequantize circuit |

Key functions: `pack_int4_values(a, b) -> u8`, `unpack_int4_values(packed) -> (i8, i8)`.

5 tests including `test_int4_mixed_precision`, `test_int4_packing`.

### calibration.rs (785 lines)

| Type | Purpose |
|------|---------|
| `MinMaxCalibration` | Track min/max of activation distribution |
| `HistogramCalibration` | Percentile-based range selection (99.99th) |
| `EntropyCalibration` | KL-divergence minimization (matches TensorRT approach) |
| `DynamicRangeVerifier` | EMA-based runtime range tracking |
| `CalibrationChip` | Halo2 chip with minmax/range_check/error_bound/scale_valid gates |
| `CalibrationCircuit` | Verifies calibration parameters are valid |

6 tests including `test_calibration_circuit`, `test_entropy_calibration`.

---

## 3. Strengths

1. **Complete gate implementations** (int8.rs:200-350, int4.rs:180-320): All quantization operations have proper halo2 gate definitions. These are real circuit constraints, not stubs.

2. **Range check via lookup argument** (int8.rs:330-345, int4.rs:300-315): More constraint-efficient than bit decomposition (1 lookup vs 8 range gates for INT8).

3. **Mixed-precision support** (int4.rs:450-600): `Int4MixedPrecisionCircuit` handles INT4-weight x INT8-activation, matching GPTQ/AWQ patterns.

4. **Three calibration methods** (calibration.rs:100-400): MinMax, Histogram (percentile), and Entropy (KL-divergence) cover standard approaches used in TensorRT and ONNX Runtime.

5. **Error bound gates** (calibration.rs:550-570): Constrains quantization error within epsilon, tying into HELIX's error budget system.

6. **Block-wise INT4** (int4.rs:60-90): Per-block scales matching GPTQ/AWQ for LLM weight compression.

---

## 4. Weaknesses

### W1: INT32 accumulator overflow not constrained in-circuit (HIGH)
- **Location**: int8.rs:550-600
- **Impact**: Matmul accumulates INT8 x INT8 products into a field element with no INT32 range constraint. Safe for typical ML layers (N < 10000) but not enforced.
- **Fix**: Add decomposition-based range check for INT32 bounds after accumulation.

### W2: Requantize gate assumes zero_point=0 (MEDIUM)
- **Location**: int8.rs:300-310
- **Code**: `s_req * (input * scale_in - output * scale_out) = 0`
- **Impact**: Only correct for symmetric quantization. Asymmetric needs zero-point terms.
- **Fix**: Update to `(input - zp_in) * scale_in - (output - zp_out) * scale_out = 0`.

### W3: PerChannelInt8Params defined but not used in circuits (MEDIUM)
- **Location**: int8.rs:150-170
- **Impact**: Per-channel quantization is important for accuracy but neither Int8MatMulCircuit nor Int8DotProductCircuit use it.
- **Fix**: Add `PerChannelInt8MatMulCircuit` or extend existing circuits.

### W4: EntropyCalibration histogram has fixed 2048 bins (LOW)
- **Location**: calibration.rs:320
- **Fix**: Make bin count configurable.

### W5: No minimum scale threshold (LOW)
- **Location**: calibration.rs:600-620
- **Impact**: `scale = (max_val - min_val) / (2^bits - 1)` can produce denormalized scales.
- **Fix**: Add `scale.max(f64::EPSILON * 100.0)`.

### W6: DynamicRangeVerifier EMA coefficient hardcoded (LOW)
- **Location**: calibration.rs:420
- **Fix**: Make configurable (0.01 for inference, 0.1 for training).

---

## 5. Health Score: B+

**Rationale**: This is one of the strongest modules in helix-circuits. All circuits have real halo2 gate implementations (not stubs), range checks use efficient lookup arguments, and the calibration methods match industry practice. The mixed-precision INT4xINT8 support and block-wise quantization demonstrate awareness of modern techniques. The main gaps are the missing INT32 accumulator constraint (W1) and the asymmetric requantize bug (W2). Overall well-structured and mostly production-ready.
