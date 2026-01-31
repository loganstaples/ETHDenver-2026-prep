//! Quantization Circuit Module for HELIX.
//!
//! This module provides production-grade quantization circuits for efficient
//! neural network verification with reduced precision arithmetic.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                     Quantization Circuit Infrastructure                  │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                    INT8 Quantization                             │   │
//! │  │  - Symmetric quantization (zero_point = 0)                       │   │
//! │  │  - Asymmetric quantization (arbitrary zero_point)                │   │
//! │  │  - Per-tensor and per-channel scales                             │   │
//! │  │  - Fused quantize-dequantize verification                        │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                    INT4 Quantization                             │   │
//! │  │  - 4-bit precision for weights                                   │   │
//! │  │  - Packing (2 INT4 values per byte)                              │   │
//! │  │  - Mixed-precision support (INT4 weights, INT8 activations)      │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                    Calibration Verification                      │   │
//! │  │  - Min-max calibration bounds                                    │   │
//! │  │  - Histogram-based calibration                                   │   │
//! │  │  - Entropy (KL divergence) calibration                           │   │
//! │  │  - Dynamic quantization range verification                       │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Error Bounds
//!
//! Quantization introduces bounded error:
//! - Per-value quantization error: ±0.5 * scale
//! - Accumulated error through operations follows algebraic rules
//! - Calibration verification ensures scales are within valid bounds

pub mod int8;
pub mod int4;
pub mod calibration;

pub use int8::{
    Int8QuantCircuit, Int8QuantConfig, Int8QuantChip,
    Int8SymmetricParams, Int8AsymmetricParams,
    Int8MatMulCircuit, Int8DotProductCircuit,
    PerChannelInt8Params, Int8QuantWitness,
};

pub use int4::{
    Int4QuantCircuit, Int4QuantConfig, Int4QuantChip,
    Int4PackedCircuit, Int4MixedPrecisionCircuit,
    Int4WeightParams, pack_int4_values, unpack_int4_values,
};

pub use calibration::{
    CalibrationCircuit, CalibrationConfig, CalibrationChip,
    MinMaxCalibration, HistogramCalibration, EntropyCalibration,
    DynamicRangeVerifier, CalibrationWitness,
    verify_calibration_bounds, compute_optimal_scale,
};

/// Maximum INT8 value.
pub const INT8_MAX: i64 = 127;
/// Minimum INT8 value.
pub const INT8_MIN: i64 = -128;
/// INT8 range size.
pub const INT8_RANGE: usize = 256;

/// Maximum INT4 value (signed).
pub const INT4_MAX: i64 = 7;
/// Minimum INT4 value (signed).
pub const INT4_MIN: i64 = -8;
/// INT4 range size.
pub const INT4_RANGE: usize = 16;

/// Maximum INT32 accumulator value (practical subset).
pub const INT32_ACCUM_MAX: i64 = 2_147_483_647;
/// Minimum INT32 accumulator value.
pub const INT32_ACCUM_MIN: i64 = -2_147_483_648;

/// Default quantization scale for INT8.
pub const DEFAULT_INT8_SCALE: f64 = 1.0 / 127.0;
/// Default quantization scale for INT4.
pub const DEFAULT_INT4_SCALE: f64 = 1.0 / 7.0;
