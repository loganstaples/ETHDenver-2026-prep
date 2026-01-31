//! Approximate Verification Circuits for Bounded Arithmetic.
//!
//! This module implements HELIX's core innovation: proving that computations
//! are correct within bounded error rather than bit-exact. This is fundamental
//! to achieving practical ZK proof overhead for neural network training.
//!
//! # Key Insight
//!
//! Neural network training is inherently approximate:
//! - Stochastic gradient descent uses noisy, sampled gradients
//! - Quantized training (INT8, INT4) is standard practice
//! - Dropout, batch normalization inject noise
//! - Convergence proofs assume bounded noise tolerance
//!
//! By proving bounded correctness instead of exact correctness, HELIX achieves
//! ~30x overhead vs the 10,000x+ that exact proofs would require.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  Approximate Verification                        │
//! │                                                                  │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                  Error Tracking                          │   │
//! │  │  - Every operation produces value + error bound          │   │
//! │  │  - Bounds propagate through computation graph            │   │
//! │  │  - Final constraint: total_error ≤ max_allowed           │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │                              │                                   │
//! │                              ↓                                   │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                  Bounded Operations                      │   │
//! │  │  - BoundedAdd: err_c = err_a + err_b                    │   │
//! │  │  - BoundedMul: err_c = |a|·err_b + |b|·err_a + ...      │   │
//! │  │  - BoundedMatMul: uses Freivalds for O(n²)              │   │
//! │  │  - Activations: via lookup tables                       │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │                              │                                   │
//! │                              ↓                                   │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                  Quantization                            │   │
//! │  │  - INT4/INT8 range constraints                          │   │
//! │  │  - Quantized arithmetic verification                    │   │
//! │  │  - Requantization (accumulator → output)                │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Error Propagation Rules
//!
//! | Operation | Error Formula |
//! |-----------|---------------|
//! | Add/Sub   | err(a±b) = err(a) + err(b) |
//! | Mul       | err(a·b) ≤ |a|·err(b) + |b|·err(a) + err(a)·err(b) |
//! | MatMul    | Sum over dot product terms |
//! | ReLU      | err(relu(x)) = err(x) if x > 0, else 0 |
//! | Div       | Complex, uses simplified upper bound |

pub mod activation;
pub mod bounded_add;
pub mod bounded_matmul;
pub mod bounded_mul;
pub mod error_accumulation;
pub mod quantization;

// Re-export key types
pub use activation::ReLUChip;
pub use bounded_add::{BoundedAddChip, BoundedAddConfig};
pub use bounded_matmul::{BoundedMatMulChip, BoundedMatMulConfig};
pub use bounded_mul::{BoundedMulChip, BoundedMulConfig};
pub use error_accumulation::{
    ErrorAccumulationChip, ErrorAccumulationCircuit, ErrorAccumulationConfig,
    OpType, OperationData, OperationWitness,
};
pub use quantization::{
    QuantFormat, QuantParams, QuantizedValue, QuantErrorTracker,
    QuantizationChip, QuantizationConfig, QuantizedMatMulCircuit,
    QuantizedActivationTable,
    INT4_RANGE, INT8_RANGE, DEFAULT_SCALE,
    estimate_layer_error,
};
