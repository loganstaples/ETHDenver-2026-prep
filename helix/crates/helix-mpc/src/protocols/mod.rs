//! MPC protocols for secure computation on secret-shared values.
//!
//! These protocols enable parties to compute functions on their shares
//! without revealing the underlying secrets.
//!
//! Linear operations (add, subtract, scale by public constant) can be done
//! locally by each party on their shares. Non-linear operations (multiply,
//! activations) require inter-party communication.
//!
//! # Available Protocols
//!
//! - `arithmetic`: Basic operations (add, multiply via Beaver triples)
//! - `proved_arithmetic`: Arithmetic operations with ZK witness capture
//! - `matmul`: Secure matrix multiplication
//! - `activation`: Activation functions (ReLU, GELU, etc.)
//! - `comparison`: Secure comparison and sign computation
//! - `normalization`: Layer normalization and batch normalization
//! - `reshare`: Periodic share refreshing
//! - `aggregation`: Secure gradient aggregation with compression

pub mod activation;
pub mod aggregation;
pub mod arithmetic;
pub mod comparison;
pub mod matmul;
pub mod normalization;
pub mod proved_arithmetic;
pub mod reshare;

pub use activation::SecureActivation;
pub use aggregation::{
    AggregationConfig, AggregationContribution, AggregationResult, AggregationStatsSnapshot,
    CompressedGradient, CompressionConfig, DropoutTolerantAggregator, GradientCompressor,
    SecureAggregator, WeightedAggregator,
};
pub use arithmetic::SecureArithmetic;
pub use comparison::{
    BitDecomposition, ComparisonConfig, GarbledComparison, SecureComparison, SecureReLUWithGradient,
};
pub use matmul::SecureMatmul;
pub use normalization::SecureNormalization;
pub use proved_arithmetic::{
    BeaverWitness, OperationWitness, ProvedArithmetic, SharedWitnessCapture, WitnessCapture,
    WitnessSummary, WitnessedOperation, create_shared_capture,
};
pub use reshare::Resharing;

use crate::field::Fr;
use rand::Rng;

/// Creates additive shares of f64 values using a cryptographically secure RNG.
///
/// Each value is converted to Fr, then split into `num_parties` additive shares
/// such that the shares sum to the original value. The last party gets the
/// residual share (value minus sum of random shares).
pub(crate) fn reshare_values(values: &[f64], num_parties: usize, rng: &mut impl Rng) -> Vec<Vec<Fr>> {
    let dim = values.len();
    let mut shares: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; num_parties];

    for d in 0..dim {
        let target = Fr::from_f64(values[d]);
        let mut sum = Fr::ZERO;
        for i in 0..num_parties - 1 {
            let r = Fr::from_f64(rng.gen_range(-100.0..100.0));
            shares[i][d] = r.clone();
            sum = Fr::add(&sum, &r);
        }
        shares[num_parties - 1][d] = Fr::sub(&target, &sum);
    }

    shares
}
