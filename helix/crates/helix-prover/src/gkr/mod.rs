//! ZK-GKR Prover - Orion-style GKR protocol implementation.
//!
//! This module implements a Zero-Knowledge variant of the GKR (Goldwasser-Kalai-Rothblum)
//! protocol optimized for neural network circuit verification. The GKR protocol is based
//! on the sumcheck protocol and provides O(d log g) prover complexity for circuits of
//! depth d with g gates per layer.
//!
//! ## Key Components
//!
//! - **Sumcheck Protocol**: Core interactive protocol for proving polynomial identities
//! - **Multilinear Polynomials**: Efficient representation and evaluation of multilinear extensions
//! - **Layered Circuits**: Abstraction for representing neural networks as arithmetic circuits
//! - **ZK Layer**: Polynomial masking for zero-knowledge property
//! - **GKR Prover**: Complete prover implementation combining all components
//!
//! ## Performance Characteristics
//!
//! The GKR protocol has several advantages over Plonk/Halo2 for neural network proofs:
//! - O(n) prover time vs O(n log n) for FFT-based systems
//! - No trusted setup required (uses Fiat-Shamir for non-interactivity)
//! - Natural parallelization across circuit layers
//! - Optimal for matrix multiplication and element-wise operations
//!
//! ## Usage
//!
//! ```ignore
//! use helix_prover::gkr::{GKRProver, LayeredCircuit, GKRConfig};
//!
//! // Create a layered circuit from neural network operations
//! let circuit = LayeredCircuit::from_matmul(weights, inputs);
//!
//! // Configure and create prover
//! let config = GKRConfig::default();
//! let prover = GKRProver::new(config);
//!
//! // Generate proof
//! let proof = prover.prove(&circuit)?;
//!
//! // Verify
//! assert!(prover.verify(&proof, &circuit.output())?);
//! ```

pub mod sumcheck;
pub mod multilinear;
pub mod layered_circuit;
pub mod zk_layer;
pub mod prover;

// Re-exports
pub use sumcheck::{SumcheckProver, SumcheckVerifier, SumcheckProof, SumcheckRound};
pub use multilinear::{
    MultilinearPolynomial, MultilinearExtension, DenseMultilinear, SparseMultilinear,
    EvaluationDomain, PolynomialCommitment,
};
pub use layered_circuit::{
    LayeredCircuit, CircuitLayer, Gate, GateType, Wire, WireType,
    CircuitBuilder, NeuralNetworkCircuit,
};
pub use zk_layer::{
    ZKMask, ZKRandomness, ZKConfig, MaskedPolynomial, ZeroKnowledgeLayer,
};
pub use prover::{
    GKRProver, GKRVerifier, GKRProof, GKRConfig, GKRLayerProof,
    ProverState, VerifierState, ProofTranscript,
};

use helix_circuits::halo2curves::bn256::Fr;
use thiserror::Error;

/// Errors that can occur during GKR proving/verification.
#[derive(Error, Debug)]
pub enum GKRError {
    #[error("Sumcheck verification failed at round {round}")]
    SumcheckFailed { round: usize },

    #[error("Invalid circuit structure: {0}")]
    InvalidCircuit(String),

    #[error("Polynomial evaluation mismatch: expected {expected:?}, got {actual:?}")]
    EvaluationMismatch { expected: Fr, actual: Fr },

    #[error("Invalid proof format: {0}")]
    InvalidProof(String),

    #[error("Metal GPU error: {0}")]
    MetalError(String),

    #[error("Commitment error: {0}")]
    CommitmentError(String),

    #[error("Transcript error: {0}")]
    TranscriptError(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Insufficient randomness")]
    InsufficientRandomness,
}

/// Result type for GKR operations.
pub type GKRResult<T> = Result<T, GKRError>;

/// Field element type used throughout GKR (BN254 scalar field).
pub type FieldElement = Fr;

/// Configuration for the GKR protocol.
#[derive(Debug, Clone)]
pub struct ProtocolConfig {
    /// Number of parallel threads to use.
    pub num_threads: usize,
    /// Whether to use Metal GPU acceleration (macOS only).
    pub use_metal: bool,
    /// Security parameter (number of bits).
    pub security_bits: usize,
    /// Whether to enable zero-knowledge.
    pub zero_knowledge: bool,
    /// Batch size for parallel operations.
    pub batch_size: usize,
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            use_metal: cfg!(target_os = "macos"),
            security_bits: 128,
            zero_knowledge: true,
            batch_size: 256,
        }
    }
}

/// Statistics from a GKR proving session.
#[derive(Debug, Clone, Default)]
pub struct GKRStats {
    /// Number of circuit layers proved.
    pub num_layers: usize,
    /// Total number of sumcheck rounds.
    pub total_rounds: usize,
    /// Proof size in bytes.
    pub proof_size: usize,
    /// Time spent on sumcheck (microseconds).
    pub sumcheck_time_us: u64,
    /// Time spent on polynomial evaluations (microseconds).
    pub eval_time_us: u64,
    /// Time spent on ZK masking (microseconds).
    pub zk_time_us: u64,
    /// Whether Metal acceleration was used.
    pub used_metal: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use helix_circuits::halo2_proofs::arithmetic::Field;

    #[test]
    fn test_default_config() {
        let config = ProtocolConfig::default();
        assert!(config.num_threads > 0);
        assert!(config.security_bits >= 128);
    }

    #[test]
    fn test_gkr_stats_default() {
        let stats = GKRStats::default();
        assert_eq!(stats.num_layers, 0);
        assert_eq!(stats.proof_size, 0);
    }
}
