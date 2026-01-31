//! Prover Backend Trait Definitions.
//!
//! This module defines the core traits that all proving backends must implement.

use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2_proofs::arithmetic::Field;
use thiserror::Error;

/// Errors from backend operations.
#[derive(Error, Debug)]
pub enum BackendError {
    #[error("Backend not initialized")]
    NotInitialized,

    #[error("Proving failed: {0}")]
    ProvingFailed(String),

    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    #[error("Unsupported circuit type: {0}")]
    UnsupportedCircuit(String),

    #[error("Invalid proof format: {0}")]
    InvalidProof(String),

    #[error("Setup failed: {0}")]
    SetupFailed(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
}

/// Result type for backend operations.
pub type BackendResult<T> = Result<T, BackendError>;

/// Identifier for a backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendId {
    /// Halo2 Plonk backend.
    Halo2,
    /// GKR sumcheck backend.
    GKR,
    /// Hybrid (GKR + Halo2 aggregation).
    Hybrid,
}

/// Configuration for a backend.
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Backend identifier.
    pub id: BackendId,
    /// K parameter (for Halo2-like systems).
    pub k: u32,
    /// Whether to enable zero-knowledge.
    pub zero_knowledge: bool,
    /// Whether to enable parallel proving.
    pub parallel: bool,
    /// Custom parameters as key-value pairs.
    pub custom: std::collections::HashMap<String, String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            id: BackendId::Halo2,
            k: 14,
            zero_knowledge: true,
            parallel: true,
            custom: std::collections::HashMap::new(),
        }
    }
}

/// Description of a circuit for backend selection.
#[derive(Debug, Clone)]
pub struct CircuitDescription {
    /// Total number of gates.
    pub num_gates: usize,
    /// Circuit depth (for layered circuits).
    pub depth: usize,
    /// Whether the circuit is layered.
    pub is_layered: bool,
    /// Whether on-chain verification is required.
    pub needs_onchain_verification: bool,
}

impl Default for CircuitDescription {
    fn default() -> Self {
        Self {
            num_gates: 0,
            depth: 0,
            is_layered: false,
            needs_onchain_verification: false,
        }
    }
}

/// Witness data for proving.
#[derive(Debug, Clone)]
pub struct WitnessData {
    /// Input values.
    pub inputs: Vec<Fr>,
    /// Auxiliary witness values.
    pub aux: Vec<Fr>,
    /// Public inputs.
    pub public_inputs: Vec<Fr>,
}

impl WitnessData {
    /// Creates witness data from just inputs.
    pub fn from_inputs(inputs: Vec<Fr>) -> Self {
        Self {
            inputs,
            aux: vec![],
            public_inputs: vec![],
        }
    }

    /// Creates witness data with public inputs.
    pub fn with_public_inputs(inputs: Vec<Fr>, public_inputs: Vec<Fr>) -> Self {
        Self {
            inputs,
            aux: vec![],
            public_inputs,
        }
    }
}

/// Proof data from a backend.
#[derive(Debug, Clone)]
pub struct ProofData {
    /// Backend that produced this proof.
    pub backend: BackendId,
    /// Serialized proof bytes.
    pub proof_bytes: Vec<u8>,
    /// Public inputs used in the proof.
    pub public_inputs: Vec<Fr>,
    /// Additional metadata.
    pub metadata: Vec<u8>,
}

impl ProofData {
    /// Returns the proof size in bytes.
    pub fn size(&self) -> usize {
        self.proof_bytes.len()
    }

    /// Checks if this proof came from the specified backend.
    pub fn is_from(&self, backend: BackendId) -> bool {
        self.backend == backend
    }
}

/// Capabilities of a proving backend.
pub struct BackendCapabilities {
    /// Whether the backend supports zero-knowledge.
    pub supports_zk: bool,
    /// Whether the backend supports proof aggregation.
    pub supports_aggregation: bool,
    /// Whether proofs can be verified on-chain (EVM).
    pub supports_onchain_verification: bool,
    /// Whether the backend supports GPU acceleration.
    pub supports_gpu: bool,
    /// Maximum circuit size (in gates).
    pub max_circuit_size: usize,
    /// Function to estimate proof size from circuit depth.
    pub proof_size_estimate_fn: Box<dyn Fn(usize) -> usize + Send + Sync>,
    /// Function to estimate proving time from gate count (microseconds).
    pub proving_time_estimate_fn: Box<dyn Fn(usize) -> usize + Send + Sync>,
}

impl std::fmt::Debug for BackendCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendCapabilities")
            .field("supports_zk", &self.supports_zk)
            .field("supports_aggregation", &self.supports_aggregation)
            .field("supports_onchain_verification", &self.supports_onchain_verification)
            .field("supports_gpu", &self.supports_gpu)
            .field("max_circuit_size", &self.max_circuit_size)
            .finish()
    }
}

/// Trait for proving backends.
///
/// All proving backends must implement this trait to integrate with HELIX.
pub trait ProverBackend: Send + Sync {
    /// Circuit type this backend accepts.
    type Circuit;

    /// Returns the backend identifier.
    fn id(&self) -> BackendId;

    /// Returns the backend's capabilities.
    fn capabilities(&self) -> BackendCapabilities;

    /// Performs any necessary setup for the given circuit.
    fn setup(&mut self, circuit: &Self::Circuit) -> BackendResult<()>;

    /// Generates a proof for the given circuit and witness.
    fn prove(
        &self,
        circuit: &Self::Circuit,
        witness: &WitnessData,
    ) -> BackendResult<ProofData>;

    /// Verifies a proof.
    fn verify(
        &self,
        proof: &ProofData,
        public_inputs: &[Fr],
    ) -> BackendResult<bool>;

    /// Returns whether this backend is ready to prove.
    fn is_ready(&self) -> bool;

    /// Resets the backend state.
    fn reset(&mut self);
}

/// Trait for verification backends.
///
/// Separate from ProverBackend for cases where we only need verification.
pub trait VerifierBackend: Send + Sync {
    /// Returns the backend identifier.
    fn id(&self) -> BackendId;

    /// Verifies a proof.
    fn verify(
        &self,
        proof: &ProofData,
        public_inputs: &[Fr],
    ) -> BackendResult<bool>;

    /// Verifies a batch of proofs.
    fn verify_batch(
        &self,
        proofs: &[ProofData],
        public_inputs: &[Vec<Fr>],
    ) -> BackendResult<Vec<bool>> {
        proofs.iter()
            .zip(public_inputs.iter())
            .map(|(proof, pi)| self.verify(proof, pi))
            .collect()
    }
}

/// Trait for backends that support proof aggregation.
pub trait AggregatableBackend: ProverBackend {
    /// Aggregated proof type.
    type AggregatedProof;

    /// Aggregates multiple proofs into one.
    fn aggregate(&self, proofs: &[ProofData]) -> BackendResult<Self::AggregatedProof>;

    /// Verifies an aggregated proof.
    fn verify_aggregated(
        &self,
        proof: &Self::AggregatedProof,
        public_inputs: &[Vec<Fr>],
    ) -> BackendResult<bool>;
}

/// Trait for backends that support GPU acceleration.
pub trait GpuAcceleratedBackend: ProverBackend {
    /// Checks if GPU is available.
    fn is_gpu_available(&self) -> bool;

    /// Enables GPU acceleration.
    fn enable_gpu(&mut self) -> BackendResult<()>;

    /// Disables GPU acceleration.
    fn disable_gpu(&mut self);

    /// Returns GPU statistics.
    fn gpu_stats(&self) -> Option<GpuStats>;
}

/// Statistics from GPU operations.
#[derive(Debug, Clone, Default)]
pub struct GpuStats {
    /// Total GPU time in microseconds.
    pub total_gpu_time_us: u64,
    /// Number of kernel dispatches.
    pub num_dispatches: usize,
    /// Peak memory usage in bytes.
    pub peak_memory: usize,
    /// Device name.
    pub device_name: String,
}

/// Extension trait for converting between backends.
pub trait BackendConvertible {
    /// Converts a proof to be compatible with another backend.
    fn convert_proof(
        &self,
        proof: &ProofData,
        target_backend: BackendId,
    ) -> BackendResult<ProofData>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_config_default() {
        let config = BackendConfig::default();
        assert_eq!(config.id, BackendId::Halo2);
        assert_eq!(config.k, 14);
        assert!(config.zero_knowledge);
    }

    #[test]
    fn test_witness_data() {
        let inputs = vec![Fr::from(1u64), Fr::from(2u64)];
        let witness = WitnessData::from_inputs(inputs.clone());

        assert_eq!(witness.inputs.len(), 2);
        assert!(witness.aux.is_empty());
        assert!(witness.public_inputs.is_empty());
    }

    #[test]
    fn test_proof_data() {
        let proof = ProofData {
            backend: BackendId::GKR,
            proof_bytes: vec![1, 2, 3, 4],
            public_inputs: vec![Fr::from(42u64)],
            metadata: vec![],
        };

        assert_eq!(proof.size(), 4);
        assert!(proof.is_from(BackendId::GKR));
        assert!(!proof.is_from(BackendId::Halo2));
    }

    #[test]
    fn test_circuit_description() {
        let desc = CircuitDescription {
            num_gates: 1000,
            depth: 5,
            is_layered: true,
            needs_onchain_verification: false,
        };

        assert!(desc.is_layered);
        assert_eq!(desc.depth, 5);
    }
}
