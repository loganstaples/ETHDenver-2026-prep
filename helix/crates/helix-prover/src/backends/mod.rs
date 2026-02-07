//! Prover Backends for HELIX.
//!
//! This module provides a unified interface for different proving backends.
//! Currently supported backends:
//!
//! - **Halo2**: Plonk-based proofs with KZG commitments (existing)
//! - **GKR**: Orion-style GKR proofs (new, optimized for neural networks)
//!
//! ## Backend Selection
//!
//! The `BackendSelector` automatically chooses the optimal backend based on:
//! - Circuit structure (layered vs. arbitrary)
//! - Circuit size (GKR better for large circuits)
//! - Available hardware (Metal GPU acceleration)
//! - Proof requirements (on-chain verification needs)

pub mod r#trait;

pub use r#trait::{
    ProverBackend, VerifierBackend, BackendCapabilities,
    BackendId, BackendConfig, BackendResult, BackendError,
    ProofData, CircuitDescription, WitnessData,
};

use crate::gkr::{GKRProver, GKRVerifier, GKRConfig, GKRProof, LayeredCircuit};
use crate::pipeline::ProverPipeline;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2_proofs::plonk::Circuit;

/// Backend type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// Halo2 Plonk backend.
    Halo2,
    /// GKR sumcheck-based backend.
    GKR,
    /// Automatic selection based on circuit.
    Auto,
}

impl Default for BackendType {
    fn default() -> Self {
        BackendType::Auto
    }
}

/// Unified prover configuration.
#[derive(Debug, Clone)]
pub struct UnifiedProverConfig {
    /// Preferred backend.
    pub backend: BackendType,
    /// GKR-specific configuration.
    pub gkr_config: GKRConfig,
    /// Halo2 K parameter (circuit size).
    pub halo2_k: u32,
    /// Whether to enable parallelism.
    pub parallel: bool,
    /// Whether to use GPU acceleration.
    pub use_gpu: bool,
    /// Threshold for backend selection (gate count).
    pub gkr_threshold: usize,
}

impl Default for UnifiedProverConfig {
    fn default() -> Self {
        Self {
            backend: BackendType::Auto,
            gkr_config: GKRConfig::default(),
            halo2_k: 14,
            parallel: true,
            use_gpu: cfg!(target_os = "macos"),
            gkr_threshold: 10000, // Use GKR for circuits with >10K gates
        }
    }
}

/// Backend selector that chooses optimal proving strategy.
pub struct BackendSelector {
    config: UnifiedProverConfig,
}

impl BackendSelector {
    /// Creates a new backend selector.
    pub fn new(config: UnifiedProverConfig) -> Self {
        Self { config }
    }

    /// Selects the optimal backend for a circuit description.
    pub fn select(&self, circuit: &CircuitDescription) -> BackendType {
        match self.config.backend {
            BackendType::Auto => self.auto_select(circuit),
            other => other,
        }
    }

    /// Automatic backend selection based on circuit properties.
    fn auto_select(&self, circuit: &CircuitDescription) -> BackendType {
        // Highest priority: On-chain verification requires Halo2
        // (GKR proofs need aggregation to Halo2 for EVM verification)
        if circuit.needs_onchain_verification {
            return BackendType::Halo2;
        }

        // Use GKR for:
        // - Layered circuits (natural for GKR)
        // - Large circuits (better asymptotic complexity)
        // - When GPU is available (GKR parallelizes well)
        if circuit.is_layered && circuit.num_gates > self.config.gkr_threshold {
            return BackendType::GKR;
        }

        // Use Halo2 for:
        // - Small circuits (lower setup overhead)
        // - Non-layered circuits
        if circuit.num_gates <= self.config.gkr_threshold {
            return BackendType::Halo2;
        }

        // Default to Halo2 for compatibility
        BackendType::Halo2
    }

    /// Returns capabilities of the selected backend.
    pub fn capabilities(&self, backend: BackendType) -> BackendCapabilities {
        match backend {
            BackendType::Halo2 => BackendCapabilities {
                supports_zk: true,
                supports_aggregation: true,
                supports_onchain_verification: true,
                supports_gpu: false, // Not yet
                max_circuit_size: 1 << 20,
                proof_size_estimate_fn: Box::new(|_| 1024), // ~1KB
                proving_time_estimate_fn: Box::new(|gates| gates * 10), // μs
            },
            BackendType::GKR => BackendCapabilities {
                supports_zk: true,
                supports_aggregation: true,
                supports_onchain_verification: false, // Needs aggregation to Halo2
                supports_gpu: true,
                max_circuit_size: 1 << 30,
                proof_size_estimate_fn: Box::new(|depth| depth * 32 * 10), // ~10 elements per layer
                proving_time_estimate_fn: Box::new(|gates| gates), // Linear!
            },
            BackendType::Auto => self.capabilities(BackendType::Halo2), // Default
        }
    }
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self::new(UnifiedProverConfig::default())
    }
}

/// Unified prover that can use either backend.
pub struct UnifiedProver {
    /// Selector for backend choice.
    selector: BackendSelector,
    /// GKR prover instance.
    gkr_prover: Option<GKRProver>,
}

impl UnifiedProver {
    /// Creates a new unified prover.
    pub fn new(config: UnifiedProverConfig) -> Self {
        let gkr_prover = Some(GKRProver::new(config.gkr_config.clone()));

        Self {
            selector: BackendSelector::new(config),
            gkr_prover,
        }
    }

    /// Proves a layered circuit using the appropriate backend.
    pub fn prove_layered(
        &mut self,
        circuit: &LayeredCircuit,
        inputs: &[Fr],
    ) -> BackendResult<ProofData> {
        let desc = CircuitDescription {
            num_gates: circuit.num_gates(),
            depth: circuit.depth(),
            is_layered: true,
            needs_onchain_verification: false,
        };

        let backend = self.selector.select(&desc);

        match backend {
            BackendType::GKR | BackendType::Auto => {
                let prover = self.gkr_prover.as_mut()
                    .ok_or(BackendError::NotInitialized)?;

                let proof = prover.prove(circuit, inputs)
                    .map_err(|e| BackendError::ProvingFailed(e.to_string()))?;

                Ok(ProofData {
                    backend: BackendId::GKR,
                    proof_bytes: proof.to_bytes(),
                    public_inputs: inputs.to_vec(),
                    metadata: vec![],
                })
            }
            BackendType::Halo2 => {
                // Would need to convert layered circuit to Halo2 circuit
                Err(BackendError::UnsupportedCircuit(
                    "Layered circuits not directly supported by Halo2".to_string()
                ))
            }
        }
    }

    /// Returns the selector.
    pub fn selector(&self) -> &BackendSelector {
        &self.selector
    }

    /// Returns mutable access to the GKR prover.
    pub fn gkr_prover_mut(&mut self) -> Option<&mut GKRProver> {
        self.gkr_prover.as_mut()
    }
}

impl Default for UnifiedProver {
    fn default() -> Self {
        Self::new(UnifiedProverConfig::default())
    }
}

/// Benchmarking utilities for backend comparison.
pub mod benchmark {
    use super::*;
    use std::time::{Duration, Instant};

    /// Results from benchmarking a backend.
    #[derive(Debug, Clone)]
    pub struct BackendBenchmark {
        /// Backend type.
        pub backend: BackendType,
        /// Setup time.
        pub setup_time: Duration,
        /// Average proving time.
        pub proving_time: Duration,
        /// Average verification time.
        pub verification_time: Duration,
        /// Proof size in bytes.
        pub proof_size: usize,
        /// Number of iterations.
        pub iterations: usize,
    }

    /// Runs a benchmark comparing GKR and Halo2.
    pub fn compare_backends(
        circuit: &LayeredCircuit,
        inputs: &[Fr],
        iterations: usize,
    ) -> Vec<BackendBenchmark> {
        let mut results = Vec::new();

        // Benchmark GKR
        let gkr_config = GKRConfig::for_testing();
        let mut gkr_prover = GKRProver::new(gkr_config.clone());
        let gkr_verifier = GKRVerifier::new(gkr_config);

        let start = Instant::now();
        let mut total_proving = Duration::ZERO;
        let mut total_verification = Duration::ZERO;
        let mut proof_size = 0;
        let mut last_proof = None;

        for _ in 0..iterations {
            let prove_start = Instant::now();
            if let Ok(proof) = gkr_prover.prove(circuit, inputs) {
                total_proving += prove_start.elapsed();
                proof_size = proof.size_bytes();
                last_proof = Some(proof);
            }
        }

        // Benchmark verification using the last generated proof
        if let Some(ref proof) = last_proof {
            for _ in 0..iterations {
                let verify_start = Instant::now();
                let _ = gkr_verifier.verify(proof, circuit, inputs);
                total_verification += verify_start.elapsed();
            }
        }

        results.push(BackendBenchmark {
            backend: BackendType::GKR,
            setup_time: Duration::ZERO, // GKR has no setup
            proving_time: total_proving / iterations as u32,
            verification_time: total_verification / iterations as u32,
            proof_size,
            iterations,
        });

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::layered_circuit::{CircuitBuilder, Gate, Wire};

    fn simple_circuit() -> LayeredCircuit {
        let mut builder = CircuitBuilder::new(2);
        builder.add_gate(Gate::add(Wire::input(0), Wire::input(1)));
        builder.finish_layer();
        builder.build()
    }

    #[test]
    fn test_backend_selector_default() {
        let selector = BackendSelector::default();

        let small_circuit = CircuitDescription {
            num_gates: 100,
            depth: 3,
            is_layered: true,
            needs_onchain_verification: false,
        };

        // Small circuits should use Halo2
        assert_eq!(selector.select(&small_circuit), BackendType::Halo2);
    }

    #[test]
    fn test_backend_selector_large_layered() {
        let selector = BackendSelector::default();

        let large_circuit = CircuitDescription {
            num_gates: 100000,
            depth: 10,
            is_layered: true,
            needs_onchain_verification: false,
        };

        // Large layered circuits should use GKR
        assert_eq!(selector.select(&large_circuit), BackendType::GKR);
    }

    #[test]
    fn test_backend_selector_onchain() {
        let selector = BackendSelector::default();

        let onchain_circuit = CircuitDescription {
            num_gates: 100000,
            depth: 10,
            is_layered: true,
            needs_onchain_verification: true,
        };

        // On-chain verification needs Halo2
        assert_eq!(selector.select(&onchain_circuit), BackendType::Halo2);
    }

    #[test]
    fn test_unified_prover_gkr() {
        let config = UnifiedProverConfig {
            backend: BackendType::GKR,
            ..Default::default()
        };

        let mut prover = UnifiedProver::new(config);
        let circuit = simple_circuit();
        let inputs = vec![Fr::from(3u64), Fr::from(5u64)];

        let result = prover.prove_layered(&circuit, &inputs);
        assert!(result.is_ok());

        let proof = result.unwrap();
        assert_eq!(proof.backend, BackendId::GKR);
        assert!(!proof.proof_bytes.is_empty());
    }

    #[test]
    fn test_backend_capabilities() {
        let selector = BackendSelector::default();

        let halo2_caps = selector.capabilities(BackendType::Halo2);
        assert!(halo2_caps.supports_onchain_verification);
        assert!(!halo2_caps.supports_gpu);

        let gkr_caps = selector.capabilities(BackendType::GKR);
        assert!(!gkr_caps.supports_onchain_verification);
        assert!(gkr_caps.supports_gpu);
    }
}
