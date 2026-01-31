//! Test fixtures for HELIX integration tests.
//!
//! Provides deterministic test data for models, training samples, and MPC configurations.

use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_mpc::types::{MPCConfig, PartyId};
use helix_prover::{TrainingWeights, V2ProverConfig};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::HashMap;

/// Standard model dimensions for testing.
pub struct ModelDimensions {
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
}

impl ModelDimensions {
    /// Tiny model for quick tests (2x2x1).
    pub const fn tiny() -> Self {
        Self { d_in: 2, d_hid: 2, d_out: 1 }
    }

    /// Small model for integration tests (4x8x2).
    pub const fn small() -> Self {
        Self { d_in: 4, d_hid: 8, d_out: 2 }
    }

    /// Medium model for performance tests (16x32x8).
    pub const fn medium() -> Self {
        Self { d_in: 16, d_hid: 32, d_out: 8 }
    }

    /// Large model for stress tests (64x128x16).
    pub const fn large() -> Self {
        Self { d_in: 64, d_hid: 128, d_out: 16 }
    }
}

/// Generates deterministic random field elements.
pub struct DeterministicRng {
    rng: ChaCha20Rng,
}

impl DeterministicRng {
    /// Creates a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    /// Generates a random field element.
    pub fn next_fr(&mut self) -> Fr {
        Fr::random(&mut self.rng)
    }

    /// Generates a vector of random field elements.
    pub fn next_fr_vec(&mut self, len: usize) -> Vec<Fr> {
        (0..len).map(|_| self.next_fr()).collect()
    }

    /// Generates a small random field element (for numerical stability).
    pub fn next_small_fr(&mut self) -> Fr {
        Fr::from(rand::Rng::gen_range(&mut self.rng, 1u64..100))
    }

    /// Generates a vector of small random field elements.
    pub fn next_small_fr_vec(&mut self, len: usize) -> Vec<Fr> {
        (0..len).map(|_| self.next_small_fr()).collect()
    }

    /// Generates a random f64 in [-1, 1].
    pub fn next_f64(&mut self) -> f64 {
        rand::Rng::gen_range(&mut self.rng, -1.0..1.0)
    }

    /// Generates a vector of random f64 values.
    pub fn next_f64_vec(&mut self, len: usize) -> Vec<f64> {
        (0..len).map(|_| self.next_f64()).collect()
    }
}

/// Test model weights fixture.
#[derive(Debug, Clone)]
pub struct TestModelWeights {
    pub dims: ModelDimensions,
    pub w1: Vec<Fr>,
    pub b1: Vec<Fr>,
    pub w2: Vec<Fr>,
    pub b2: Vec<Fr>,
}

impl TestModelWeights {
    /// Creates a model with deterministic random weights.
    pub fn random(dims: ModelDimensions, seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed);
        Self {
            w1: rng.next_small_fr_vec(dims.d_hid * dims.d_in),
            b1: vec![Fr::zero(); dims.d_hid],
            w2: rng.next_small_fr_vec(dims.d_out * dims.d_hid),
            b2: vec![Fr::zero(); dims.d_out],
            dims,
        }
    }

    /// Creates a model with known weights for verification.
    pub fn known(dims: ModelDimensions) -> Self {
        let w1: Vec<Fr> = (0..dims.d_hid * dims.d_in)
            .map(|i| Fr::from((i % 5 + 1) as u64))
            .collect();
        let b1 = vec![Fr::zero(); dims.d_hid];
        let w2: Vec<Fr> = (0..dims.d_out * dims.d_hid)
            .map(|i| Fr::from((i % 3 + 1) as u64))
            .collect();
        let b2 = vec![Fr::zero(); dims.d_out];
        Self { dims, w1, b1, w2, b2 }
    }

    /// Converts to helix-prover TrainingWeights.
    pub fn to_training_weights(&self) -> TrainingWeights {
        TrainingWeights::new(
            self.dims.d_in,
            self.dims.d_hid,
            self.dims.d_out,
            self.w1.clone(),
            self.b1.clone(),
            self.w2.clone(),
            self.b2.clone(),
        )
    }
}

/// Test training sample.
#[derive(Debug, Clone)]
pub struct TestSample {
    pub x: Vec<Fr>,
    pub target: Vec<Fr>,
}

impl TestSample {
    /// Creates a random sample.
    pub fn random(d_in: usize, d_out: usize, seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed);
        Self {
            x: rng.next_small_fr_vec(d_in),
            target: rng.next_small_fr_vec(d_out),
        }
    }

    /// Creates a sample with known values.
    pub fn known(d_in: usize, d_out: usize) -> Self {
        Self {
            x: (0..d_in).map(|i| Fr::from((i + 1) as u64)).collect(),
            target: (0..d_out).map(|i| Fr::from((i + 5) as u64)).collect(),
        }
    }
}

/// Test training dataset.
#[derive(Debug, Clone)]
pub struct TestDataset {
    pub samples: Vec<TestSample>,
}

impl TestDataset {
    /// Creates a dataset with the specified number of samples.
    pub fn new(d_in: usize, d_out: usize, num_samples: usize, seed: u64) -> Self {
        let samples = (0..num_samples)
            .map(|i| TestSample::random(d_in, d_out, seed + i as u64))
            .collect();
        Self { samples }
    }

    /// Returns samples as (x, target) tuples for prover.
    pub fn to_tuples(&self) -> Vec<(Vec<Fr>, Vec<Fr>)> {
        self.samples
            .iter()
            .map(|s| (s.x.clone(), s.target.clone()))
            .collect()
    }
}

/// MPC test configuration.
#[derive(Debug, Clone)]
pub struct MPCTestConfig {
    pub num_parties: usize,
    pub threshold: usize,
    pub party_ids: Vec<PartyId>,
    pub learning_rate: f64,
    pub max_gradient_norm: f64,
}

impl MPCTestConfig {
    /// Standard 3-party configuration.
    pub fn three_party() -> Self {
        Self {
            num_parties: 3,
            threshold: 2,
            party_ids: (0..3).map(PartyId::from_index).collect(),
            learning_rate: 0.01,
            max_gradient_norm: 100.0,
        }
    }

    /// Two-party configuration for simpler tests.
    pub fn two_party() -> Self {
        Self {
            num_parties: 2,
            threshold: 2,
            party_ids: (0..2).map(PartyId::from_index).collect(),
            learning_rate: 0.01,
            max_gradient_norm: 100.0,
        }
    }

    /// Five-party configuration for robustness tests.
    pub fn five_party() -> Self {
        Self {
            num_parties: 5,
            threshold: 3,
            party_ids: (0..5).map(PartyId::from_index).collect(),
            learning_rate: 0.01,
            max_gradient_norm: 100.0,
        }
    }

    /// Converts to helix-mpc MPCConfig.
    pub fn to_mpc_config(&self) -> MPCConfig {
        MPCConfig {
            num_parties: self.num_parties,
            reshare_interval: 50,
            verify_shares: true,
            ..MPCConfig::default()
        }
    }
}

/// Prover test configuration.
#[derive(Debug, Clone)]
pub struct ProverTestConfig {
    pub k: u32,
    pub relu_range: usize,
    pub exp_range: usize,
    pub use_freivalds: bool,
}

impl ProverTestConfig {
    /// Standard configuration for tests.
    pub fn standard() -> Self {
        Self {
            k: 14,
            relu_range: 128,
            exp_range: 256,
            use_freivalds: true,
        }
    }

    /// Minimal configuration for quick tests.
    pub fn minimal() -> Self {
        Self {
            k: 12,
            relu_range: 64,
            exp_range: 128,
            use_freivalds: true,
        }
    }

    /// Large configuration for stress tests.
    pub fn large() -> Self {
        Self {
            k: 16,
            relu_range: 256,
            exp_range: 512,
            use_freivalds: true,
        }
    }

    /// Converts to V2ProverConfig.
    pub fn to_v2_config(&self) -> V2ProverConfig {
        V2ProverConfig {
            k: self.k,
            relu_range: self.relu_range,
            exp_range: self.exp_range,
            exp_scale: 1000,
            use_freivalds: self.use_freivalds,
            base_error: Fr::from(1u64),
        }
    }
}

/// Known-good training result for verification.
#[derive(Debug, Clone)]
pub struct KnownGoodResult {
    pub initial_loss: f64,
    pub final_loss: f64,
    pub loss_reduction_pct: f64,
    pub num_steps: usize,
}

impl KnownGoodResult {
    /// Expected result for tiny model XOR problem.
    pub fn xor_tiny() -> Self {
        Self {
            initial_loss: 0.5,
            final_loss: 0.1,
            loss_reduction_pct: 80.0,
            num_steps: 100,
        }
    }

    /// Expected result for small model regression.
    pub fn regression_small() -> Self {
        Self {
            initial_loss: 1.0,
            final_loss: 0.3,
            loss_reduction_pct: 70.0,
            num_steps: 50,
        }
    }
}

/// Checkpoint fixture for resume tests.
#[derive(Debug, Clone)]
pub struct CheckpointFixture {
    pub step: u64,
    pub weights: TestModelWeights,
    pub loss: f64,
    pub error_bound: Fr,
}

impl CheckpointFixture {
    /// Creates a checkpoint at the given step.
    pub fn at_step(step: u64, dims: ModelDimensions, seed: u64) -> Self {
        Self {
            step,
            weights: TestModelWeights::random(dims, seed),
            loss: 0.5 - (step as f64 * 0.01),
            error_bound: Fr::from(step * 10),
        }
    }
}

/// Creates a comprehensive test scenario.
#[derive(Debug)]
pub struct TestScenario {
    pub name: String,
    pub dims: ModelDimensions,
    pub model: TestModelWeights,
    pub dataset: TestDataset,
    pub mpc_config: MPCTestConfig,
    pub prover_config: ProverTestConfig,
    pub expected_result: KnownGoodResult,
}

impl TestScenario {
    /// Standard integration test scenario.
    pub fn standard() -> Self {
        let dims = ModelDimensions::tiny();
        Self {
            name: "standard_integration".to_string(),
            model: TestModelWeights::known(dims),
            dataset: TestDataset::new(dims.d_in, dims.d_out, 10, 42),
            mpc_config: MPCTestConfig::three_party(),
            prover_config: ProverTestConfig::standard(),
            expected_result: KnownGoodResult::xor_tiny(),
            dims,
        }
    }

    /// Quick test scenario for CI.
    pub fn quick() -> Self {
        let dims = ModelDimensions::tiny();
        Self {
            name: "quick_ci".to_string(),
            model: TestModelWeights::known(dims),
            dataset: TestDataset::new(dims.d_in, dims.d_out, 3, 42),
            mpc_config: MPCTestConfig::two_party(),
            prover_config: ProverTestConfig::minimal(),
            expected_result: KnownGoodResult::xor_tiny(),
            dims,
        }
    }

    /// Stress test scenario.
    pub fn stress() -> Self {
        let dims = ModelDimensions::small();
        Self {
            name: "stress_test".to_string(),
            model: TestModelWeights::random(dims, 42),
            dataset: TestDataset::new(dims.d_in, dims.d_out, 50, 42),
            mpc_config: MPCTestConfig::five_party(),
            prover_config: ProverTestConfig::standard(),
            expected_result: KnownGoodResult::regression_small(),
            dims,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_rng() {
        let mut rng1 = DeterministicRng::new(42);
        let mut rng2 = DeterministicRng::new(42);

        for _ in 0..10 {
            assert_eq!(rng1.next_fr(), rng2.next_fr());
        }
    }

    #[test]
    fn test_model_weights_dimensions() {
        let dims = ModelDimensions::small();
        let weights = TestModelWeights::random(dims, 42);

        assert_eq!(weights.w1.len(), 8 * 4);
        assert_eq!(weights.b1.len(), 8);
        assert_eq!(weights.w2.len(), 2 * 8);
        assert_eq!(weights.b2.len(), 2);
    }

    #[test]
    fn test_dataset_generation() {
        let dataset = TestDataset::new(4, 2, 10, 42);
        assert_eq!(dataset.samples.len(), 10);

        for sample in &dataset.samples {
            assert_eq!(sample.x.len(), 4);
            assert_eq!(sample.target.len(), 2);
        }
    }

    #[test]
    fn test_mpc_config_parties() {
        let config = MPCTestConfig::three_party();
        assert_eq!(config.party_ids.len(), 3);
        assert_eq!(config.threshold, 2);
    }
}
