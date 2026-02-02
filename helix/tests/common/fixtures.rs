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
#[derive(Debug, Clone, Copy)]
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
            use_freivalds: self.use_freivalds,
            ..V2ProverConfig::default()
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

// ============================================================================
// Extended Fixtures for Reusable Test Models
// ============================================================================

/// Pre-configured test model for XOR problem.
pub struct XORTestModel {
    pub dims: ModelDimensions,
    pub weights: TestModelWeights,
    pub dataset: TestDataset,
}

impl XORTestModel {
    /// Creates an XOR test model with known-good configuration.
    pub fn new() -> Self {
        let dims = ModelDimensions::tiny();
        let weights = TestModelWeights::known(dims);
        let dataset = TestDataset::new(dims.d_in, dims.d_out, 4, 42);
        Self { dims, weights, dataset }
    }

    /// Returns training samples as tuples.
    pub fn samples(&self) -> Vec<(Vec<Fr>, Vec<Fr>)> {
        self.dataset.to_tuples()
    }
}

impl Default for XORTestModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-configured regression test model.
pub struct RegressionTestModel {
    pub dims: ModelDimensions,
    pub weights: TestModelWeights,
    pub dataset: TestDataset,
}

impl RegressionTestModel {
    /// Creates a regression test model with small architecture.
    pub fn new(num_samples: usize) -> Self {
        let dims = ModelDimensions::small();
        let weights = TestModelWeights::random(dims, 42);
        let dataset = TestDataset::new(dims.d_in, dims.d_out, num_samples, 42);
        Self { dims, weights, dataset }
    }

    /// Creates with specific seed for reproducibility.
    pub fn with_seed(num_samples: usize, seed: u64) -> Self {
        let dims = ModelDimensions::small();
        let weights = TestModelWeights::random(dims, seed);
        let dataset = TestDataset::new(dims.d_in, dims.d_out, num_samples, seed);
        Self { dims, weights, dataset }
    }
}

/// Pre-configured stress test model.
pub struct StressTestModel {
    pub dims: ModelDimensions,
    pub weights: TestModelWeights,
    pub dataset: TestDataset,
}

impl StressTestModel {
    /// Creates a stress test model with medium architecture.
    pub fn new(num_samples: usize) -> Self {
        let dims = ModelDimensions::medium();
        let weights = TestModelWeights::random(dims, 42);
        let dataset = TestDataset::new(dims.d_in, dims.d_out, num_samples, 42);
        Self { dims, weights, dataset }
    }

    /// Creates a large model for memory stress testing.
    pub fn large(num_samples: usize) -> Self {
        let dims = ModelDimensions::large();
        let weights = TestModelWeights::random(dims, 42);
        let dataset = TestDataset::new(dims.d_in, dims.d_out, num_samples, 42);
        Self { dims, weights, dataset }
    }
}

/// Builder for constructing custom test scenarios.
pub struct TestScenarioBuilder {
    dims: Option<ModelDimensions>,
    weights_seed: u64,
    dataset_seed: u64,
    num_samples: usize,
    mpc_config: Option<MPCTestConfig>,
    prover_config: Option<ProverTestConfig>,
}

impl TestScenarioBuilder {
    pub fn new() -> Self {
        Self {
            dims: None,
            weights_seed: 42,
            dataset_seed: 42,
            num_samples: 10,
            mpc_config: None,
            prover_config: None,
        }
    }

    pub fn with_dims(mut self, dims: ModelDimensions) -> Self {
        self.dims = Some(dims);
        self
    }

    pub fn tiny(self) -> Self {
        self.with_dims(ModelDimensions::tiny())
    }

    pub fn small(self) -> Self {
        self.with_dims(ModelDimensions::small())
    }

    pub fn medium(self) -> Self {
        self.with_dims(ModelDimensions::medium())
    }

    pub fn large(self) -> Self {
        self.with_dims(ModelDimensions::large())
    }

    pub fn with_weights_seed(mut self, seed: u64) -> Self {
        self.weights_seed = seed;
        self
    }

    pub fn with_dataset_seed(mut self, seed: u64) -> Self {
        self.dataset_seed = seed;
        self
    }

    pub fn with_samples(mut self, count: usize) -> Self {
        self.num_samples = count;
        self
    }

    pub fn with_mpc_config(mut self, config: MPCTestConfig) -> Self {
        self.mpc_config = Some(config);
        self
    }

    pub fn with_prover_config(mut self, config: ProverTestConfig) -> Self {
        self.prover_config = Some(config);
        self
    }

    pub fn build(self) -> TestScenario {
        let dims = self.dims.unwrap_or(ModelDimensions::tiny());
        TestScenario {
            name: format!("custom_{}x{}x{}", dims.d_in, dims.d_hid, dims.d_out),
            model: TestModelWeights::random(dims, self.weights_seed),
            dataset: TestDataset::new(dims.d_in, dims.d_out, self.num_samples, self.dataset_seed),
            mpc_config: self.mpc_config.unwrap_or(MPCTestConfig::three_party()),
            prover_config: self.prover_config.unwrap_or(ProverTestConfig::standard()),
            expected_result: KnownGoodResult::xor_tiny(),
            dims,
        }
    }
}

impl Default for TestScenarioBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Collection of pre-built test fixtures for common scenarios.
pub struct TestFixtures;

impl TestFixtures {
    /// Returns a tiny model suitable for quick unit tests.
    pub fn tiny_model() -> (ModelDimensions, TestModelWeights, TestSample) {
        let dims = ModelDimensions::tiny();
        let weights = TestModelWeights::known(dims);
        let sample = TestSample::known(dims.d_in, dims.d_out);
        (dims, weights, sample)
    }

    /// Returns a small model suitable for integration tests.
    pub fn small_model() -> (ModelDimensions, TestModelWeights, TestDataset) {
        let dims = ModelDimensions::small();
        let weights = TestModelWeights::random(dims, 42);
        let dataset = TestDataset::new(dims.d_in, dims.d_out, 10, 42);
        (dims, weights, dataset)
    }

    /// Returns a model and samples for chain testing.
    pub fn chain_test_setup(num_steps: usize) -> (ModelDimensions, TestModelWeights, Vec<(Vec<Fr>, Vec<Fr>)>) {
        let dims = ModelDimensions::tiny();
        let weights = TestModelWeights::known(dims);
        let dataset = TestDataset::new(dims.d_in, dims.d_out, num_steps, 42);
        (dims, weights, dataset.to_tuples())
    }

    /// Returns fixtures for adversarial testing.
    pub fn adversarial_setup() -> (ModelDimensions, TestModelWeights, MPCTestConfig) {
        let dims = ModelDimensions::tiny();
        let weights = TestModelWeights::known(dims);
        let mpc_config = MPCTestConfig::three_party();
        (dims, weights, mpc_config)
    }

    /// Returns fixtures for performance testing.
    pub fn performance_setup() -> (ModelDimensions, TestModelWeights, ProverTestConfig) {
        let dims = ModelDimensions::small();
        let weights = TestModelWeights::random(dims, 42);
        let prover_config = ProverTestConfig::standard();
        (dims, weights, prover_config)
    }
}

/// Validation helpers for test assertions.
pub struct TestValidation;

impl TestValidation {
    /// Validates that a proof result has correct structure.
    pub fn validate_proof_result(
        proof: &helix_prover::TrainingProofResultV2,
        expected_step: u64,
    ) -> Result<(), String> {
        use helix_circuits::ml::training_step_v2::NUM_PUBLIC_INPUTS;

        if proof.proof.is_empty() {
            return Err("Proof bytes are empty".to_string());
        }

        if proof.proof.len() < 64 {
            return Err(format!(
                "Proof too short: {} bytes, need at least 64",
                proof.proof.len()
            ));
        }

        if proof.public_inputs.len() != NUM_PUBLIC_INPUTS {
            return Err(format!(
                "Wrong public input count: {} != {}",
                proof.public_inputs.len(),
                NUM_PUBLIC_INPUTS
            ));
        }

        if proof.step_number != expected_step {
            return Err(format!(
                "Step number mismatch: {} != {}",
                proof.step_number, expected_step
            ));
        }

        Ok(())
    }

    /// Validates chain continuity between two proofs.
    pub fn validate_chain_link(
        current: &helix_prover::TrainingProofResultV2,
        next: &helix_prover::TrainingProofResultV2,
    ) -> Result<(), String> {
        if current.new_state_hash != next.old_state_hash {
            return Err(format!(
                "Chain broken: current.new_hash {:?} != next.old_hash {:?}",
                current.new_state_hash, next.old_state_hash
            ));
        }

        if next.step_number != current.step_number + 1 {
            return Err(format!(
                "Step number not sequential: {} + 1 != {}",
                current.step_number, next.step_number
            ));
        }

        Ok(())
    }

    /// Validates a complete proof chain.
    pub fn validate_proof_chain(
        proofs: &[helix_prover::TrainingProofResultV2],
    ) -> Result<(), String> {
        if proofs.is_empty() {
            return Err("Empty proof chain".to_string());
        }

        for (i, proof) in proofs.iter().enumerate() {
            Self::validate_proof_result(proof, (i + 1) as u64)?;
        }

        for i in 0..proofs.len() - 1 {
            Self::validate_chain_link(&proofs[i], &proofs[i + 1])?;
        }

        Ok(())
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

    #[test]
    fn test_xor_test_model() {
        let model = XORTestModel::new();
        assert_eq!(model.dims.d_in, 2);
        assert_eq!(model.dims.d_hid, 2);
        assert_eq!(model.dims.d_out, 1);
    }

    #[test]
    fn test_scenario_builder() {
        let scenario = TestScenarioBuilder::new()
            .tiny()
            .with_samples(5)
            .with_weights_seed(123)
            .build();

        assert_eq!(scenario.dims.d_in, 2);
        assert_eq!(scenario.dataset.samples.len(), 5);
    }

    #[test]
    fn test_fixtures_tiny() {
        let (dims, weights, sample) = TestFixtures::tiny_model();
        assert_eq!(dims.d_in, 2);
        assert_eq!(weights.w1.len(), dims.d_hid * dims.d_in);
        assert_eq!(sample.x.len(), dims.d_in);
    }

    #[test]
    fn test_fixtures_chain_setup() {
        let (dims, weights, samples) = TestFixtures::chain_test_setup(3);
        assert_eq!(samples.len(), 3);
        assert_eq!(weights.dims.d_in, dims.d_in);
    }
}
