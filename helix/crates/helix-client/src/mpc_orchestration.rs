//! MPC training orchestration for the model owner.
//!
//! This module wraps the MPC integration pipeline with owner-side logic:
//!
//! 1. **Configuration** — model owner specifies architecture, training params,
//!    and worker count
//! 2. **Session Setup** — creates transport mesh, distributes shares
//! 3. **Training Execution** — runs the full MPC training pipeline
//! 4. **Result Collection** — gathers losses, checkpoints, and final weights
//! 5. **On-chain Interaction** — (optional) submits attestations and checkpoints
//!
//! The orchestrator is the owner-facing entry point that delegates the actual
//! MPC protocol execution to `helix_mpc::e2e_integration`.

use std::time::Instant;

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use helix_mpc::e2e_integration::{
    MPCIntegrationConfig, MPCIntegrationResult, InitialWeights,
    run_mpc_training, run_mpc_training_with_cheater,
};

// ============================================================================
// Orchestrator Configuration
// ============================================================================

/// Configuration for the MPC training orchestrator.
///
/// This is the owner-facing configuration that gets translated into the
/// lower-level `MPCIntegrationConfig` for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MPCOrchestrationConfig {
    /// Model architecture: input dimension.
    pub input_dim: usize,
    /// Model architecture: hidden dimension.
    pub hidden_dim: usize,
    /// Model architecture: output dimension.
    pub output_dim: usize,
    /// Number of MPC worker parties.
    pub num_workers: usize,
    /// Total number of training steps.
    pub total_steps: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Pedersen checkpoint interval.
    pub checkpoint_interval: usize,
    /// MAC verification interval (0 to disable).
    pub mac_check_interval: u64,
    /// Beaver triple batch size.
    pub beaver_batch_size: usize,
    /// Initial model weights (optional).
    pub initial_weights: Option<InitialWeights>,
    /// Training data samples.
    pub training_data: Vec<(Vec<f64>, Vec<f64>)>,
    /// Random seed for deterministic execution.
    pub seed: u64,
    /// Model identifier (for on-chain registration).
    pub model_id: Option<String>,
    /// Whether to use NodeTransport (true) or LocalTransport (false).
    pub use_node_transport: bool,
    /// Whether to use TcpTransport for real TCP connections.
    /// Takes precedence over `use_node_transport` when true.
    /// Requires the `network-mpc` feature in helix-mpc.
    pub use_tcp_transport: bool,
}

impl Default for MPCOrchestrationConfig {
    fn default() -> Self {
        Self {
            input_dim: 2,
            hidden_dim: 4,
            output_dim: 1,
            num_workers: 3,
            total_steps: 10,
            learning_rate: 0.01,
            checkpoint_interval: 5,
            mac_check_interval: 5,
            beaver_batch_size: 512,
            initial_weights: None,
            training_data: vec![
                (vec![1.0, 0.5], vec![1.0]),
                (vec![0.5, 1.0], vec![0.0]),
                (vec![0.0, 0.0], vec![0.0]),
                (vec![1.0, 1.0], vec![1.0]),
            ],
            seed: 42,
            model_id: None,
            use_node_transport: false,
            use_tcp_transport: false,
        }
    }
}

impl MPCOrchestrationConfig {
    /// Creates a configuration for a small test model.
    pub fn small_test(num_workers: usize) -> Self {
        Self {
            input_dim: 2,
            hidden_dim: 2,
            output_dim: 1,
            num_workers,
            total_steps: 5,
            learning_rate: 0.01,
            checkpoint_interval: 5,
            mac_check_interval: 5,
            beaver_batch_size: 256,
            initial_weights: Some(InitialWeights {
                w1: vec![0.1, 0.2, 0.3, 0.4],
                b1: vec![0.01, 0.02],
                w2: vec![0.5, 0.6],
                b2: vec![0.03],
            }),
            training_data: vec![
                (vec![1.0, 0.5], vec![1.0]),
                (vec![0.5, 1.0], vec![0.0]),
            ],
            seed: 42,
            model_id: None,
            use_node_transport: false,
            use_tcp_transport: false,
        }
    }

    /// Creates a configuration for MNIST-scale training.
    pub fn mnist_scale(num_workers: usize) -> Self {
        Self {
            input_dim: 784,
            hidden_dim: 32,
            output_dim: 10,
            num_workers,
            total_steps: 100,
            learning_rate: 0.001,
            checkpoint_interval: 10,
            mac_check_interval: 10,
            beaver_batch_size: 2048,
            initial_weights: None,
            training_data: Vec::new(), // Must be filled by caller
            seed: 42,
            model_id: None,
            use_node_transport: false,
            use_tcp_transport: false,
        }
    }
}

// ============================================================================
// Orchestration Result
// ============================================================================

/// Result of an orchestrated MPC training session.
#[derive(Debug)]
pub struct OrchestrationResult {
    /// The underlying MPC integration result.
    pub training_result: MPCIntegrationResult,
    /// Whether training completed successfully (no cheater detected).
    pub success: bool,
    /// Human-readable summary of the training run.
    pub summary: String,
}

// ============================================================================
// MPC Training Orchestrator
// ============================================================================

/// Owner-side orchestrator for MPC training sessions.
///
/// The orchestrator manages the lifecycle of a distributed training session:
///
/// 1. Validates the configuration
/// 2. Translates to the internal MPC integration config
/// 3. Executes the training pipeline
/// 4. Collects and formats results
///
/// For on-chain interaction, the orchestrator can optionally submit
/// checkpoint attestations and final model commitments.
pub struct MPCTrainingOrchestrator {
    config: MPCOrchestrationConfig,
}

impl MPCTrainingOrchestrator {
    /// Creates a new orchestrator with the given configuration.
    pub fn new(config: MPCOrchestrationConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &MPCOrchestrationConfig {
        &self.config
    }

    /// Validates the configuration before running.
    pub fn validate(&self) -> Result<(), String> {
        if self.config.num_workers < 2 {
            return Err("At least 2 workers are required for MPC".to_string());
        }
        if self.config.total_steps == 0 {
            return Err("Must run at least 1 training step".to_string());
        }
        if self.config.training_data.is_empty() {
            return Err("Training data must not be empty".to_string());
        }
        for (i, (input, target)) in self.config.training_data.iter().enumerate() {
            if input.len() != self.config.input_dim {
                return Err(format!(
                    "Training sample {} input dim {} != config input_dim {}",
                    i, input.len(), self.config.input_dim
                ));
            }
            if target.len() != self.config.output_dim {
                return Err(format!(
                    "Training sample {} target dim {} != config output_dim {}",
                    i, target.len(), self.config.output_dim
                ));
            }
        }
        if let Some(ref iw) = self.config.initial_weights {
            let expected_w1 = self.config.hidden_dim * self.config.input_dim;
            if iw.w1.len() != expected_w1 {
                return Err(format!(
                    "Initial w1 length {} != expected {}",
                    iw.w1.len(), expected_w1
                ));
            }
            if iw.b1.len() != self.config.hidden_dim {
                return Err(format!(
                    "Initial b1 length {} != expected {}",
                    iw.b1.len(), self.config.hidden_dim
                ));
            }
            let expected_w2 = self.config.output_dim * self.config.hidden_dim;
            if iw.w2.len() != expected_w2 {
                return Err(format!(
                    "Initial w2 length {} != expected {}",
                    iw.w2.len(), expected_w2
                ));
            }
            if iw.b2.len() != self.config.output_dim {
                return Err(format!(
                    "Initial b2 length {} != expected {}",
                    iw.b2.len(), self.config.output_dim
                ));
            }
        }
        Ok(())
    }

    /// Runs the MPC training pipeline.
    ///
    /// This is the main entry point for the model owner. It:
    /// 1. Validates the configuration
    /// 2. Translates to the internal MPC config
    /// 3. Executes the training pipeline
    /// 4. Returns the orchestration result
    pub async fn run(&self) -> Result<OrchestrationResult, anyhow::Error> {
        self.validate().map_err(|e| anyhow::anyhow!("Configuration validation failed: {}", e))?;

        let start = Instant::now();
        info!(
            model_id = ?self.config.model_id,
            workers = self.config.num_workers,
            steps = self.config.total_steps,
            "Starting MPC training orchestration"
        );

        let integration_config = self.build_integration_config();
        let training_result = run_mpc_training(integration_config).await?;

        let success = training_result.cheater_detected.is_none();
        let summary = self.build_summary(&training_result, start.elapsed().as_secs_f64());

        info!(
            success = success,
            steps = training_result.steps_completed,
            final_loss = training_result.final_loss,
            "MPC training orchestration complete"
        );

        Ok(OrchestrationResult {
            training_result,
            success,
            summary,
        })
    }

    /// Runs MPC training with a cheater injected for testing.
    pub async fn run_with_cheater(
        &self,
        cheater_party: usize,
        corrupt_at_step: u64,
    ) -> Result<OrchestrationResult, anyhow::Error> {
        self.validate().map_err(|e| anyhow::anyhow!("Validation failed: {}", e))?;

        let start = Instant::now();
        info!(
            cheater = cheater_party,
            corrupt_step = corrupt_at_step,
            "Starting MPC training with cheater injection"
        );

        let integration_config = self.build_integration_config();
        let training_result = run_mpc_training_with_cheater(
            integration_config, cheater_party, corrupt_at_step,
        ).await?;

        let success = training_result.cheater_detected.is_none();
        let summary = self.build_summary(&training_result, start.elapsed().as_secs_f64());

        Ok(OrchestrationResult {
            training_result,
            success,
            summary,
        })
    }

    /// Translates the orchestration config to an MPC integration config.
    fn build_integration_config(&self) -> MPCIntegrationConfig {
        MPCIntegrationConfig {
            d_in: self.config.input_dim,
            d_hid: self.config.hidden_dim,
            d_out: self.config.output_dim,
            num_workers: self.config.num_workers,
            num_steps: self.config.total_steps,
            learning_rate: self.config.learning_rate,
            checkpoint_interval: self.config.checkpoint_interval,
            mac_check_interval: self.config.mac_check_interval,
            beaver_batch_size: self.config.beaver_batch_size,
            initial_weights: self.config.initial_weights.clone(),
            training_data: self.config.training_data.clone(),
            seed: self.config.seed,
            use_node_transport: self.config.use_node_transport,
            use_tcp_transport: self.config.use_tcp_transport,
        }
    }

    /// Builds a human-readable summary of the training run.
    fn build_summary(&self, result: &MPCIntegrationResult, elapsed_secs: f64) -> String {
        let mut summary = String::new();
        summary.push_str(&format!(
            "MPC Training Summary\n\
             ====================\n\
             Model: {}x{}x{} ({}+{}+{}+{} = {} params)\n\
             Workers: {}\n\
             Steps: {}/{}\n\
             Final Loss: {:.6}\n\
             Checkpoints: {}\n\
             MAC Checks Passed: {}\n\
             Time: {:.2}s\n",
            self.config.input_dim, self.config.hidden_dim, self.config.output_dim,
            self.config.hidden_dim * self.config.input_dim,
            self.config.hidden_dim,
            self.config.output_dim * self.config.hidden_dim,
            self.config.output_dim,
            self.config.hidden_dim * self.config.input_dim
                + self.config.hidden_dim
                + self.config.output_dim * self.config.hidden_dim
                + self.config.output_dim,
            self.config.num_workers,
            result.steps_completed,
            self.config.total_steps,
            result.final_loss,
            result.checkpoints.len(),
            result.mac_checks_passed,
            elapsed_secs,
        ));

        if let Some(ref cheater) = result.cheater_detected {
            summary.push_str(&format!(
                "\nCHEATER DETECTED: party {} at step {}\n",
                cheater.party_index, cheater.detected_at_step
            ));
        } else {
            summary.push_str("\nNo cheating detected - all integrity checks passed.\n");
        }

        summary
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation_success() {
        let config = MPCOrchestrationConfig::small_test(3);
        let orchestrator = MPCTrainingOrchestrator::new(config);
        assert!(orchestrator.validate().is_ok());
    }

    #[test]
    fn test_config_validation_insufficient_workers() {
        let mut config = MPCOrchestrationConfig::small_test(3);
        config.num_workers = 1;
        let orchestrator = MPCTrainingOrchestrator::new(config);
        assert!(orchestrator.validate().is_err());
    }

    #[test]
    fn test_config_validation_no_steps() {
        let mut config = MPCOrchestrationConfig::small_test(3);
        config.total_steps = 0;
        let orchestrator = MPCTrainingOrchestrator::new(config);
        assert!(orchestrator.validate().is_err());
    }

    #[test]
    fn test_config_validation_empty_data() {
        let mut config = MPCOrchestrationConfig::small_test(3);
        config.training_data = Vec::new();
        let orchestrator = MPCTrainingOrchestrator::new(config);
        assert!(orchestrator.validate().is_err());
    }

    #[test]
    fn test_config_validation_dimension_mismatch() {
        let mut config = MPCOrchestrationConfig::small_test(3);
        config.training_data = vec![(vec![1.0, 2.0, 3.0], vec![1.0])]; // 3 != input_dim=2
        let orchestrator = MPCTrainingOrchestrator::new(config);
        assert!(orchestrator.validate().is_err());
    }

    #[test]
    fn test_build_integration_config() {
        let config = MPCOrchestrationConfig::small_test(3);
        let orchestrator = MPCTrainingOrchestrator::new(config.clone());
        let ic = orchestrator.build_integration_config();

        assert_eq!(ic.d_in, config.input_dim);
        assert_eq!(ic.d_hid, config.hidden_dim);
        assert_eq!(ic.d_out, config.output_dim);
        assert_eq!(ic.num_workers, config.num_workers);
        assert_eq!(ic.num_steps, config.total_steps);
    }

    #[tokio::test]
    async fn test_orchestrator_run() {
        let config = MPCOrchestrationConfig::small_test(3);
        let orchestrator = MPCTrainingOrchestrator::new(config);
        let result = orchestrator.run().await.expect("orchestration should succeed");
        assert!(result.success);
        assert!(result.training_result.steps_completed > 0);
        assert!(!result.summary.is_empty());
    }

    #[test]
    fn test_mnist_config() {
        let mut config = MPCOrchestrationConfig::mnist_scale(3);
        // Add dummy training data.
        config.training_data = vec![(vec![0.0; 784], vec![0.0; 10])];
        let orchestrator = MPCTrainingOrchestrator::new(config);
        assert!(orchestrator.validate().is_ok());
    }
}
