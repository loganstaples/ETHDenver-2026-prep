//! Configuration structures for HELIX components.

use crate::error::{HelixResult, ValidationError};
use crate::types::Precision;
use serde::{Deserialize, Serialize};

/// Configuration for the Approximate Virtual Machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VMConfig {
    /// Default precision for operations.
    pub default_precision: Precision,
    /// Maximum error accumulation before warning.
    pub max_error_accumulation: f64,
    /// Maximum safe error threshold for detecting numerical instability.
    /// If any single value's error exceeds this, the computation is flagged.
    pub max_safe_error: f64,
    /// Whether to collect execution trace for proofs.
    pub collect_trace: bool,
    /// Memory limit in bytes.
    pub memory_limit: usize,
    /// Maximum number of operations per execution.
    pub max_operations: usize,
}

impl VMConfig {
    /// Creates a `VMConfig` with precision-appropriate error budgets.
    ///
    /// - `F32`: max_error_accumulation = 0.01
    /// - `F16` / `BF16`: max_error_accumulation = 0.05
    /// - `INT8` / `INT4`: max_error_accumulation = 0.10
    pub fn for_precision(precision: Precision) -> Self {
        use crate::constants::error_bounds;

        let max_error_accumulation = match precision {
            Precision::F32 => error_bounds::MAX_ERROR_ACCUMULATION,
            Precision::F16 | Precision::BF16 => error_bounds::BF16_MAX_ERROR_ACCUMULATION,
            Precision::INT8 | Precision::INT4 => error_bounds::INT8_MAX_ERROR_ACCUMULATION,
            Precision::Custom { max_relative_error, .. } => {
                // For custom precision, scale budget based on relative error
                if max_relative_error <= 1.19e-7 { error_bounds::MAX_ERROR_ACCUMULATION }
                else if max_relative_error <= 9.77e-4 { error_bounds::BF16_MAX_ERROR_ACCUMULATION }
                else { error_bounds::INT8_MAX_ERROR_ACCUMULATION }
            }
        };

        Self {
            default_precision: precision,
            max_error_accumulation,
            max_safe_error: 1e6,
            collect_trace: true,
            memory_limit: 1024 * 1024 * 1024,
            max_operations: 10_000_000,
        }
    }
}

impl Default for VMConfig {
    fn default() -> Self {
        Self {
            default_precision: Precision::F32,
            max_error_accumulation: 0.01,
            max_safe_error: 1e6,
            collect_trace: true,
            memory_limit: 1024 * 1024 * 1024, // 1GB
            max_operations: 10_000_000,
        }
    }
}

impl VMConfig {
    /// Validates this configuration.
    pub fn validate(&self) -> HelixResult<()> {
        if self.max_error_accumulation <= 0.0 {
            return Err(ValidationError::InvalidValue {
                field: "max_error_accumulation".to_string(),
                value: self.max_error_accumulation.to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }
        if self.memory_limit == 0 {
            return Err(ValidationError::InvalidValue {
                field: "memory_limit".to_string(),
                value: "0".to_string(),
                reason: "must be greater than 0".to_string(),
            }
            .into());
        }
        if self.max_operations == 0 {
            return Err(ValidationError::InvalidValue {
                field: "max_operations".to_string(),
                value: "0".to_string(),
                reason: "must be greater than 0".to_string(),
            }
            .into());
        }
        Ok(())
    }
}

/// Configuration for the ZK prover.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverConfig {
    /// Number of threads for proof generation.
    pub num_threads: usize,
    /// Path to proving key cache.
    pub key_cache_path: Option<String>,
    /// Whether to use recursive proofs.
    pub recursive_proofs: bool,
    /// Maximum chunk size for large computations.
    pub max_chunk_size: usize,
}

impl Default for ProverConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus(),
            key_cache_path: None,
            recursive_proofs: false,
            max_chunk_size: 1024,
        }
    }
}

impl ProverConfig {
    /// Validates this configuration.
    pub fn validate(&self) -> HelixResult<()> {
        if self.num_threads == 0 {
            return Err(ValidationError::InvalidValue {
                field: "num_threads".to_string(),
                value: "0".to_string(),
                reason: "must be greater than 0".to_string(),
            }
            .into());
        }
        if self.max_chunk_size == 0 {
            return Err(ValidationError::InvalidValue {
                field: "max_chunk_size".to_string(),
                value: "0".to_string(),
                reason: "must be greater than 0".to_string(),
            }
            .into());
        }
        Ok(())
    }
}

/// Configuration for network operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Listen address.
    pub listen_addr: String,
    /// Bootstrap peers.
    pub bootstrap_peers: Vec<String>,
    /// Connection timeout in seconds.
    pub connection_timeout: u64,
    /// Maximum number of peers.
    pub max_peers: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: "/ip4/0.0.0.0/tcp/9000".to_string(),
            bootstrap_peers: vec![],
            connection_timeout: 30,
            max_peers: 50,
        }
    }
}

/// Configuration for training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size for training.
    pub batch_size: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Number of rounds.
    pub num_rounds: usize,
    /// Gradient precision.
    pub gradient_precision: Precision,
    /// Maximum gradient error allowed.
    pub max_gradient_error: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 0.001,
            num_rounds: 1000,
            gradient_precision: Precision::F32,
            max_gradient_error: 0.001,
        }
    }
}

impl TrainingConfig {
    /// Validates this configuration.
    pub fn validate(&self) -> HelixResult<()> {
        if self.batch_size == 0 {
            return Err(ValidationError::InvalidValue {
                field: "batch_size".to_string(),
                value: "0".to_string(),
                reason: "must be greater than 0".to_string(),
            }
            .into());
        }
        if self.learning_rate <= 0.0 {
            return Err(ValidationError::InvalidValue {
                field: "learning_rate".to_string(),
                value: self.learning_rate.to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }
        if self.max_gradient_error <= 0.0 {
            return Err(ValidationError::InvalidValue {
                field: "max_gradient_error".to_string(),
                value: self.max_gradient_error.to_string(),
                reason: "must be positive".to_string(),
            }
            .into());
        }
        Ok(())
    }
}

/// Complete configuration for a HELIX node.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HelixConfig {
    /// VM configuration.
    pub vm: VMConfig,
    /// Prover configuration.
    pub prover: ProverConfig,
    /// Network configuration.
    pub network: NetworkConfig,
    /// Training configuration.
    pub training: TrainingConfig,
}

impl HelixConfig {
    /// Validates all sub-configurations.
    pub fn validate(&self) -> HelixResult<()> {
        self.vm.validate()?;
        self.prover.validate()?;
        self.training.validate()?;
        Ok(())
    }
}

/// Helper to get number of CPUs.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HelixConfig::default();
        assert_eq!(config.vm.default_precision, Precision::F32);
        assert!(config.prover.num_threads > 0);
    }

    #[test]
    fn test_recursive_proofs_default_false() {
        let config = ProverConfig::default();
        assert!(!config.recursive_proofs);
    }

    #[test]
    fn test_valid_config_validates() {
        let config = HelixConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_vm_config_invalid_error_accumulation() {
        let mut config = VMConfig::default();
        config.max_error_accumulation = 0.0;
        assert!(config.validate().is_err());
        config.max_error_accumulation = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_vm_config_invalid_memory_limit() {
        let mut config = VMConfig::default();
        config.memory_limit = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_vm_config_invalid_max_operations() {
        let mut config = VMConfig::default();
        config.max_operations = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_prover_config_invalid_threads() {
        let mut config = ProverConfig::default();
        config.num_threads = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_prover_config_invalid_chunk_size() {
        let mut config = ProverConfig::default();
        config.max_chunk_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_training_config_invalid_batch_size() {
        let mut config = TrainingConfig::default();
        config.batch_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_training_config_invalid_learning_rate() {
        let mut config = TrainingConfig::default();
        config.learning_rate = 0.0;
        assert!(config.validate().is_err());
        config.learning_rate = -0.01;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_training_config_invalid_gradient_error() {
        let mut config = TrainingConfig::default();
        config.max_gradient_error = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_helix_config_propagates_validation() {
        let mut config = HelixConfig::default();
        config.vm.memory_limit = 0;
        assert!(config.validate().is_err());

        let mut config = HelixConfig::default();
        config.prover.num_threads = 0;
        assert!(config.validate().is_err());

        let mut config = HelixConfig::default();
        config.training.batch_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_vm_config_for_precision() {
        let f32_config = VMConfig::for_precision(Precision::F32);
        assert_eq!(f32_config.max_error_accumulation, 0.01);
        assert_eq!(f32_config.max_safe_error, 1e6);

        let bf16_config = VMConfig::for_precision(Precision::BF16);
        assert_eq!(bf16_config.max_error_accumulation, 0.05);

        let int8_config = VMConfig::for_precision(Precision::INT8);
        assert_eq!(int8_config.max_error_accumulation, 0.10);

        let f16_config = VMConfig::for_precision(Precision::F16);
        assert_eq!(f16_config.max_error_accumulation, 0.05);

        let int4_config = VMConfig::for_precision(Precision::INT4);
        assert_eq!(int4_config.max_error_accumulation, 0.10);

        // Budget increases with decreasing precision
        assert!(f32_config.max_error_accumulation < bf16_config.max_error_accumulation);
        assert!(bf16_config.max_error_accumulation < int8_config.max_error_accumulation);
    }
}
