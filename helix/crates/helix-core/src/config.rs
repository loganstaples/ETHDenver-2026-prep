//! Configuration structures for HELIX components.

use crate::types::Precision;
use serde::{Deserialize, Serialize};

/// Configuration for the Approximate Virtual Machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VMConfig {
    /// Default precision for operations.
    pub default_precision: Precision,
    /// Maximum error accumulation before warning.
    pub max_error_accumulation: f64,
    /// Whether to collect execution trace for proofs.
    pub collect_trace: bool,
    /// Memory limit in bytes.
    pub memory_limit: usize,
    /// Maximum number of operations per execution.
    pub max_operations: usize,
}

impl Default for VMConfig {
    fn default() -> Self {
        Self {
            default_precision: Precision::F32,
            max_error_accumulation: 0.01,
            collect_trace: true,
            memory_limit: 1024 * 1024 * 1024, // 1GB
            max_operations: 10_000_000,
        }
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
            recursive_proofs: true,
            max_chunk_size: 1024,
        }
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
}
