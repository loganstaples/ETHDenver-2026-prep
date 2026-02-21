//! Training Configuration Module
//!
//! Defines comprehensive configuration structures for HELIX training jobs.
//! Supports TOML file format for training parameter specification.

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

// ============================================================================
// Training Job Configuration
// ============================================================================

/// Complete training job configuration loaded from TOML file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJobConfig {
    /// Model configuration
    pub model: ModelConfig,
    /// Training hyperparameters
    pub training: TrainingHyperParams,
    /// Data configuration
    pub data: DataConfig,
    /// Network/distributed settings
    #[serde(default)]
    pub network: NetworkSettings,
    /// Proof generation settings
    #[serde(default)]
    pub proof: ProofSettings,
    /// Checkpoint settings
    #[serde(default)]
    pub checkpoint: CheckpointSettings,
    /// Resource limits
    #[serde(default)]
    pub resources: ResourceLimits,
    /// Logging and output settings
    #[serde(default)]
    pub output: OutputSettings,
}

impl TrainingJobConfig {
    /// Load configuration from a TOML file
    pub fn load(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read training config: {}", path.display()))?;
        let config: Self = toml::from_str(&content)
            .with_context(|| format!("Failed to parse training config: {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Model validation
        if self.model.name.is_empty() {
            return Err(anyhow!("Model name cannot be empty"));
        }
        if self.model.hidden_dim == 0 {
            return Err(anyhow!("Hidden dimension must be > 0"));
        }
        if self.model.num_layers == 0 {
            return Err(anyhow!("Number of layers must be > 0"));
        }

        // Training validation
        if self.training.batch_size == 0 {
            return Err(anyhow!("Batch size must be > 0"));
        }
        if self.training.learning_rate <= 0.0 {
            return Err(anyhow!("Learning rate must be > 0"));
        }
        if self.training.max_rounds == 0 {
            return Err(anyhow!("Max rounds must be > 0"));
        }

        // Data validation
        if self.data.source.is_empty() {
            return Err(anyhow!("Data source cannot be empty"));
        }

        // Network validation
        if self.network.min_workers == 0 {
            return Err(anyhow!("Minimum workers must be > 0"));
        }

        // Proof validation
        if self.proof.max_error_bound <= 0.0 {
            return Err(anyhow!("Max error bound must be > 0"));
        }

        Ok(())
    }

    /// Create a default configuration for a simple demo
    pub fn demo_config() -> Self {
        Self {
            model: ModelConfig {
                name: "helix-demo-model".to_string(),
                architecture: ModelArchitecture::Transformer,
                input_dim: 16,
                hidden_dim: 32,
                output_dim: 4,
                num_layers: 2,
                num_heads: Some(4),
                vocab_size: None,
                activation: ActivationType::ReLU,
                precision: PrecisionType::FP16,
                quantization: None,
                weights_path: None,
            },
            training: TrainingHyperParams {
                batch_size: 32,
                learning_rate: 0.001,
                max_rounds: 10,
                warmup_rounds: 2,
                optimizer: OptimizerType::Adam,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.0,
                gradient_clip: Some(1.0),
                lr_scheduler: None,
                early_stopping: None,
            },
            data: DataConfig {
                source: "synthetic".to_string(),
                dataset_type: DatasetType::Synthetic,
                train_split: 0.8,
                validation_split: 0.1,
                test_split: 0.1,
                shuffle: true,
                seed: Some(42),
                preprocessing: None,
            },
            network: NetworkSettings::default(),
            proof: ProofSettings::default(),
            checkpoint: CheckpointSettings::default(),
            resources: ResourceLimits::default(),
            output: OutputSettings::default(),
        }
    }

    /// Create configuration for quick demo (30 seconds)
    pub fn quick_demo_config() -> Self {
        let mut config = Self::demo_config();
        config.model.hidden_dim = 16;
        config.model.num_layers = 1;
        config.training.max_rounds = 3;
        config.proof.circuit_k = 11;
        config
    }

    /// Create configuration for full demo (90 seconds)
    pub fn full_demo_config() -> Self {
        let mut config = Self::demo_config();
        config.training.max_rounds = 8;
        config.proof.circuit_k = 12;
        config
    }

    /// Generate example configuration file content
    pub fn example_toml() -> String {
        r#"# HELIX Training Configuration
# Complete configuration for distributed ML training with ZK verification

[model]
name = "my-transformer"
architecture = "transformer"
input_dim = 32
hidden_dim = 64
output_dim = 10
num_layers = 4
num_heads = 8
# vocab_size = 32000  # For language models
activation = "relu"
precision = "fp16"
# quantization = { bits = 8, scheme = "symmetric" }
# weights_path = "./pretrained_weights.bin"

[training]
batch_size = 32
learning_rate = 0.001
max_rounds = 100
warmup_rounds = 10
optimizer = "adam"
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 0.01
gradient_clip = 1.0
# lr_scheduler = { type = "cosine", min_lr = 1e-6 }
# early_stopping = { patience = 10, min_delta = 0.0001 }

[data]
source = "ipfs://QmXoYPm8rPnxV3YHNqpGd8tVwL5c7sW9eZyMbTxNxNwXYZ"
dataset_type = "custom"
train_split = 0.8
validation_split = 0.1
test_split = 0.1
shuffle = true
seed = 42
# preprocessing = { normalize = true, augment = false }

[network]
min_workers = 3
max_workers = 10
round_timeout_secs = 120
coordinator = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
stake_amount = 0.5
auto_stake = true
byzantine_threshold = 0.33

[proof]
proof_system = "approximate"
max_error_bound = 1000.0
circuit_k = 12
use_freivalds = true
batch_proofs = true
proof_compression = false
verification_timeout_secs = 60
checkpoint_proving_interval = 1  # Generate ZK proof every N steps (1 = every step)

[checkpoint]
enabled = true
interval_rounds = 10
max_checkpoints = 5
save_path = "./checkpoints"
resume_from = ""

[resources]
max_memory_mb = 8192
max_proof_time_ms = 500
gpu_enabled = false
num_threads = 0  # 0 = auto-detect

[output]
log_level = "info"
metrics_interval_secs = 10
progress_bar = true
save_metrics = true
metrics_path = "./metrics"
verbose = false
"#.to_string()
    }
}

impl Default for TrainingJobConfig {
    fn default() -> Self {
        Self::demo_config()
    }
}

// ============================================================================
// Model Configuration
// ============================================================================

/// Model architecture and structure configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name/identifier
    pub name: String,
    /// Model architecture type
    #[serde(default)]
    pub architecture: ModelArchitecture,
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads (for transformers)
    #[serde(default)]
    pub num_heads: Option<usize>,
    /// Vocabulary size (for language models)
    #[serde(default)]
    pub vocab_size: Option<usize>,
    /// Activation function type
    #[serde(default)]
    pub activation: ActivationType,
    /// Precision type
    #[serde(default)]
    pub precision: PrecisionType,
    /// Quantization settings (if enabled)
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    /// Path to pretrained weights (optional)
    #[serde(default)]
    pub weights_path: Option<PathBuf>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "helix-model".to_string(),
            architecture: ModelArchitecture::MLP,
            input_dim: 16,
            hidden_dim: 32,
            output_dim: 4,
            num_layers: 2,
            num_heads: None,
            vocab_size: None,
            activation: ActivationType::ReLU,
            precision: PrecisionType::FP16,
            quantization: None,
            weights_path: None,
        }
    }
}

/// Model architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelArchitecture {
    /// Multi-layer perceptron
    MLP,
    /// Transformer (decoder-only, like GPT)
    Transformer,
    /// BERT-style encoder
    BERT,
    /// Convolutional neural network
    CNN,
    /// Recurrent neural network
    RNN,
    /// Custom architecture
    Custom,
}

impl Default for ModelArchitecture {
    fn default() -> Self {
        Self::MLP
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActivationType {
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    SiLU,
    Softmax,
}

impl Default for ActivationType {
    fn default() -> Self {
        Self::ReLU
    }
}

/// Precision types for computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrecisionType {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
}

impl Default for PrecisionType {
    fn default() -> Self {
        Self::FP16
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Number of bits for quantization
    pub bits: u32,
    /// Quantization scheme
    #[serde(default)]
    pub scheme: QuantizationScheme,
    /// Per-channel or per-tensor
    #[serde(default)]
    pub granularity: QuantizationGranularity,
}

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationScheme {
    Symmetric,
    Asymmetric,
    Dynamic,
}

impl Default for QuantizationScheme {
    fn default() -> Self {
        Self::Symmetric
    }
}

/// Quantization granularity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationGranularity {
    PerTensor,
    PerChannel,
    PerGroup,
}

impl Default for QuantizationGranularity {
    fn default() -> Self {
        Self::PerTensor
    }
}

// ============================================================================
// Training Hyperparameters
// ============================================================================

/// Training hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHyperParams {
    /// Batch size
    pub batch_size: u32,
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum number of training rounds
    pub max_rounds: u32,
    /// Number of warmup rounds
    #[serde(default)]
    pub warmup_rounds: u32,
    /// Optimizer type
    #[serde(default)]
    pub optimizer: OptimizerType,
    /// Adam beta1
    #[serde(default = "default_beta1")]
    pub beta1: f64,
    /// Adam beta2
    #[serde(default = "default_beta2")]
    pub beta2: f64,
    /// Adam epsilon
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    /// Weight decay
    #[serde(default)]
    pub weight_decay: f64,
    /// Gradient clipping value (optional)
    #[serde(default)]
    pub gradient_clip: Option<f64>,
    /// Learning rate scheduler (optional)
    #[serde(default)]
    pub lr_scheduler: Option<LRSchedulerConfig>,
    /// Early stopping configuration (optional)
    #[serde(default)]
    pub early_stopping: Option<EarlyStoppingConfig>,
}

fn default_beta1() -> f64 { 0.9 }
fn default_beta2() -> f64 { 0.999 }
fn default_epsilon() -> f64 { 1e-8 }

impl Default for TrainingHyperParams {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 0.001,
            max_rounds: 100,
            warmup_rounds: 10,
            optimizer: OptimizerType::Adam,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            gradient_clip: Some(1.0),
            lr_scheduler: None,
            early_stopping: None,
        }
    }
}

/// Optimizer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
}

impl Default for OptimizerType {
    fn default() -> Self {
        Self::Adam
    }
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRSchedulerConfig {
    /// Scheduler type
    #[serde(rename = "type")]
    pub scheduler_type: LRSchedulerType,
    /// Minimum learning rate
    #[serde(default = "default_min_lr")]
    pub min_lr: f64,
    /// Step size (for step scheduler)
    #[serde(default)]
    pub step_size: Option<u32>,
    /// Decay factor (for step scheduler)
    #[serde(default)]
    pub gamma: Option<f64>,
}

fn default_min_lr() -> f64 { 1e-6 }

/// Learning rate scheduler types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LRSchedulerType {
    Constant,
    Step,
    Cosine,
    Linear,
    Exponential,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Patience (rounds to wait)
    pub patience: u32,
    /// Minimum improvement delta
    #[serde(default)]
    pub min_delta: f64,
    /// Metric to monitor
    #[serde(default = "default_monitor_metric")]
    pub monitor: String,
}

fn default_monitor_metric() -> String { "loss".to_string() }

// ============================================================================
// Data Configuration
// ============================================================================

/// Data source and preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Data source (path, URL, or IPFS hash)
    pub source: String,
    /// Dataset type
    #[serde(default)]
    pub dataset_type: DatasetType,
    /// Training split ratio
    #[serde(default = "default_train_split")]
    pub train_split: f64,
    /// Validation split ratio
    #[serde(default = "default_val_split")]
    pub validation_split: f64,
    /// Test split ratio
    #[serde(default = "default_test_split")]
    pub test_split: f64,
    /// Shuffle data
    #[serde(default = "default_true")]
    pub shuffle: bool,
    /// Random seed
    #[serde(default)]
    pub seed: Option<u64>,
    /// Preprocessing configuration
    #[serde(default)]
    pub preprocessing: Option<PreprocessingConfig>,
}

fn default_train_split() -> f64 { 0.8 }
fn default_val_split() -> f64 { 0.1 }
fn default_test_split() -> f64 { 0.1 }
fn default_true() -> bool { true }

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            source: "synthetic".to_string(),
            dataset_type: DatasetType::Synthetic,
            train_split: 0.8,
            validation_split: 0.1,
            test_split: 0.1,
            shuffle: true,
            seed: Some(42),
            preprocessing: None,
        }
    }
}

/// Dataset types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetType {
    /// Synthetic data (for testing/demo)
    Synthetic,
    /// Image classification
    Image,
    /// Text/language
    Text,
    /// Tabular data
    Tabular,
    /// Time series
    TimeSeries,
    /// Custom format
    Custom,
}

impl Default for DatasetType {
    fn default() -> Self {
        Self::Synthetic
    }
}

/// Data preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Normalize data
    #[serde(default)]
    pub normalize: bool,
    /// Apply data augmentation
    #[serde(default)]
    pub augment: bool,
    /// Tokenizer (for text)
    #[serde(default)]
    pub tokenizer: Option<String>,
    /// Max sequence length (for text)
    #[serde(default)]
    pub max_seq_len: Option<usize>,
}

// ============================================================================
// Network Settings
// ============================================================================

/// Distributed network settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSettings {
    /// Minimum number of workers required
    #[serde(default = "default_min_workers")]
    pub min_workers: u32,
    /// Maximum number of workers
    #[serde(default = "default_max_workers")]
    pub max_workers: u32,
    /// Round timeout in seconds
    #[serde(default = "default_round_timeout")]
    pub round_timeout_secs: u64,
    /// Coordinator contract address
    #[serde(default)]
    pub coordinator: String,
    /// Stake amount in ETH
    #[serde(default = "default_stake")]
    pub stake_amount: f64,
    /// Automatically stake when starting
    #[serde(default)]
    pub auto_stake: bool,
    /// Byzantine fault tolerance threshold
    #[serde(default = "default_byzantine_threshold")]
    pub byzantine_threshold: f64,
}

fn default_min_workers() -> u32 { 3 }
fn default_max_workers() -> u32 { 10 }
fn default_round_timeout() -> u64 { 120 }
fn default_stake() -> f64 { 0.001 }
fn default_byzantine_threshold() -> f64 { 0.33 }

impl Default for NetworkSettings {
    fn default() -> Self {
        Self {
            min_workers: 3,
            max_workers: 10,
            round_timeout_secs: 120,
            coordinator: String::new(),
            stake_amount: 0.001,
            auto_stake: true,
            byzantine_threshold: 0.33,
        }
    }
}

// ============================================================================
// Proof Settings
// ============================================================================

/// ZK proof generation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofSettings {
    /// Proof system type
    #[serde(default)]
    pub proof_system: ProofSystemType,
    /// Maximum allowed error bound
    #[serde(default = "default_error_bound")]
    pub max_error_bound: f64,
    /// Circuit k parameter (2^k rows)
    #[serde(default = "default_circuit_k")]
    pub circuit_k: u32,
    /// Use Freivalds algorithm for matrix verification
    #[serde(default = "default_true")]
    pub use_freivalds: bool,
    /// Batch multiple proofs together
    #[serde(default)]
    pub batch_proofs: bool,
    /// Compress proofs for storage/transmission
    #[serde(default)]
    pub proof_compression: bool,
    /// Verification timeout in seconds
    #[serde(default = "default_verify_timeout")]
    pub verification_timeout_secs: u64,
    /// Checkpoint proving interval: generate ZK proof every N steps (1 = every step).
    /// Higher values reduce proving overhead by using MPC consensus between checkpoints.
    #[serde(default = "default_checkpoint_proving_interval")]
    pub checkpoint_proving_interval: u32,
}

fn default_error_bound() -> f64 { 1000.0 }
fn default_circuit_k() -> u32 { 12 }
fn default_verify_timeout() -> u64 { 60 }
fn default_checkpoint_proving_interval() -> u32 { 1 }

impl Default for ProofSettings {
    fn default() -> Self {
        Self {
            proof_system: ProofSystemType::Approximate,
            max_error_bound: 1000.0,
            circuit_k: 12,
            use_freivalds: true,
            batch_proofs: false,
            proof_compression: false,
            verification_timeout_secs: 60,
            checkpoint_proving_interval: 1,
        }
    }
}

/// Proof system types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProofSystemType {
    /// Approximate ZK proofs (faster, bounded error)
    Approximate,
    /// Full ZK proofs (exact but slower)
    Full,
    /// Optimistic proofs with fraud detection
    Optimistic,
}

impl Default for ProofSystemType {
    fn default() -> Self {
        Self::Approximate
    }
}

// ============================================================================
// Checkpoint Settings
// ============================================================================

/// Checkpoint and recovery settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSettings {
    /// Enable checkpointing
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Checkpoint interval (rounds)
    #[serde(default = "default_checkpoint_interval")]
    pub interval_rounds: u32,
    /// Maximum number of checkpoints to keep
    #[serde(default = "default_max_checkpoints")]
    pub max_checkpoints: u32,
    /// Directory to save checkpoints
    #[serde(default = "default_checkpoint_path")]
    pub save_path: PathBuf,
    /// Resume from checkpoint (path or round number)
    #[serde(default)]
    pub resume_from: String,
}

fn default_checkpoint_interval() -> u32 { 10 }
fn default_max_checkpoints() -> u32 { 5 }
fn default_checkpoint_path() -> PathBuf { PathBuf::from("./checkpoints") }

impl Default for CheckpointSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_rounds: 10,
            max_checkpoints: 5,
            save_path: PathBuf::from("./checkpoints"),
            resume_from: String::new(),
        }
    }
}

// ============================================================================
// Resource Limits
// ============================================================================

/// Resource allocation and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage (MB)
    #[serde(default = "default_max_memory")]
    pub max_memory_mb: u64,
    /// Maximum proof generation time (ms)
    #[serde(default = "default_max_proof_time")]
    pub max_proof_time_ms: u64,
    /// Enable GPU acceleration
    #[serde(default)]
    pub gpu_enabled: bool,
    /// Number of CPU threads (0 = auto-detect)
    #[serde(default)]
    pub num_threads: u32,
}

fn default_max_memory() -> u64 { 8192 }
fn default_max_proof_time() -> u64 { 500 }

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 8192,
            max_proof_time_ms: 500,
            gpu_enabled: false,
            num_threads: 0,
        }
    }
}

// ============================================================================
// Output Settings
// ============================================================================

/// Logging and output settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSettings {
    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Metrics collection interval (seconds)
    #[serde(default = "default_metrics_interval")]
    pub metrics_interval_secs: u64,
    /// Show progress bar
    #[serde(default = "default_true")]
    pub progress_bar: bool,
    /// Save metrics to file
    #[serde(default = "default_true")]
    pub save_metrics: bool,
    /// Metrics output directory
    #[serde(default = "default_metrics_path")]
    pub metrics_path: PathBuf,
    /// Verbose output
    #[serde(default)]
    pub verbose: bool,
}

fn default_log_level() -> String { "info".to_string() }
fn default_metrics_interval() -> u64 { 10 }
fn default_metrics_path() -> PathBuf { PathBuf::from("./metrics") }

impl Default for OutputSettings {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            metrics_interval_secs: 10,
            progress_bar: true,
            save_metrics: true,
            metrics_path: PathBuf::from("./metrics"),
            verbose: false,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config_validates() {
        let config = TrainingJobConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_demo_configs_validate() {
        assert!(TrainingJobConfig::demo_config().validate().is_ok());
        assert!(TrainingJobConfig::quick_demo_config().validate().is_ok());
        assert!(TrainingJobConfig::full_demo_config().validate().is_ok());
    }

    #[test]
    fn test_example_toml_parses() {
        let toml_content = TrainingJobConfig::example_toml();
        let config: Result<TrainingJobConfig, _> = toml::from_str(&toml_content);
        assert!(config.is_ok());
    }

    #[test]
    fn test_save_and_load() {
        let config = TrainingJobConfig::demo_config();
        let mut file = NamedTempFile::new().unwrap();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        file.write_all(toml_str.as_bytes()).unwrap();

        let loaded = TrainingJobConfig::load(&file.path().to_path_buf());
        assert!(loaded.is_ok());

        let loaded = loaded.unwrap();
        assert_eq!(loaded.model.name, config.model.name);
    }

    #[test]
    fn test_validation_failures() {
        let mut config = TrainingJobConfig::default();

        // Empty model name
        config.model.name = String::new();
        assert!(config.validate().is_err());
        config.model.name = "test".to_string();

        // Zero batch size
        config.training.batch_size = 0;
        assert!(config.validate().is_err());
        config.training.batch_size = 32;

        // Negative learning rate
        config.training.learning_rate = -0.001;
        assert!(config.validate().is_err());
    }
}
