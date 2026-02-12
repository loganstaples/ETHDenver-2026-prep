//! SDK model and training types for the HELIX Client.

use serde::{Deserialize, Serialize};

use crate::error::HelixError;

/// Describes a model's layer dimensions: (input, hidden, output).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArchitecture {
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
}

impl ModelArchitecture {
    pub fn new(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self { d_in, d_hid, d_out }
    }

    /// Total trainable parameters (weights + biases for a 2-layer MLP).
    pub fn parameter_count(&self) -> usize {
        // Layer 1: d_in * d_hid weights + d_hid biases
        // Layer 2: d_hid * d_out weights + d_out biases
        self.d_in * self.d_hid + self.d_hid + self.d_hid * self.d_out + self.d_out
    }
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}x{}", self.d_in, self.d_hid, self.d_out)
    }
}

/// Configuration for registering a new model via the SDK.
///
/// Use the builder pattern to construct:
/// ```ignore
/// let config = SdkModelConfig::new("my-model", ModelArchitecture::new(2, 4, 1))
///     .with_min_stake(0.5)
///     .with_initial_weights(weights_bytes);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdkModelConfig {
    /// Human-readable model name.
    pub name: String,
    /// Model architecture dimensions.
    pub architecture: ModelArchitecture,
    /// Optional serialized initial weights.
    pub initial_weights: Option<Vec<u8>>,
    /// Minimum stake required to participate (ETH). Defaults to 0.1.
    pub min_stake: f64,
}

impl SdkModelConfig {
    /// Create a new model config with required fields.
    pub fn new(name: impl Into<String>, architecture: ModelArchitecture) -> Self {
        Self {
            name: name.into(),
            architecture,
            min_stake: 0.1,
            initial_weights: None,
        }
    }

    /// Set the minimum stake.
    pub fn with_min_stake(mut self, min_stake: f64) -> Self {
        self.min_stake = min_stake;
        self
    }

    /// Attach initial weights.
    pub fn with_initial_weights(mut self, weights: Vec<u8>) -> Self {
        self.initial_weights = Some(weights);
        self
    }

    /// Validate this configuration, returning an error if invalid.
    pub fn validate(&self) -> Result<(), HelixError> {
        if self.name.trim().is_empty() {
            return Err(HelixError::model("model name cannot be empty"));
        }
        if self.architecture.d_in == 0 || self.architecture.d_hid == 0 || self.architecture.d_out == 0 {
            return Err(HelixError::model("architecture dimensions must be > 0"));
        }
        if self.min_stake < 0.0 {
            return Err(HelixError::model("min_stake cannot be negative"));
        }
        Ok(())
    }
}

/// Handle returned after successfully registering a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHandle {
    /// On-chain model ID.
    pub model_id: u64,
    /// Human-readable name.
    pub name: String,
    /// Current commitment hash (hex string).
    pub commitment: String,
    /// Architecture dimensions.
    pub architecture: ModelArchitecture,
}

/// Parameters for a training session.
///
/// ```ignore
/// let params = TrainingParams::default()
///     .with_rounds(20)
///     .with_learning_rate(0.001);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Number of training rounds.
    pub rounds: u64,
    /// Learning rate.
    pub learning_rate: f64,
    /// Batch size per round.
    pub batch_size: u32,
    /// Duration of each round in seconds.
    pub round_duration_secs: u64,
    /// Polling interval in milliseconds for status updates.
    pub poll_interval_ms: u64,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            rounds: 10,
            learning_rate: 0.001,
            batch_size: 32,
            round_duration_secs: 60,
            poll_interval_ms: 500,
        }
    }
}

impl TrainingParams {
    pub fn with_rounds(mut self, rounds: u64) -> Self {
        self.rounds = rounds;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_batch_size(mut self, batch_size: u32) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_round_duration(mut self, secs: u64) -> Self {
        self.round_duration_secs = secs;
        self
    }

    pub fn with_poll_interval(mut self, ms: u64) -> Self {
        self.poll_interval_ms = ms;
        self
    }
}
