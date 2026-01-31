//! Training Module for HELIX Decentralized Training.
//!
//! This module provides a complete federated learning training pipeline with:
//!
//! - **Round Management**: State machine for training rounds with timeout handling
//! - **Data Loading**: Streaming data loader with tokenization and sharding
//! - **Model Management**: Weight loading, saving, and gradient application
//! - **Checkpointing**: Automatic checkpoints with verification
//! - **Aggregation**: Byzantine-fault-tolerant gradient aggregation
//! - **Metrics**: Training metrics and convergence detection
//! - **Coordination**: Distributed training orchestration with leader election
//! - **Verification**: Proof verification before gradient aggregation
//! - **Orchestration**: Multi-node coordination with failure detection

pub mod round;
pub mod data_loader;
pub mod model;
pub mod checkpoint;
pub mod aggregation;
pub mod metrics;
pub mod coordinator;
pub mod session;
pub mod mpc;
pub mod orchestrator;
pub mod verification;

// Re-export key types
pub use round::{
    RoundId, RoundState, RoundConfig, RoundFailure,
    TrainingRound, RoundManager, RoundSummary,
    Participant, GradientSubmission, AggregationResult, RoundError,
};

pub use data_loader::{
    TokenId, TrainingBatch, DataLoaderConfig, DataLoader,
    SimpleTokenizer, SpecialTokens, DataShard, ShardSource,
};

pub use model::{
    WeightId, ModelVersion, ModelMetadata, LayerWeights, WeightData,
    ModelWeights, LayerGradient, ModelGradient, ModelError,
};

pub use checkpoint::{
    CheckpointId, CheckpointMetadata, OptimizerState, Checkpoint,
    CheckpointConfig, CheckpointManager, CheckpointError,
};

pub use aggregation::{
    AggregationStrategy, AggregationConfig, WeightedGradient,
    AggregatedGradient, GradientAggregator, AggregationError,
};

pub use metrics::{
    MetricsTracker, MetricsConfig, IterationMetrics, TrainingSummary,
    ConvergenceDetector, LearningRateSchedule, ScheduleType,
};

pub use coordinator::{
    NodeId, NodeRole, CoordinatorConfig, CoordinatorState,
    CoordinatorEvent, TrainingCoordinator, PeerInfo, CoordinatorError,
    ByzantineDetection, ErrorBoundTracker, GradientOutlierDetector, RecoveryState,
};

pub use session::{
    MlpDataset, SessionConfig, SessionResult, ProvedTrainingSession,
};

pub use mpc::{
    MPCTrainingConfig, MPCTrainingRound, MPCStepResult,
    WorkerComputation, AdversarialDetector, SlashingEvent, SlashingReason,
    model_to_flat, flat_to_model,
};

pub use orchestrator::{
    CollectedGradient, OrchestratorConfig, OrchestratorError, OrchestratorEvent,
    RoundPhase, RoundState as OrchestratorRoundState, TrainingOrchestrator,
    WorkerState, WorkerStats, WorkerStatus,
};

pub use verification::{
    GradientValidator, ProofType, ProofVerifier, ValidationResult,
    VerificationConfig, VerificationResult, VerificationStats,
};

/// Convenience type alias for training results.
pub type TrainingResult<T> = Result<T, TrainingError>;

/// Top-level training error type.
#[derive(Debug)]
pub enum TrainingError {
    /// Round-related error.
    Round(RoundError),
    /// Model-related error.
    Model(ModelError),
    /// Checkpoint-related error.
    Checkpoint(CheckpointError),
    /// Aggregation-related error.
    Aggregation(AggregationError),
    /// Coordinator-related error.
    Coordinator(CoordinatorError),
    /// MPC-related error.
    MPC(helix_mpc::MPCError),
    /// IO error.
    Io(std::io::Error),
    /// Custom error message.
    Custom(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Round(e) => write!(f, "Round error: {}", e),
            Self::Model(e) => write!(f, "Model error: {}", e),
            Self::Checkpoint(e) => write!(f, "Checkpoint error: {}", e),
            Self::Aggregation(e) => write!(f, "Aggregation error: {}", e),
            Self::Coordinator(e) => write!(f, "Coordinator error: {}", e),
            Self::MPC(e) => write!(f, "MPC error: {}", e),
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for TrainingError {}

impl From<RoundError> for TrainingError {
    fn from(e: RoundError) -> Self {
        Self::Round(e)
    }
}

impl From<ModelError> for TrainingError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}

impl From<CheckpointError> for TrainingError {
    fn from(e: CheckpointError) -> Self {
        Self::Checkpoint(e)
    }
}

impl From<AggregationError> for TrainingError {
    fn from(e: AggregationError) -> Self {
        Self::Aggregation(e)
    }
}

impl From<CoordinatorError> for TrainingError {
    fn from(e: CoordinatorError) -> Self {
        Self::Coordinator(e)
    }
}

impl From<helix_mpc::MPCError> for TrainingError {
    fn from(e: helix_mpc::MPCError) -> Self {
        Self::MPC(e)
    }
}

impl From<std::io::Error> for TrainingError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_module_integration() {
        // Create a model
        let model_meta = ModelMetadata {
            name: "integration-test".to_string(),
            hidden_dim: 16,
            num_layers: 2,
            num_heads: 2,
            vocab_size: 100,
            ..Default::default()
        };
        let model = ModelWeights::random(model_meta, 42);

        // Create a tokenizer and data
        let tokenizer = SimpleTokenizer::basic(256);
        let data = vec![
            "Hello world, this is a test.".to_string(),
            "Training data for HELIX.".to_string(),
        ];

        let loader_config = DataLoaderConfig {
            batch_size: 2,
            max_seq_len: 32,
            drop_last: false,
            ..Default::default()
        };

        let mut loader = DataLoader::from_memory(data, loader_config, tokenizer);

        // Get a batch
        let batch = loader.next_batch();
        assert!(batch.is_some());

        // Verify model parameters
        assert!(model.count_parameters() > 0);
    }
}
