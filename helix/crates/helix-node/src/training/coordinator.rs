//! Training Coordinator.
//!
//! Orchestrates the distributed training process including leader election,
//! task distribution, and consensus on aggregated gradients.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

use super::round::{RoundId, RoundManager, RoundConfig, RoundState, TrainingRound, GradientSubmission, AggregationResult};
use super::model::{ModelWeights, ModelGradient, ModelMetadata};
use super::aggregation::{GradientAggregator, AggregationConfig, WeightedGradient, AggregationStrategy};
use super::checkpoint::{CheckpointManager, CheckpointConfig};
use super::metrics::{MetricsTracker, MetricsConfig, IterationMetrics, LearningRateSchedule, ScheduleType};
use super::data_loader::{DataLoader, DataLoaderConfig, SimpleTokenizer};

/// Unique identifier for a training node.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    /// Creates a new node ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Role of a node in the current round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Node is the leader/aggregator for this round.
    Leader,
    /// Node is a participant/worker.
    Participant,
    /// Node is observing (not actively participating).
    Observer,
}

/// Configuration for training coordinator.
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// This node's ID.
    pub node_id: NodeId,
    /// Round configuration.
    pub round_config: RoundConfig,
    /// Aggregation configuration.
    pub aggregation_config: AggregationConfig,
    /// Checkpoint configuration.
    pub checkpoint_config: CheckpointConfig,
    /// Metrics configuration.
    pub metrics_config: MetricsConfig,
    /// Initial learning rate.
    pub learning_rate: f64,
    /// Learning rate schedule type.
    pub lr_schedule: ScheduleType,
    /// Number of warmup steps.
    pub warmup_steps: u64,
    /// Total expected training steps.
    pub total_steps: u64,
    /// Maximum rounds before stopping.
    pub max_rounds: u64,
    /// How often to rotate leadership.
    pub leader_rotation_interval: u64,
    /// Node stake (for weighted aggregation).
    pub stake: u64,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            node_id: NodeId::new("node-0"),
            round_config: RoundConfig::default(),
            aggregation_config: AggregationConfig::default(),
            checkpoint_config: CheckpointConfig::default(),
            metrics_config: MetricsConfig::default(),
            learning_rate: 0.001,
            lr_schedule: ScheduleType::CosineAnnealing,
            warmup_steps: 100,
            total_steps: 10000,
            max_rounds: 1000,
            leader_rotation_interval: 10,
            stake: 1000,
        }
    }
}

/// State of the training coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinatorState {
    /// Not yet initialized.
    Uninitialized,
    /// Ready to participate in training.
    Ready,
    /// Waiting for round to start.
    WaitingForRound,
    /// Computing local gradient.
    Computing,
    /// Waiting for aggregation.
    WaitingForAggregation,
    /// Leader is aggregating gradients.
    Aggregating,
    /// Committing update to chain.
    Committing,
    /// Training complete.
    Complete,
    /// Error state.
    Error,
}

/// Event from the training coordinator.
#[derive(Debug, Clone)]
pub enum CoordinatorEvent {
    /// New round started.
    RoundStarted { round_id: RoundId },
    /// Local gradient computed.
    GradientComputed { round_id: RoundId, gradient_hash: [u8; 32] },
    /// Received gradient from peer.
    GradientReceived { round_id: RoundId, from: NodeId },
    /// Aggregation complete.
    AggregationComplete { round_id: RoundId, num_contributors: usize },
    /// Model updated.
    ModelUpdated { round_id: RoundId, new_commitment: [u8; 32] },
    /// Round failed.
    RoundFailed { round_id: RoundId, reason: String },
    /// Training converged.
    Converged { final_loss: f64 },
    /// Checkpoint created.
    CheckpointCreated { checkpoint_id: u64 },
}

/// Training coordinator for a single node.
#[derive(Debug)]
pub struct TrainingCoordinator {
    /// Configuration.
    config: CoordinatorConfig,
    /// Current state.
    state: CoordinatorState,
    /// Current role.
    role: NodeRole,
    /// Round manager.
    rounds: RoundManager,
    /// Model weights.
    model: Option<ModelWeights>,
    /// Gradient aggregator.
    aggregator: GradientAggregator,
    /// Checkpoint manager.
    checkpoints: Option<CheckpointManager>,
    /// Metrics tracker.
    metrics: MetricsTracker,
    /// Learning rate schedule.
    lr_schedule: LearningRateSchedule,
    /// Known peers.
    peers: HashMap<NodeId, PeerInfo>,
    /// Pending events.
    pending_events: Vec<CoordinatorEvent>,
    /// Local gradient for current round.
    local_gradient: Option<ModelGradient>,
    /// Current leader.
    current_leader: Option<NodeId>,
}

/// Information about a peer node.
#[derive(Debug, Clone)]
pub struct PeerInfo {
    /// Peer's node ID.
    pub id: NodeId,
    /// Peer's stake.
    pub stake: u64,
    /// Whether peer is active.
    pub is_active: bool,
    /// Last seen timestamp.
    pub last_seen: Instant,
    /// Address for communication.
    pub address: String,
}

impl TrainingCoordinator {
    /// Creates a new training coordinator.
    pub fn new(config: CoordinatorConfig) -> Self {
        let rounds = RoundManager::new(config.round_config.clone());
        let aggregator = GradientAggregator::new(config.aggregation_config.clone());
        let metrics = MetricsTracker::new(config.metrics_config.clone());
        let lr_schedule = LearningRateSchedule::new(
            config.learning_rate,
            config.lr_schedule,
            config.warmup_steps,
            config.total_steps,
        );

        Self {
            config,
            state: CoordinatorState::Uninitialized,
            role: NodeRole::Observer,
            rounds,
            model: None,
            aggregator,
            checkpoints: None,
            metrics,
            lr_schedule,
            peers: HashMap::new(),
            pending_events: Vec::new(),
            local_gradient: None,
            current_leader: None,
        }
    }

    /// Initializes the coordinator with a model.
    pub fn initialize(&mut self, model: ModelWeights) -> Result<(), CoordinatorError> {
        // Initialize checkpoint manager
        self.checkpoints = Some(
            CheckpointManager::new(self.config.checkpoint_config.clone())
                .map_err(|e| CoordinatorError::InitializationFailed(e.to_string()))?
        );

        self.model = Some(model);
        self.state = CoordinatorState::Ready;
        self.metrics.start();

        Ok(())
    }

    /// Registers a peer.
    pub fn register_peer(&mut self, id: NodeId, stake: u64, address: String) {
        self.peers.insert(id.clone(), PeerInfo {
            id,
            stake,
            is_active: true,
            last_seen: Instant::now(),
            address,
        });
    }

    /// Unregisters a peer.
    pub fn unregister_peer(&mut self, id: &NodeId) {
        self.peers.remove(id);
    }

    /// Returns the current state.
    pub fn state(&self) -> CoordinatorState {
        self.state
    }

    /// Returns the current role.
    pub fn role(&self) -> NodeRole {
        self.role
    }

    /// Returns the current round ID.
    pub fn current_round_id(&self) -> Option<RoundId> {
        self.rounds.current().map(|r| r.id)
    }

    /// Returns whether this node is the leader.
    pub fn is_leader(&self) -> bool {
        self.role == NodeRole::Leader
    }

    /// Starts a new training round.
    pub fn start_round(&mut self) -> Result<RoundId, CoordinatorError> {
        let model = self.model.as_ref()
            .ok_or(CoordinatorError::ModelNotInitialized)?;

        let round_id = self.rounds.start_round(model.metadata.commitment)
            .map_err(|e| CoordinatorError::RoundError(e.to_string()))?;

        // Determine leader using deterministic rotation
        self.current_leader = Some(self.elect_leader(round_id));
        self.role = if self.current_leader.as_ref() == Some(&self.config.node_id) {
            NodeRole::Leader
        } else {
            NodeRole::Participant
        };

        // Register this node as participant
        if let Some(round) = self.rounds.current_mut() {
            round.register_participant(
                self.config.node_id.0.clone(),
                self.config.stake,
            ).map_err(|e| CoordinatorError::RoundError(e.to_string()))?;

            round.start_collection()
                .map_err(|e| CoordinatorError::RoundError(e.to_string()))?;
        }

        self.state = CoordinatorState::WaitingForRound;
        self.pending_events.push(CoordinatorEvent::RoundStarted { round_id });

        Ok(round_id)
    }

    /// Elects a leader for a round using deterministic rotation.
    fn elect_leader(&self, round_id: RoundId) -> NodeId {
        let mut all_nodes: Vec<&NodeId> = self.peers.keys().collect();
        all_nodes.push(&self.config.node_id);
        all_nodes.sort_by(|a, b| a.0.cmp(&b.0));

        if all_nodes.is_empty() {
            return self.config.node_id.clone();
        }

        // Simple rotation based on round number
        let leader_idx = (round_id.0 as usize / self.config.leader_rotation_interval as usize) % all_nodes.len();
        all_nodes[leader_idx].clone()
    }

    /// Submits a computed gradient for the current round.
    pub fn submit_gradient(
        &mut self,
        gradient: ModelGradient,
        error_bound: f64,
        proof: Vec<u8>,
    ) -> Result<(), CoordinatorError> {
        let round = self.rounds.current_mut()
            .ok_or(CoordinatorError::NoActiveRound)?;

        if round.state != RoundState::Collecting {
            return Err(CoordinatorError::WrongState);
        }

        let gradient_hash = gradient.commitment();

        // Submit to round
        let submission = GradientSubmission {
            participant_id: self.config.node_id.0.clone(),
            gradient_hash,
            gradient_data: vec![], // Serialized separately
            proof,
            error_bound,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        round.submit_gradient(submission)
            .map_err(|e| CoordinatorError::RoundError(e.to_string()))?;

        // Store local gradient for aggregation
        self.local_gradient = Some(gradient);

        self.pending_events.push(CoordinatorEvent::GradientComputed {
            round_id: round.id,
            gradient_hash,
        });

        self.state = CoordinatorState::WaitingForAggregation;

        Ok(())
    }

    /// Receives a gradient from a peer.
    pub fn receive_gradient(
        &mut self,
        from: NodeId,
        gradient: ModelGradient,
        stake: u64,
        error_bound: f64,
        gradient_hash: [u8; 32],
        proof: Vec<u8>,
    ) -> Result<(), CoordinatorError> {
        let round = self.rounds.current_mut()
            .ok_or(CoordinatorError::NoActiveRound)?;

        if round.state != RoundState::Collecting {
            return Err(CoordinatorError::WrongState);
        }

        let round_id = round.id;

        // Submit to round tracking
        let submission = GradientSubmission {
            participant_id: from.0.clone(),
            gradient_hash,
            gradient_data: vec![],
            proof,
            error_bound,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        round.submit_gradient(submission)
            .map_err(|e| CoordinatorError::RoundError(e.to_string()))?;

        // Add to aggregator (leader only)
        if self.is_leader() {
            self.aggregator.add_gradient(WeightedGradient {
                participant_id: from.0.clone(),
                stake,
                gradient,
                error_bound,
                is_valid: true, // TODO: verify proof
            });
        }

        self.pending_events.push(CoordinatorEvent::GradientReceived {
            round_id,
            from,
        });

        Ok(())
    }

    /// Performs aggregation (leader only).
    pub fn aggregate(&mut self) -> Result<AggregationResult, CoordinatorError> {
        if !self.is_leader() {
            return Err(CoordinatorError::NotLeader);
        }

        let round = self.rounds.current_mut()
            .ok_or(CoordinatorError::NoActiveRound)?;

        let round_id = round.id;

        // Add local gradient to aggregator
        if let Some(local_grad) = self.local_gradient.take() {
            self.aggregator.add_gradient(WeightedGradient {
                participant_id: self.config.node_id.0.clone(),
                stake: self.config.stake,
                gradient: local_grad,
                error_bound: 0.01, // TODO: track properly
                is_valid: true,
            });
        }

        // Start aggregation phase
        round.start_aggregation()
            .map_err(|e| CoordinatorError::RoundError(e.to_string()))?;

        self.state = CoordinatorState::Aggregating;

        // Perform aggregation
        let aggregated = self.aggregator.aggregate()
            .map_err(|e| CoordinatorError::AggregationFailed(e.to_string()))?;

        let result = AggregationResult {
            aggregated_hash: aggregated.gradient.commitment(),
            aggregated_gradient: vec![], // Serialized separately
            combined_error_bound: aggregated.error_bound,
            total_stake: aggregated.total_stake,
            num_contributors: aggregated.num_included,
            excluded_participants: aggregated.excluded.clone(),
        };

        // Set result on round
        round.set_aggregation_result(result.clone())
            .map_err(|e| CoordinatorError::RoundError(e.to_string()))?;

        self.pending_events.push(CoordinatorEvent::AggregationComplete {
            round_id,
            num_contributors: aggregated.num_included,
        });

        // Apply gradient to model
        if let Some(model) = self.model.as_mut() {
            let lr = self.lr_schedule.step();
            model.apply_gradient(&aggregated.gradient, lr)
                .map_err(|e| CoordinatorError::ModelUpdateFailed(e.to_string()))?;

            self.pending_events.push(CoordinatorEvent::ModelUpdated {
                round_id,
                new_commitment: model.metadata.commitment,
            });
        }

        // Clear aggregator for next round
        self.aggregator.clear();

        self.state = CoordinatorState::Committing;

        Ok(result)
    }

    /// Completes the current round.
    pub fn complete_round(&mut self) -> Result<(), CoordinatorError> {
        let model = self.model.as_ref()
            .ok_or(CoordinatorError::ModelNotInitialized)?;

        let round = self.rounds.current_mut()
            .ok_or(CoordinatorError::NoActiveRound)?;

        round.complete(model.metadata.commitment)
            .map_err(|e| CoordinatorError::RoundError(e.to_string()))?;

        // Check if we should checkpoint
        if let Some(ref mut ckpt_mgr) = self.checkpoints {
            if ckpt_mgr.should_checkpoint(model.metadata.training_iterations) {
                if let Ok(ckpt_id) = ckpt_mgr.save(model, None, Some(round.id)) {
                    self.pending_events.push(CoordinatorEvent::CheckpointCreated {
                        checkpoint_id: ckpt_id.0,
                    });
                }
            }
        }

        // Check for convergence
        if self.metrics.has_converged() {
            self.state = CoordinatorState::Complete;
            self.pending_events.push(CoordinatorEvent::Converged {
                final_loss: self.metrics.current_loss().unwrap_or(0.0),
            });
        } else {
            self.state = CoordinatorState::Ready;
        }

        Ok(())
    }

    /// Records training metrics for the current iteration.
    pub fn record_metrics(&mut self, loss: f64, gradient_norm: f64, batch_size: usize, duration: Duration) {
        if let Some(model) = &self.model {
            let metrics = IterationMetrics::new(model.metadata.training_iterations)
                .with_loss(loss)
                .with_gradient_norm(gradient_norm)
                .with_error_bound(0.0) // TODO
                .with_learning_rate(self.lr_schedule.current())
                .with_batch_size(batch_size)
                .with_processing_time(duration);
            self.metrics.record(metrics);
        }
    }

    /// Returns pending events and clears them.
    pub fn drain_events(&mut self) -> Vec<CoordinatorEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Returns a reference to the model.
    pub fn model(&self) -> Option<&ModelWeights> {
        self.model.as_ref()
    }

    /// Returns a mutable reference to the model.
    pub fn model_mut(&mut self) -> Option<&mut ModelWeights> {
        self.model.as_mut()
    }

    /// Returns the metrics tracker.
    pub fn metrics(&self) -> &MetricsTracker {
        &self.metrics
    }

    /// Returns the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.lr_schedule.current()
    }

    /// Returns the number of active peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Checks timeouts on the current round.
    pub fn check_timeouts(&mut self) -> Option<CoordinatorEvent> {
        if let Some(round) = self.rounds.current_mut() {
            if let Some(failure) = round.check_timeouts() {
                let round_id = round.id;
                self.state = CoordinatorState::Error;
                return Some(CoordinatorEvent::RoundFailed {
                    round_id,
                    reason: format!("{:?}", failure),
                });
            }
        }
        None
    }
}

/// Coordinator errors.
#[derive(Debug)]
pub enum CoordinatorError {
    /// Model not initialized.
    ModelNotInitialized,
    /// Initialization failed.
    InitializationFailed(String),
    /// Round error.
    RoundError(String),
    /// No active round.
    NoActiveRound,
    /// Wrong state for operation.
    WrongState,
    /// Not the leader.
    NotLeader,
    /// Aggregation failed.
    AggregationFailed(String),
    /// Model update failed.
    ModelUpdateFailed(String),
}

impl std::fmt::Display for CoordinatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelNotInitialized => write!(f, "Model not initialized"),
            Self::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            Self::RoundError(msg) => write!(f, "Round error: {}", msg),
            Self::NoActiveRound => write!(f, "No active round"),
            Self::WrongState => write!(f, "Wrong state for operation"),
            Self::NotLeader => write!(f, "Not the leader for this round"),
            Self::AggregationFailed(msg) => write!(f, "Aggregation failed: {}", msg),
            Self::ModelUpdateFailed(msg) => write!(f, "Model update failed: {}", msg),
        }
    }
}

impl std::error::Error for CoordinatorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let config = CoordinatorConfig::default();
        let coordinator = TrainingCoordinator::new(config);
        assert_eq!(coordinator.state(), CoordinatorState::Uninitialized);
    }

    #[test]
    fn test_coordinator_initialization() {
        let mut config = CoordinatorConfig::default();
        config.checkpoint_config.checkpoint_dir = std::path::PathBuf::from("/tmp/helix-test-ckpt");

        let mut coordinator = TrainingCoordinator::new(config);

        let model_meta = ModelMetadata {
            name: "test".to_string(),
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 1,
            vocab_size: 10,
            ..Default::default()
        };
        let model = ModelWeights::random(model_meta, 42);

        coordinator.initialize(model).unwrap();
        assert_eq!(coordinator.state(), CoordinatorState::Ready);
    }
}
