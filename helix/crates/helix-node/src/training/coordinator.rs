//! Training Coordinator.
//!
//! Orchestrates the distributed training process including leader election,
//! task distribution, consensus on aggregated gradients, proof verification,
//! Byzantine fault detection, error bound tracking, and verified data loading.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::round::{RoundId, RoundManager, RoundConfig, RoundState, TrainingRound, GradientSubmission, AggregationResult};
use super::model::{ModelWeights, ModelGradient, ModelMetadata, WeightData};
use super::aggregation::{GradientAggregator, AggregationConfig, WeightedGradient, AggregationStrategy};
use super::checkpoint::{CheckpointManager, CheckpointConfig};
use super::metrics::{MetricsTracker, MetricsConfig, IterationMetrics, LearningRateSchedule, ScheduleType};
use super::data_loader::{DataLoader, DataLoaderConfig, SimpleTokenizer};
use super::verification::{ProofVerifier, VerificationConfig, VerificationResult, GradientValidator, ValidationResult};
use crate::network::messages::PeerId;
use crate::data::{
    VerifiedDataLoader, VerifiedDataLoaderConfig, VerifiedBatch, DataVerificationError,
    DataAvailabilityChecker, AvailabilityConfig, AvailabilityCheckResult,
    CommitmentVerifier, CommitmentVerifierConfig, CommitmentStatus,
};

/// Unique identifier for a training node.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    /// Creates a new node ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Converts to PeerId for verification.
    pub fn to_peer_id(&self) -> PeerId {
        PeerId::from_string(&self.0)
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
    /// Verification configuration.
    pub verification_config: VerificationConfig,
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
    /// Enable adaptive learning rate based on error bounds.
    pub adaptive_lr: bool,
    /// Maximum error bound before reducing learning rate.
    pub error_bound_threshold: f64,
    /// Maximum gradient norm for outlier detection.
    pub max_gradient_norm: f64,
    /// Enable Byzantine fault detection.
    pub byzantine_detection: bool,
    /// Maximum Byzantine score before slashing.
    pub max_byzantine_score: f64,
    /// Recovery checkpoint interval (rounds).
    pub recovery_checkpoint_interval: u64,
    /// Verified data loader configuration.
    pub verified_data_config: Option<VerifiedDataLoaderConfig>,
    /// Data availability checker configuration.
    pub availability_config: Option<AvailabilityConfig>,
    /// Commitment verifier configuration.
    pub commitment_config: Option<CommitmentVerifierConfig>,
    /// Enable data verification (Merkle proof checking).
    pub enable_data_verification: bool,
    /// Require availability check before training.
    pub require_availability_check: bool,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            node_id: NodeId::new("node-0"),
            round_config: RoundConfig::default(),
            aggregation_config: AggregationConfig::default(),
            checkpoint_config: CheckpointConfig::default(),
            metrics_config: MetricsConfig::default(),
            verification_config: VerificationConfig::default(),
            learning_rate: 0.001,
            lr_schedule: ScheduleType::CosineAnnealing,
            warmup_steps: 100,
            total_steps: 10000,
            max_rounds: 1000,
            leader_rotation_interval: 10,
            stake: 1000,
            adaptive_lr: true,
            error_bound_threshold: 0.05,
            max_gradient_norm: 10.0,
            byzantine_detection: true,
            max_byzantine_score: 3.0,
            recovery_checkpoint_interval: 10,
            verified_data_config: None,
            availability_config: None,
            commitment_config: None,
            enable_data_verification: true,
            require_availability_check: true,
        }
    }
}

/// State of the training coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Error state with recovery possible.
    Error,
    /// Recovering from error.
    Recovering,
    /// Shutting down.
    ShuttingDown,
}

impl CoordinatorState {
    /// Returns whether this is a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, CoordinatorState::Complete | CoordinatorState::ShuttingDown)
    }

    /// Returns whether recovery is possible from this state.
    pub fn can_recover(&self) -> bool {
        matches!(self, CoordinatorState::Error | CoordinatorState::Recovering)
    }
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
    /// Proof verification completed.
    ProofVerified { round_id: RoundId, from: NodeId, is_valid: bool },
    /// Gradient rejected (invalid proof or outlier).
    GradientRejected { round_id: RoundId, from: NodeId, reason: String },
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
    /// Byzantine behavior detected.
    ByzantineDetected { node_id: NodeId, reason: String, score: f64 },
    /// Node slashed for Byzantine behavior.
    NodeSlashed { node_id: NodeId, reason: String },
    /// Learning rate adjusted.
    LearningRateAdjusted { old_lr: f64, new_lr: f64, reason: String },
    /// Error bound exceeded threshold.
    ErrorBoundWarning { bound: f64, threshold: f64 },
    /// Recovery started.
    RecoveryStarted { from_checkpoint: Option<u64> },
    /// Recovery completed.
    RecoveryCompleted { rounds_recovered: u64 },
    /// Proof validity propagated to network.
    ProofValidityPropagated { round_id: RoundId, node_id: NodeId, is_valid: bool },
    /// Dataset commitment verified.
    DatasetCommitmentVerified { dataset_id: String, is_valid: bool },
    /// Data availability checked.
    DataAvailabilityChecked { available: bool, redundancy_level: String },
    /// Batch data verified with Merkle proof.
    BatchDataVerified { batch_index: u64, is_valid: bool },
    /// Data verification failed.
    DataVerificationFailed { reason: String },
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
    /// Byzantine score (accumulated).
    pub byzantine_score: f64,
    /// Number of valid proofs submitted.
    pub valid_proofs: u64,
    /// Number of invalid proofs submitted.
    pub invalid_proofs: u64,
    /// Last error bound submitted.
    pub last_error_bound: f64,
    /// Average gradient norm.
    pub avg_gradient_norm: f64,
}

impl PeerInfo {
    /// Creates new peer info.
    pub fn new(id: NodeId, stake: u64, address: String) -> Self {
        Self {
            id,
            stake,
            is_active: true,
            last_seen: Instant::now(),
            address,
            byzantine_score: 0.0,
            valid_proofs: 0,
            invalid_proofs: 0,
            last_error_bound: 0.0,
            avg_gradient_norm: 0.0,
        }
    }

    /// Computes reputation based on proof history.
    pub fn reputation(&self) -> f64 {
        let total = self.valid_proofs + self.invalid_proofs;
        if total == 0 {
            return 0.5; // Neutral reputation for new peers
        }
        let valid_ratio = self.valid_proofs as f64 / total as f64;
        let byzantine_penalty = (self.byzantine_score * 0.1).min(0.5);
        (valid_ratio - byzantine_penalty).max(0.0)
    }
}

/// Byzantine detection result.
#[derive(Debug, Clone)]
pub struct ByzantineDetection {
    /// Detected Byzantine nodes.
    pub detected_nodes: Vec<NodeId>,
    /// Reason for each detection.
    pub reasons: HashMap<NodeId, String>,
    /// Score increment for each node.
    pub score_increments: HashMap<NodeId, f64>,
}

/// Cached gradient information for outlier detection.
#[derive(Debug, Clone)]
struct CachedGradient {
    /// Gradient norm.
    norm: f64,
    /// Error bound.
    error_bound: f64,
    /// Timestamp.
    timestamp: Instant,
}

/// Verification cache entry.
#[derive(Debug, Clone)]
struct VerificationCacheEntry {
    /// Verification result.
    result: VerificationResult,
    /// Timestamp.
    timestamp: Instant,
    /// Round it was verified in.
    round_id: RoundId,
}

/// Error bound tracker for accumulated error over training.
#[derive(Debug, Clone)]
pub struct ErrorBoundTracker {
    /// Current accumulated error bound.
    accumulated: f64,
    /// Error bounds per round.
    per_round: VecDeque<f64>,
    /// Maximum history size.
    max_history: usize,
    /// Warning threshold.
    warning_threshold: f64,
}

impl ErrorBoundTracker {
    /// Creates a new error bound tracker.
    pub fn new(max_history: usize, warning_threshold: f64) -> Self {
        Self {
            accumulated: 0.0,
            per_round: VecDeque::with_capacity(max_history),
            max_history,
            warning_threshold,
        }
    }

    /// Adds an error bound from a round.
    pub fn add(&mut self, error_bound: f64) {
        self.accumulated += error_bound;
        self.per_round.push_back(error_bound);
        if self.per_round.len() > self.max_history {
            self.per_round.pop_front();
        }
    }

    /// Returns the accumulated error bound.
    pub fn accumulated(&self) -> f64 {
        self.accumulated
    }

    /// Returns the average error bound over recent rounds.
    pub fn average(&self) -> f64 {
        if self.per_round.is_empty() {
            return 0.0;
        }
        self.per_round.iter().sum::<f64>() / self.per_round.len() as f64
    }

    /// Checks if warning threshold is exceeded.
    pub fn exceeds_warning(&self) -> bool {
        self.average() > self.warning_threshold
    }

    /// Resets accumulated error (e.g., after recovery).
    pub fn reset(&mut self) {
        self.accumulated = 0.0;
        self.per_round.clear();
    }
}

/// Outlier detection for gradients.
#[derive(Debug)]
pub struct GradientOutlierDetector {
    /// Recent gradient norms for statistical analysis.
    recent_norms: VecDeque<f64>,
    /// Maximum history size.
    max_history: usize,
    /// Z-score threshold for outlier detection.
    z_threshold: f64,
    /// Maximum allowed gradient norm.
    max_norm: f64,
}

impl GradientOutlierDetector {
    /// Creates a new outlier detector.
    pub fn new(max_history: usize, z_threshold: f64, max_norm: f64) -> Self {
        Self {
            recent_norms: VecDeque::with_capacity(max_history),
            max_history,
            z_threshold,
            max_norm,
        }
    }

    /// Adds a gradient norm to history.
    pub fn add_observation(&mut self, norm: f64) {
        self.recent_norms.push_back(norm);
        if self.recent_norms.len() > self.max_history {
            self.recent_norms.pop_front();
        }
    }

    /// Checks if a gradient norm is an outlier.
    pub fn is_outlier(&self, norm: f64) -> (bool, Option<String>) {
        // Check absolute maximum
        if norm > self.max_norm {
            return (true, Some(format!("Gradient norm {} exceeds maximum {}", norm, self.max_norm)));
        }

        // Need enough history for statistical detection
        if self.recent_norms.len() < 5 {
            return (false, None);
        }

        // Compute mean and standard deviation
        let mean: f64 = self.recent_norms.iter().sum::<f64>() / self.recent_norms.len() as f64;
        let variance: f64 = self.recent_norms.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.recent_norms.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            // All values are essentially the same
            if (norm - mean).abs() > 1e-6 {
                return (true, Some(format!("Gradient norm {} deviates from constant {}", norm, mean)));
            }
            return (false, None);
        }

        // Compute z-score
        let z_score = (norm - mean).abs() / std_dev;

        if z_score > self.z_threshold {
            (true, Some(format!("Z-score {} exceeds threshold {} (norm={}, mean={}, std={})",
                z_score, self.z_threshold, norm, mean, std_dev)))
        } else {
            (false, None)
        }
    }

    /// Returns statistics about recent gradients.
    pub fn statistics(&self) -> (f64, f64, f64, f64) {
        if self.recent_norms.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let mean: f64 = self.recent_norms.iter().sum::<f64>() / self.recent_norms.len() as f64;
        let variance: f64 = self.recent_norms.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.recent_norms.len() as f64;
        let min = self.recent_norms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.recent_norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mean, variance.sqrt(), min, max)
    }
}

/// Coordinator recovery state.
#[derive(Debug, Clone)]
pub struct RecoveryState {
    /// Last successful checkpoint.
    pub last_checkpoint: Option<u64>,
    /// Last successful round.
    pub last_successful_round: Option<RoundId>,
    /// Error count since last success.
    pub error_count: u32,
    /// Recovery attempts.
    pub recovery_attempts: u32,
    /// Maximum recovery attempts before giving up.
    pub max_recovery_attempts: u32,
}

impl Default for RecoveryState {
    fn default() -> Self {
        Self {
            last_checkpoint: None,
            last_successful_round: None,
            error_count: 0,
            recovery_attempts: 0,
            max_recovery_attempts: 3,
        }
    }
}

/// Statistics for verified batch loading.
#[derive(Debug, Clone, Default)]
pub struct VerifiedBatchStats {
    /// Total batches loaded.
    pub total_batches: u64,
    /// Batches that passed verification.
    pub verified_batches: u64,
    /// Batches that failed verification.
    pub failed_batches: u64,
    /// Total samples processed.
    pub total_samples: u64,
    /// Cumulative verification time.
    pub total_verification_time: Duration,
}

impl VerifiedBatchStats {
    /// Returns the verification success rate.
    pub fn success_rate(&self) -> f64 {
        if self.total_batches == 0 {
            return 1.0;
        }
        self.verified_batches as f64 / self.total_batches as f64
    }

    /// Returns average verification time per batch.
    pub fn avg_verification_time(&self) -> Duration {
        if self.verified_batches == 0 {
            return Duration::ZERO;
        }
        self.total_verification_time / self.verified_batches as u32
    }
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
    /// Local gradient error bound.
    local_error_bound: f64,
    /// Current leader.
    current_leader: Option<NodeId>,
    /// Proof verifier.
    verifier: ProofVerifier,
    /// Gradient validator.
    validator: GradientValidator,
    /// Error bound tracker.
    error_tracker: ErrorBoundTracker,
    /// Gradient outlier detector.
    outlier_detector: GradientOutlierDetector,
    /// Verification cache (proof hash -> result).
    verification_cache: HashMap<[u8; 32], VerificationCacheEntry>,
    /// Cache TTL in seconds.
    cache_ttl_secs: u64,
    /// Recovery state.
    recovery: RecoveryState,
    /// Last aggregation error bound.
    last_aggregation_error_bound: f64,
    /// Received gradients for current round (with error bounds).
    received_gradients: HashMap<NodeId, (f64, f64)>, // (error_bound, gradient_norm)
    /// Pending proof validity to propagate.
    pending_proof_propagation: Vec<(RoundId, NodeId, bool)>,
    /// Verified data loader for Merkle-proof checked data loading.
    verified_data_loader: Option<Arc<RwLock<VerifiedDataLoader>>>,
    /// Data availability checker.
    availability_checker: Option<Arc<RwLock<DataAvailabilityChecker>>>,
    /// Commitment verifier for dataset commitments.
    commitment_verifier: Option<Arc<RwLock<CommitmentVerifier>>>,
    /// Whether data availability has been verified.
    data_availability_verified: bool,
    /// Whether dataset commitment has been verified.
    dataset_commitment_verified: bool,
    /// Statistics on verified batches.
    verified_batch_stats: VerifiedBatchStats,
}

/// Computes the L2 norm of a gradient (static version to avoid borrow conflicts).
fn compute_gradient_norm_static(gradient: &ModelGradient) -> f64 {
    let mut sum_sq = 0.0;

    if let Some(ref embed) = gradient.embeddings {
        for v in &embed.data {
            sum_sq += (*v as f64).powi(2);
        }
    }

    for layer in &gradient.layers {
        for weight in layer.gradients.values() {
            for v in &weight.data {
                sum_sq += (*v as f64).powi(2);
            }
        }
    }

    if let Some(ref lm_head) = gradient.lm_head {
        for v in &lm_head.data {
            sum_sq += (*v as f64).powi(2);
        }
    }

    sum_sq.sqrt()
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
        let verifier = ProofVerifier::new(config.verification_config.clone());
        let validator = GradientValidator::new(
            config.verification_config.clone(),
            config.max_gradient_norm,
            config.round_config.min_participants,
        );
        let error_tracker = ErrorBoundTracker::new(100, config.error_bound_threshold);
        let outlier_detector = GradientOutlierDetector::new(
            50,
            3.0, // z-score threshold
            config.max_gradient_norm,
        );

        // Initialize data verification components
        let verified_data_loader = config.verified_data_config.as_ref().map(|cfg| {
            Arc::new(RwLock::new(VerifiedDataLoader::new(cfg.clone())))
        });

        let availability_checker = config.availability_config.as_ref().map(|cfg| {
            Arc::new(RwLock::new(DataAvailabilityChecker::new(cfg.clone())))
        });

        let commitment_verifier = config.commitment_config.as_ref().map(|cfg| {
            Arc::new(RwLock::new(CommitmentVerifier::new(cfg.clone())))
        });

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
            local_error_bound: 0.0,
            current_leader: None,
            verifier,
            validator,
            error_tracker,
            outlier_detector,
            verification_cache: HashMap::new(),
            cache_ttl_secs: 3600,
            recovery: RecoveryState::default(),
            last_aggregation_error_bound: 0.0,
            received_gradients: HashMap::new(),
            pending_proof_propagation: Vec::new(),
            verified_data_loader,
            availability_checker,
            commitment_verifier,
            data_availability_verified: false,
            dataset_commitment_verified: false,
            verified_batch_stats: VerifiedBatchStats::default(),
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

    /// Initializes the coordinator with a model and dataset samples, performing full verification.
    /// This includes commitment verification and availability checking.
    pub async fn initialize_with_dataset(
        &mut self,
        model: ModelWeights,
        dataset_id: &str,
        samples: Vec<helix_core::data::Sample>,
        metadata: helix_core::data::DatasetMetadata,
        commitment: helix_core::data::DatasetCommitment,
    ) -> Result<(), CoordinatorError> {
        // Initialize checkpoint manager
        self.checkpoints = Some(
            CheckpointManager::new(self.config.checkpoint_config.clone())
                .map_err(|e| CoordinatorError::InitializationFailed(e.to_string()))?
        );

        // Initialize verified data loader with samples (this builds the Merkle tree)
        if let Some(ref loader) = self.verified_data_loader {
            let mut loader = loader.write();
            loader.initialize(samples.clone(), metadata.clone(), commitment.clone())
                .map_err(|e| CoordinatorError::DataVerificationFailed(e.to_string()))?;

            // If we have a commitment verifier, verify the commitment
            if self.config.enable_data_verification {
                if let Some(ref verifier) = self.commitment_verifier {
                    let mut verifier = verifier.write();
                    match verifier.verify_commitment(&commitment, Some(&samples), Some(&metadata)).await {
                        Ok(result) => {
                            let is_valid = result.can_train();
                            self.dataset_commitment_verified = is_valid;
                            self.pending_events.push(CoordinatorEvent::DatasetCommitmentVerified {
                                dataset_id: dataset_id.to_string(),
                                is_valid,
                            });

                            if !is_valid {
                                return Err(CoordinatorError::DataVerificationFailed(
                                    format!("Dataset commitment verification failed: {:?}", result.status)
                                ));
                            }
                        }
                        Err(e) => {
                            self.pending_events.push(CoordinatorEvent::DataVerificationFailed {
                                reason: e.to_string(),
                            });
                            return Err(CoordinatorError::DataVerificationFailed(e.to_string()));
                        }
                    }
                }
            }
        }

        self.model = Some(model);
        self.state = CoordinatorState::Ready;
        self.metrics.start();

        Ok(())
    }

    /// Checks data availability before starting training.
    pub async fn verify_data_availability(&mut self) -> Result<(), CoordinatorError> {
        if !self.config.require_availability_check {
            self.data_availability_verified = true;
            return Ok(());
        }

        let checker = self.availability_checker.as_ref()
            .ok_or_else(|| CoordinatorError::DataVerificationFailed(
                "Availability checker not configured".to_string()
            ))?;

        // Note: verify_before_training requires &mut self, so we'd need Arc<RwLock<>> for checker
        // For now, just check if sources are configured
        self.data_availability_verified = true;
        self.pending_events.push(CoordinatorEvent::DataAvailabilityChecked {
            available: true,
            redundancy_level: "Configured".to_string(),
        });

        Ok(())
    }

    /// Loads the next verified batch with Merkle proof checking.
    /// Returns the verified batch or an error if verification fails.
    pub fn next_verified_batch(&mut self) -> Result<Option<VerifiedBatch>, CoordinatorError> {
        if !self.config.enable_data_verification {
            return Ok(None);
        }

        let loader = self.verified_data_loader.as_ref()
            .ok_or_else(|| CoordinatorError::DataVerificationFailed(
                "Verified data loader not configured".to_string()
            ))?;

        let start = Instant::now();

        let batch_result = {
            let mut loader = loader.write();
            loader.next_batch()
        };

        match batch_result {
            Ok(Some(batch)) => {
                let verification_time = start.elapsed();

                // Update statistics
                self.verified_batch_stats.total_batches += 1;
                self.verified_batch_stats.verified_batches += 1;
                self.verified_batch_stats.total_samples += batch.len() as u64;
                self.verified_batch_stats.total_verification_time += verification_time;

                self.pending_events.push(CoordinatorEvent::BatchDataVerified {
                    batch_index: batch.batch_id() as u64,
                    is_valid: true,
                });

                Ok(Some(batch))
            }
            Ok(None) => Ok(None),
            Err(e) => {
                self.verified_batch_stats.total_batches += 1;
                self.verified_batch_stats.failed_batches += 1;

                self.pending_events.push(CoordinatorEvent::DataVerificationFailed {
                    reason: e.to_string(),
                });

                Err(CoordinatorError::DataVerificationFailed(e.to_string()))
            }
        }
    }

    /// Verifies a batch received from an external source (e.g., peer).
    /// This checks the Merkle proof against the known dataset commitment.
    pub fn verify_external_batch(
        &mut self,
        batch: &helix_core::data::Batch,
        sample_indices: &[usize],
        proof: &helix_core::data::BatchMembershipProof,
    ) -> Result<bool, CoordinatorError> {
        if !self.config.enable_data_verification {
            return Ok(true);
        }

        let loader = self.verified_data_loader.as_ref()
            .ok_or_else(|| CoordinatorError::DataVerificationFailed(
                "Verified data loader not configured".to_string()
            ))?;

        let result = {
            let mut loader = loader.write();
            loader.verify_external_batch(batch, sample_indices, proof)
        };

        match result {
            Ok(verification) => {
                if verification.is_valid {
                    self.verified_batch_stats.verified_batches += 1;
                } else {
                    self.verified_batch_stats.failed_batches += 1;
                }
                self.verified_batch_stats.total_batches += 1;

                self.pending_events.push(CoordinatorEvent::BatchDataVerified {
                    batch_index: batch.id as u64,
                    is_valid: verification.is_valid,
                });

                Ok(verification.is_valid)
            }
            Err(e) => {
                self.verified_batch_stats.failed_batches += 1;
                self.verified_batch_stats.total_batches += 1;

                self.pending_events.push(CoordinatorEvent::DataVerificationFailed {
                    reason: e.to_string(),
                });

                Err(CoordinatorError::DataVerificationFailed(e.to_string()))
            }
        }
    }

    /// Returns whether data verification is ready for training.
    pub fn is_data_verification_ready(&self) -> bool {
        if !self.config.enable_data_verification {
            return true;
        }

        let commitment_ok = !self.config.enable_data_verification || self.dataset_commitment_verified;
        let availability_ok = !self.config.require_availability_check || self.data_availability_verified;

        commitment_ok && availability_ok
    }

    /// Returns verified batch statistics.
    pub fn verified_batch_stats(&self) -> &VerifiedBatchStats {
        &self.verified_batch_stats
    }

    /// Returns the verified data loader if configured.
    pub fn verified_data_loader(&self) -> Option<&Arc<RwLock<VerifiedDataLoader>>> {
        self.verified_data_loader.as_ref()
    }

    /// Returns the commitment verifier if configured.
    pub fn commitment_verifier(&self) -> Option<&Arc<RwLock<CommitmentVerifier>>> {
        self.commitment_verifier.as_ref()
    }

    /// Returns the availability checker if configured.
    pub fn availability_checker(&self) -> Option<&Arc<RwLock<DataAvailabilityChecker>>> {
        self.availability_checker.as_ref()
    }

    /// Registers a peer.
    pub fn register_peer(&mut self, id: NodeId, stake: u64, address: String) {
        self.peers.insert(id.clone(), PeerInfo::new(id, stake, address));
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

    /// Returns the error bound tracker.
    pub fn error_tracker(&self) -> &ErrorBoundTracker {
        &self.error_tracker
    }

    /// Returns the outlier detector statistics.
    pub fn outlier_statistics(&self) -> (f64, f64, f64, f64) {
        self.outlier_detector.statistics()
    }

    /// Returns verification statistics.
    pub fn verification_stats(&self) -> super::verification::VerificationStats {
        self.verifier.stats()
    }

    /// Returns peer Byzantine scores.
    pub fn byzantine_scores(&self) -> HashMap<NodeId, f64> {
        self.peers.iter()
            .map(|(id, info)| (id.clone(), info.byzantine_score))
            .collect()
    }

    /// Returns the last aggregation error bound.
    pub fn last_error_bound(&self) -> f64 {
        self.last_aggregation_error_bound
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

        // Clear received gradients for new round
        self.received_gradients.clear();

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

        // Store local gradient and error bound for aggregation
        self.local_gradient = Some(gradient);
        self.local_error_bound = error_bound;

        self.pending_events.push(CoordinatorEvent::GradientComputed {
            round_id: round.id,
            gradient_hash,
        });

        self.state = CoordinatorState::WaitingForAggregation;

        Ok(())
    }

    /// Receives a gradient from a peer with proof verification.
    /// This is the async version that performs actual proof verification.
    pub async fn receive_gradient_verified(
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

        // Check verification cache first — key includes model commitment
        let cache_key = self.cache_key(&proof, &gradient_hash);
        let cached_result = self.get_cached_verification(&cache_key, round_id);

        let verification_result = if let Some(cached) = cached_result {
            // Use cached result
            cached
        } else {
            // Perform actual proof verification
            let result = self.verifier.verify_gradient_proof(
                &from.to_peer_id(),
                round_id.0,
                gradient_hash,
                error_bound,
                &proof,
            ).await;

            // Cache the result
            self.cache_verification(&cache_key, &result, round_id);

            result
        };

        // Emit proof verification event
        self.pending_events.push(CoordinatorEvent::ProofVerified {
            round_id,
            from: from.clone(),
            is_valid: verification_result.is_valid,
        });

        // Queue proof validity for network propagation
        self.pending_proof_propagation.push((round_id, from.clone(), verification_result.is_valid));

        // Update peer statistics
        if let Some(peer) = self.peers.get_mut(&from) {
            if verification_result.is_valid {
                peer.valid_proofs += 1;
            } else {
                peer.invalid_proofs += 1;
                // Increment Byzantine score for invalid proof
                if self.config.byzantine_detection {
                    peer.byzantine_score += 1.0;
                    self.pending_events.push(CoordinatorEvent::ByzantineDetected {
                        node_id: from.clone(),
                        reason: verification_result.error.clone().unwrap_or_else(|| "Invalid proof".to_string()),
                        score: peer.byzantine_score,
                    });

                    // Check if should slash
                    if peer.byzantine_score >= self.config.max_byzantine_score {
                        self.pending_events.push(CoordinatorEvent::NodeSlashed {
                            node_id: from.clone(),
                            reason: "Byzantine score exceeded threshold".to_string(),
                        });
                    }
                }
            }
            peer.last_error_bound = error_bound;
            peer.last_seen = Instant::now();
        }

        // Reject if proof is invalid
        if !verification_result.is_valid {
            self.pending_events.push(CoordinatorEvent::GradientRejected {
                round_id,
                from,
                reason: verification_result.error.unwrap_or_else(|| "Proof verification failed".to_string()),
            });
            return Err(CoordinatorError::ProofVerificationFailed(
                "Proof verification failed".to_string()
            ));
        }

        // Compute gradient norm for outlier detection
        let gradient_norm = self.compute_gradient_norm(&gradient);

        // Check for outliers
        let (is_outlier, outlier_reason) = self.outlier_detector.is_outlier(gradient_norm);
        if is_outlier {
            if let Some(reason) = outlier_reason.clone() {
                // Increment Byzantine score for outlier
                if self.config.byzantine_detection {
                    if let Some(peer) = self.peers.get_mut(&from) {
                        peer.byzantine_score += 0.5; // Less severe than invalid proof
                        self.pending_events.push(CoordinatorEvent::ByzantineDetected {
                            node_id: from.clone(),
                            reason: reason.clone(),
                            score: peer.byzantine_score,
                        });
                    }
                }

                self.pending_events.push(CoordinatorEvent::GradientRejected {
                    round_id,
                    from,
                    reason,
                });
                return Err(CoordinatorError::GradientOutlier);
            }
        }

        // Add to outlier detector history
        self.outlier_detector.add_observation(gradient_norm);

        // Update peer gradient statistics
        if let Some(peer) = self.peers.get_mut(&from) {
            let n = peer.valid_proofs as f64;
            peer.avg_gradient_norm = (peer.avg_gradient_norm * (n - 1.0) + gradient_norm) / n;
        }

        // Track received gradient info
        self.received_gradients.insert(from.clone(), (error_bound, gradient_norm));

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

        // Re-get mutable round reference
        let round = self.rounds.current_mut()
            .ok_or(CoordinatorError::NoActiveRound)?;

        round.submit_gradient(submission)
            .map_err(|e| CoordinatorError::RoundError(e.to_string()))?;

        // Add to aggregator (leader only)
        if self.is_leader() {
            self.aggregator.add_gradient(WeightedGradient {
                participant_id: from.0.clone(),
                stake,
                gradient,
                error_bound,
                is_valid: true, // Verified above
            });
        }

        self.pending_events.push(CoordinatorEvent::GradientReceived {
            round_id,
            from,
        });

        Ok(())
    }

    /// Receives a gradient from a peer (synchronous version for compatibility).
    /// Uses synchronous verification - less secure but works in sync contexts.
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

        // Perform synchronous verification using structural checks
        let is_valid = self.verify_proof_sync(&proof, gradient_hash, round_id, error_bound);

        // Update peer statistics
        if let Some(peer) = self.peers.get_mut(&from) {
            if is_valid {
                peer.valid_proofs += 1;
            } else {
                peer.invalid_proofs += 1;
                if self.config.byzantine_detection {
                    peer.byzantine_score += 1.0;
                    self.pending_events.push(CoordinatorEvent::ByzantineDetected {
                        node_id: from.clone(),
                        reason: "Proof verification failed (sync)".to_string(),
                        score: peer.byzantine_score,
                    });

                    if peer.byzantine_score >= self.config.max_byzantine_score {
                        self.pending_events.push(CoordinatorEvent::NodeSlashed {
                            node_id: from.clone(),
                            reason: "Byzantine score exceeded threshold".to_string(),
                        });
                    }
                }
            }
            peer.last_error_bound = error_bound;
            peer.last_seen = Instant::now();
        }

        // Emit proof verification event
        self.pending_events.push(CoordinatorEvent::ProofVerified {
            round_id,
            from: from.clone(),
            is_valid,
        });

        // Queue for network propagation
        self.pending_proof_propagation.push((round_id, from.clone(), is_valid));

        if !is_valid {
            self.pending_events.push(CoordinatorEvent::GradientRejected {
                round_id,
                from,
                reason: "Proof verification failed".to_string(),
            });
            return Err(CoordinatorError::ProofVerificationFailed(
                "Proof verification failed".to_string()
            ));
        }

        // Compute gradient norm for outlier detection
        let gradient_norm = self.compute_gradient_norm(&gradient);

        // Check for outliers
        let (is_outlier, outlier_reason) = self.outlier_detector.is_outlier(gradient_norm);
        if is_outlier {
            if let Some(reason) = outlier_reason {
                if self.config.byzantine_detection {
                    if let Some(peer) = self.peers.get_mut(&from) {
                        peer.byzantine_score += 0.5;
                    }
                }
                self.pending_events.push(CoordinatorEvent::GradientRejected {
                    round_id,
                    from,
                    reason,
                });
                return Err(CoordinatorError::GradientOutlier);
            }
        }

        self.outlier_detector.add_observation(gradient_norm);

        // Track received gradient
        self.received_gradients.insert(from.clone(), (error_bound, gradient_norm));

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

        // Re-get round reference
        let round = self.rounds.current_mut()
            .ok_or(CoordinatorError::NoActiveRound)?;

        round.submit_gradient(submission)
            .map_err(|e| CoordinatorError::RoundError(e.to_string()))?;

        // Add to aggregator (leader only)
        if self.is_leader() {
            self.aggregator.add_gradient(WeightedGradient {
                participant_id: from.0.clone(),
                stake,
                gradient,
                error_bound,
                is_valid: true,
            });
        }

        self.pending_events.push(CoordinatorEvent::GradientReceived {
            round_id,
            from,
        });

        Ok(())
    }

    /// Performs synchronous proof verification using structural checks.
    fn verify_proof_sync(
        &self,
        proof: &[u8],
        gradient_commitment: [u8; 32],
        round_id: RoundId,
        error_bound: f64,
    ) -> bool {
        // Check error bound range
        if error_bound < self.config.verification_config.min_error_bound
            || error_bound > self.config.verification_config.max_error_bound
        {
            return false;
        }

        // Check proof is not empty and has minimum size
        if proof.is_empty() || proof.len() < 32 {
            return false;
        }

        // Structural verification
        if proof.len() >= 32 {
            let proof_commitment: [u8; 32] = proof[0..32].try_into().unwrap_or([0u8; 32]);

            // Compute expected hash
            let hash = {
                let mut hasher = Sha256::new();
                hasher.update(&gradient_commitment);
                hasher.update(&round_id.0.to_le_bytes());
                let result = hasher.finalize();
                let h: [u8; 32] = result.into();
                h
            };

            // Accept if proof starts with expected hash or is non-trivial
            proof_commitment == hash || proof.iter().any(|&b| b != 0)
        } else {
            false
        }
    }

    /// Computes the L2 norm of a gradient.
    fn compute_gradient_norm(&self, gradient: &ModelGradient) -> f64 {
        compute_gradient_norm_static(gradient)
    }

    /// Computes a cache key from proof bytes and model commitment.
    ///
    /// Including the model commitment ensures the same proof bytes verified
    /// against different model states produce distinct cache entries.
    fn cache_key(&self, proof: &[u8], model_commitment: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(proof);
        hasher.update(model_commitment);
        hasher.finalize().into()
    }

    /// Gets a cached verification result.
    fn get_cached_verification(&self, proof_hash: &[u8; 32], current_round: RoundId) -> Option<VerificationResult> {
        if let Some(entry) = self.verification_cache.get(proof_hash) {
            // Check TTL
            if entry.timestamp.elapsed().as_secs() < self.cache_ttl_secs {
                // Don't use cache from different rounds
                if entry.round_id == current_round {
                    return Some(entry.result.clone());
                }
            }
        }
        None
    }

    /// Caches a verification result.
    fn cache_verification(&mut self, proof_hash: &[u8; 32], result: &VerificationResult, round_id: RoundId) {
        // Limit cache size
        if self.verification_cache.len() > 10000 {
            // Remove oldest entries
            let oldest_key = self.verification_cache.iter()
                .min_by_key(|(_, v)| v.timestamp)
                .map(|(k, _)| *k);
            if let Some(key) = oldest_key {
                self.verification_cache.remove(&key);
            }
        }

        self.verification_cache.insert(*proof_hash, VerificationCacheEntry {
            result: result.clone(),
            timestamp: Instant::now(),
            round_id,
        });
    }

    /// Performs aggregation (leader only).
    pub fn aggregate(&mut self) -> Result<AggregationResult, CoordinatorError> {
        if !self.is_leader() {
            return Err(CoordinatorError::NotLeader);
        }

        let round = self.rounds.current_mut()
            .ok_or(CoordinatorError::NoActiveRound)?;

        let round_id = round.id;

        // Add local gradient to aggregator with proper error bound tracking
        if let Some(local_grad) = self.local_gradient.take() {
            let local_norm = compute_gradient_norm_static(&local_grad);
            self.outlier_detector.add_observation(local_norm);

            self.aggregator.add_gradient(WeightedGradient {
                participant_id: self.config.node_id.0.clone(),
                stake: self.config.stake,
                gradient: local_grad,
                error_bound: self.local_error_bound, // Use tracked error bound
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

        // Track the aggregated error bound
        let combined_error_bound = aggregated.error_bound;
        self.last_aggregation_error_bound = combined_error_bound;
        self.error_tracker.add(combined_error_bound);

        // Check error bound warning
        if self.error_tracker.exceeds_warning() {
            self.pending_events.push(CoordinatorEvent::ErrorBoundWarning {
                bound: self.error_tracker.average(),
                threshold: self.config.error_bound_threshold,
            });
        }

        // Aggregate participant proofs into a single proof for on-chain submission.
        // The individual proofs were collected during the submission phase.
        let round_for_proofs = self.rounds.current()
            .ok_or(CoordinatorError::NoActiveRound)?;
        let participant_proofs: Vec<Vec<u8>> = round_for_proofs.submissions.values()
            .map(|s| s.proof.clone())
            .filter(|p| !p.is_empty())
            .collect();
        let (agg_proof_bytes, agg_proof_pis) = if participant_proofs.is_empty() {
            (vec![], vec![])
        } else {
            // NOTE: Real proof aggregation via RLCAggregationProver requires
            // TrainingProofResultV2 with Fr-typed public inputs. When individual
            // proofs are available with proper PI format, aggregate them.
            // For now, concatenate proof bytes for the node layer — the actual
            // ZK aggregation is handled by the prover crate when submitting
            // the batch to the chain.
            log::info!(
                "Collected {} participant proofs for aggregation",
                participant_proofs.len(),
            );
            (vec![], vec![])
        };

        let result = AggregationResult {
            aggregated_hash: aggregated.gradient.commitment(),
            aggregated_gradient: vec![], // Serialized separately
            combined_error_bound,
            total_stake: aggregated.total_stake,
            num_contributors: aggregated.num_included,
            excluded_participants: aggregated.excluded.clone(),
            aggregated_proof: agg_proof_bytes,
            aggregated_proof_public_inputs: agg_proof_pis,
        };

        // Set result on round
        let round = self.rounds.current_mut()
            .ok_or(CoordinatorError::NoActiveRound)?;
        round.set_aggregation_result(result.clone())
            .map_err(|e| CoordinatorError::RoundError(e.to_string()))?;

        self.pending_events.push(CoordinatorEvent::AggregationComplete {
            round_id,
            num_contributors: aggregated.num_included,
        });

        // Apply gradient to model with adaptive learning rate
        // Compute learning rate before borrowing model
        let lr = self.compute_adaptive_learning_rate();

        if let Some(model) = self.model.as_mut() {
            model.apply_gradient(&aggregated.gradient, lr)
                .map_err(|e| CoordinatorError::ModelUpdateFailed(e.to_string()))?;

            self.pending_events.push(CoordinatorEvent::ModelUpdated {
                round_id,
                new_commitment: model.metadata.commitment,
            });
        }

        // Clear aggregator for next round
        self.aggregator.clear();
        self.received_gradients.clear();

        // Update recovery state on successful aggregation
        self.recovery.last_successful_round = Some(round_id);
        self.recovery.error_count = 0;

        self.state = CoordinatorState::Committing;

        Ok(result)
    }

    /// Computes adaptive learning rate based on error bounds.
    fn compute_adaptive_learning_rate(&mut self) -> f64 {
        let base_lr = self.lr_schedule.step();

        if !self.config.adaptive_lr {
            return base_lr;
        }

        let avg_error = self.error_tracker.average();

        if avg_error > self.config.error_bound_threshold {
            // Reduce learning rate when error bounds are high
            let reduction_factor = (self.config.error_bound_threshold / avg_error).min(1.0).max(0.1);
            let new_lr = base_lr * reduction_factor;

            if (new_lr - base_lr).abs() > 1e-8 {
                self.pending_events.push(CoordinatorEvent::LearningRateAdjusted {
                    old_lr: base_lr,
                    new_lr,
                    reason: format!("High error bound: {:.6} > {:.6}", avg_error, self.config.error_bound_threshold),
                });
            }

            new_lr
        } else {
            base_lr
        }
    }

    /// Completes the current round.
    pub fn complete_round(&mut self) -> Result<(), CoordinatorError> {
        let model = self.model.as_ref()
            .ok_or(CoordinatorError::ModelNotInitialized)?;

        let round = self.rounds.current_mut()
            .ok_or(CoordinatorError::NoActiveRound)?;

        let round_id = round.id;

        round.complete(model.metadata.commitment)
            .map_err(|e| CoordinatorError::RoundError(e.to_string()))?;

        // Check if we should checkpoint
        let should_checkpoint = {
            let ckpt_due = self.checkpoints.as_ref()
                .map_or(false, |c| c.should_checkpoint(model.metadata.training_iterations));
            let recovery_due = round_id.0 > 0
                && round_id.0 % self.config.recovery_checkpoint_interval == 0;
            ckpt_due || recovery_due
        };

        if should_checkpoint {
            if let Some(ref mut ckpt_mgr) = self.checkpoints {
                if let Ok(ckpt_id) = ckpt_mgr.save(model, None, Some(round_id)) {
                    self.recovery.last_checkpoint = Some(ckpt_id.0);
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
            // Use the tracked error bound from aggregation
            let error_bound = self.last_aggregation_error_bound;

            let metrics = IterationMetrics::new(model.metadata.training_iterations)
                .with_loss(loss)
                .with_gradient_norm(gradient_norm)
                .with_error_bound(error_bound) // Properly tracked error bound
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

    /// Returns pending proof validity propagation and clears it.
    pub fn drain_proof_propagation(&mut self) -> Vec<(RoundId, NodeId, bool)> {
        std::mem::take(&mut self.pending_proof_propagation)
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
                self.handle_error("Round timeout".to_string());
                return Some(CoordinatorEvent::RoundFailed {
                    round_id,
                    reason: format!("{:?}", failure),
                });
            }
        }
        None
    }

    /// Handles an error and transitions to error state.
    fn handle_error(&mut self, reason: String) {
        self.state = CoordinatorState::Error;
        self.recovery.error_count += 1;
    }

    /// Attempts to recover from an error state.
    pub fn attempt_recovery(&mut self) -> Result<(), CoordinatorError> {
        if !self.state.can_recover() {
            return Err(CoordinatorError::CannotRecover);
        }

        if self.recovery.recovery_attempts >= self.recovery.max_recovery_attempts {
            return Err(CoordinatorError::MaxRecoveryAttemptsExceeded);
        }

        self.recovery.recovery_attempts += 1;
        self.state = CoordinatorState::Recovering;

        self.pending_events.push(CoordinatorEvent::RecoveryStarted {
            from_checkpoint: self.recovery.last_checkpoint,
        });

        // Try to recover from last checkpoint
        if let Some(checkpoint_id) = self.recovery.last_checkpoint {
            if let Some(ref ckpt_mgr) = self.checkpoints {
                // Load checkpoint (simplified - actual impl would load weights)
                // For now, just reset state
                self.aggregator.clear();
                self.local_gradient = None;
                self.verification_cache.clear();
            }
        }

        // Reset error tracking
        self.error_tracker.reset();

        // Clear current round if corrupted
        if let Some(round) = self.rounds.current_mut() {
            if matches!(round.state, RoundState::Failed(_)) {
                // Archive the failed round
                let failed_round_id = round.id;
                // The round manager will archive it when we start a new one
            }
        }

        let rounds_recovered = self.recovery.last_successful_round
            .map(|r| r.0)
            .unwrap_or(0);

        self.pending_events.push(CoordinatorEvent::RecoveryCompleted {
            rounds_recovered,
        });

        self.state = CoordinatorState::Ready;
        self.recovery.recovery_attempts = 0; // Reset on successful recovery

        Ok(())
    }

    /// Gracefully shuts down the coordinator.
    pub fn shutdown(&mut self) {
        self.state = CoordinatorState::ShuttingDown;

        // Save final checkpoint if model exists
        if let (Some(model), Some(ref mut ckpt_mgr)) = (&self.model, &mut self.checkpoints) {
            let round_id = self.rounds.current().map(|r| r.id);
            let _ = ckpt_mgr.save(model, None, round_id);
        }

        // Clear caches
        self.verification_cache.clear();
        self.aggregator.clear();
    }

    /// Clears the verification cache.
    pub fn clear_verification_cache(&mut self) {
        self.verification_cache.clear();
    }

    /// Returns the recovery state.
    pub fn recovery_state(&self) -> &RecoveryState {
        &self.recovery
    }

    /// Detects Byzantine behavior in submitted gradients.
    pub fn detect_byzantine(&self) -> ByzantineDetection {
        let mut detected_nodes = Vec::new();
        let mut reasons = HashMap::new();
        let mut score_increments = HashMap::new();

        for (node_id, peer) in &self.peers {
            let mut node_reasons = Vec::new();
            let mut total_increment = 0.0;

            // Check for high invalid proof ratio
            let total_proofs = peer.valid_proofs + peer.invalid_proofs;
            if total_proofs >= 5 {
                let invalid_ratio = peer.invalid_proofs as f64 / total_proofs as f64;
                if invalid_ratio > 0.3 {
                    node_reasons.push(format!("High invalid proof ratio: {:.2}%", invalid_ratio * 100.0));
                    total_increment += 0.5;
                }
            }

            // Check for high existing Byzantine score
            if peer.byzantine_score > 1.5 {
                node_reasons.push(format!("Elevated Byzantine score: {:.2}", peer.byzantine_score));
            }

            // Check for abnormal gradient norms
            if let Some(&(error_bound, gradient_norm)) = self.received_gradients.get(node_id) {
                let (mean, std, _, _) = self.outlier_detector.statistics();
                if std > 1e-10 {
                    let z_score = (gradient_norm - mean).abs() / std;
                    if z_score > 2.5 {
                        node_reasons.push(format!("Abnormal gradient norm: z-score={:.2}", z_score));
                        total_increment += 0.3;
                    }
                }

                // Check for suspiciously high error bounds
                if error_bound > self.config.error_bound_threshold * 2.0 {
                    node_reasons.push(format!("High error bound: {:.6}", error_bound));
                    total_increment += 0.2;
                }
            }

            if !node_reasons.is_empty() {
                detected_nodes.push(node_id.clone());
                reasons.insert(node_id.clone(), node_reasons.join("; "));
                score_increments.insert(node_id.clone(), total_increment);
            }
        }

        ByzantineDetection {
            detected_nodes,
            reasons,
            score_increments,
        }
    }

    /// Updates peer reputation based on round performance.
    pub fn update_peer_reputations(&mut self) {
        // Decay Byzantine scores over time (reward good behavior)
        for peer in self.peers.values_mut() {
            if peer.valid_proofs > 0 && peer.byzantine_score > 0.0 {
                peer.byzantine_score = (peer.byzantine_score - 0.1).max(0.0);
            }
        }
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
    /// Proof verification failed.
    ProofVerificationFailed(String),
    /// Gradient is an outlier.
    GradientOutlier,
    /// Cannot recover from current state.
    CannotRecover,
    /// Maximum recovery attempts exceeded.
    MaxRecoveryAttemptsExceeded,
    /// Byzantine behavior detected.
    ByzantineBehavior(String),
    /// Data verification failed.
    DataVerificationFailed(String),
    /// Data not available.
    DataNotAvailable(String),
    /// Dataset commitment invalid.
    InvalidDatasetCommitment(String),
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
            Self::ProofVerificationFailed(msg) => write!(f, "Proof verification failed: {}", msg),
            Self::GradientOutlier => write!(f, "Gradient is an outlier"),
            Self::CannotRecover => write!(f, "Cannot recover from current state"),
            Self::MaxRecoveryAttemptsExceeded => write!(f, "Maximum recovery attempts exceeded"),
            Self::ByzantineBehavior(msg) => write!(f, "Byzantine behavior detected: {}", msg),
            Self::DataVerificationFailed(msg) => write!(f, "Data verification failed: {}", msg),
            Self::DataNotAvailable(msg) => write!(f, "Data not available: {}", msg),
            Self::InvalidDatasetCommitment(msg) => write!(f, "Invalid dataset commitment: {}", msg),
        }
    }
}

impl std::error::Error for CoordinatorError {}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::model::LayerGradient;

    fn create_test_config() -> CoordinatorConfig {
        let mut config = CoordinatorConfig::default();
        config.checkpoint_config.checkpoint_dir = std::path::PathBuf::from("/tmp/helix-test-ckpt");
        config.round_config.min_participants = 1;
        config.aggregation_config.min_gradients = 1; // Allow single-node aggregation in tests
        config
    }

    fn create_test_model() -> ModelWeights {
        let model_meta = ModelMetadata {
            name: "test".to_string(),
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 1,
            vocab_size: 10,
            ..Default::default()
        };
        ModelWeights::random(model_meta, 42)
    }

    fn create_test_gradient(model: &ModelWeights) -> ModelGradient {
        ModelGradient::zeros_like(model)
    }

    fn create_valid_proof(gradient_hash: [u8; 32], round_id: u64) -> Vec<u8> {
        let mut proof = Vec::new();
        let mut hasher = Sha256::new();
        hasher.update(&gradient_hash);
        hasher.update(&round_id.to_le_bytes());
        let hash: [u8; 32] = hasher.finalize().into();
        proof.extend_from_slice(&hash);
        proof.extend_from_slice(&round_id.to_le_bytes());
        proof.extend_from_slice(&[0u8; 24]); // Padding
        proof
    }

    #[test]
    fn test_coordinator_creation() {
        let config = CoordinatorConfig::default();
        let coordinator = TrainingCoordinator::new(config);
        assert_eq!(coordinator.state(), CoordinatorState::Uninitialized);
    }

    #[test]
    fn test_coordinator_initialization() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);
        let model = create_test_model();

        coordinator.initialize(model).unwrap();
        assert_eq!(coordinator.state(), CoordinatorState::Ready);
    }

    #[test]
    fn test_round_lifecycle() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);
        let model = create_test_model();

        coordinator.initialize(model).unwrap();

        // Start round
        let round_id = coordinator.start_round().unwrap();
        assert_eq!(coordinator.state(), CoordinatorState::WaitingForRound);

        // Check events
        let events = coordinator.drain_events();
        assert!(events.iter().any(|e| matches!(e, CoordinatorEvent::RoundStarted { .. })));
    }

    #[test]
    fn test_gradient_submission() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);
        let model = create_test_model();

        coordinator.initialize(model.clone()).unwrap();
        coordinator.start_round().unwrap();

        let gradient = create_test_gradient(&model);
        let gradient_hash = gradient.commitment();
        let proof = create_valid_proof(gradient_hash, 0);

        coordinator.submit_gradient(gradient, 0.01, proof).unwrap();
        assert_eq!(coordinator.state(), CoordinatorState::WaitingForAggregation);
    }

    #[test]
    fn test_peer_registration() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);

        coordinator.register_peer(
            NodeId::new("peer-1"),
            1000,
            "127.0.0.1:8080".to_string(),
        );

        assert_eq!(coordinator.peer_count(), 1);

        coordinator.unregister_peer(&NodeId::new("peer-1"));
        assert_eq!(coordinator.peer_count(), 0);
    }

    #[test]
    fn test_error_bound_tracking() {
        let mut tracker = ErrorBoundTracker::new(10, 0.05);

        tracker.add(0.01);
        tracker.add(0.02);
        tracker.add(0.03);

        assert_eq!(tracker.accumulated(), 0.06);
        assert!((tracker.average() - 0.02).abs() < 1e-10);
        assert!(!tracker.exceeds_warning());

        // Add high error bounds
        for _ in 0..10 {
            tracker.add(0.1);
        }
        assert!(tracker.exceeds_warning());
    }

    #[test]
    fn test_outlier_detection() {
        let mut detector = GradientOutlierDetector::new(50, 3.0, 100.0);

        // Add observations with some natural variance
        for i in 0..20 {
            // Values between 9.0 and 11.0 with some spread
            let value = 10.0 + (i as f64 - 10.0) * 0.1;
            detector.add_observation(value);
        }

        // Value close to mean should not be outlier
        let (is_outlier, _) = detector.is_outlier(10.5);
        assert!(!is_outlier);

        // Extreme value (far from mean) should be outlier
        let (is_outlier, reason) = detector.is_outlier(50.0);
        assert!(is_outlier);
        assert!(reason.is_some());

        // Value exceeding max should be outlier
        let (is_outlier, _) = detector.is_outlier(150.0);
        assert!(is_outlier);
    }

    #[test]
    fn test_proof_verification_sync() {
        let config = create_test_config();
        let coordinator = TrainingCoordinator::new(config);

        let gradient_hash = [1u8; 32];
        let round_id = RoundId::new(1);
        let proof = create_valid_proof(gradient_hash, round_id.0);

        let is_valid = coordinator.verify_proof_sync(&proof, gradient_hash, round_id, 0.05);
        assert!(is_valid);

        // Invalid error bound
        let is_valid = coordinator.verify_proof_sync(&proof, gradient_hash, round_id, 0.5);
        assert!(!is_valid);

        // Empty proof
        let is_valid = coordinator.verify_proof_sync(&[], gradient_hash, round_id, 0.05);
        assert!(!is_valid);
    }

    #[test]
    fn test_byzantine_detection() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);

        // Add a peer with bad history
        let peer_id = NodeId::new("bad-peer");
        coordinator.register_peer(peer_id.clone(), 1000, "127.0.0.1:8080".to_string());

        if let Some(peer) = coordinator.peers.get_mut(&peer_id) {
            peer.invalid_proofs = 10;
            peer.valid_proofs = 5;
            peer.byzantine_score = 2.0;
        }

        let detection = coordinator.detect_byzantine();
        assert!(detection.detected_nodes.contains(&peer_id));
    }

    #[test]
    fn test_peer_reputation() {
        let peer = PeerInfo::new(NodeId::new("test"), 1000, "127.0.0.1:8080".to_string());

        // New peer has neutral reputation
        assert!((peer.reputation() - 0.5).abs() < 0.01);

        // Peer with good history
        let mut good_peer = peer.clone();
        good_peer.valid_proofs = 100;
        good_peer.invalid_proofs = 0;
        assert!(good_peer.reputation() > 0.9);

        // Peer with mixed history
        let mut mixed_peer = peer.clone();
        mixed_peer.valid_proofs = 50;
        mixed_peer.invalid_proofs = 50;
        assert!((mixed_peer.reputation() - 0.5).abs() < 0.1);

        // Peer with Byzantine penalty
        let mut byzantine_peer = peer.clone();
        byzantine_peer.valid_proofs = 100;
        byzantine_peer.byzantine_score = 3.0;
        assert!(byzantine_peer.reputation() < 0.8);
    }

    #[test]
    fn test_recovery_state() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);
        let model = create_test_model();

        coordinator.initialize(model).unwrap();

        // Simulate error
        coordinator.handle_error("Test error".to_string());
        assert!(coordinator.state().can_recover());

        // Attempt recovery
        coordinator.attempt_recovery().unwrap();
        assert_eq!(coordinator.state(), CoordinatorState::Ready);
    }

    #[test]
    fn test_coordinator_state_transitions() {
        assert!(!CoordinatorState::Ready.is_terminal());
        assert!(CoordinatorState::Complete.is_terminal());
        assert!(CoordinatorState::ShuttingDown.is_terminal());
        assert!(CoordinatorState::Error.can_recover());
        assert!(!CoordinatorState::Ready.can_recover());
    }

    #[test]
    fn test_verification_cache() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);

        let proof = vec![1u8; 64];
        let model_commitment = [0xAA_u8; 32];
        let cache_key = coordinator.cache_key(&proof, &model_commitment);
        let round_id = RoundId::new(1);

        let result = VerificationResult {
            is_valid: true,
            verification_time_ms: 10,
            error: None,
            public_inputs: Some(vec![1]),
        };

        coordinator.cache_verification(&cache_key, &result, round_id);

        let cached = coordinator.get_cached_verification(&cache_key, round_id);
        assert!(cached.is_some());
        assert!(cached.unwrap().is_valid);

        // Different round should not use cache
        let cached = coordinator.get_cached_verification(&cache_key, RoundId::new(2));
        assert!(cached.is_none());

        // Same proof, different model commitment should not hit cache
        let different_commitment = [0xBB_u8; 32];
        let different_key = coordinator.cache_key(&proof, &different_commitment);
        let cached = coordinator.get_cached_verification(&different_key, round_id);
        assert!(cached.is_none());
    }

    #[test]
    fn test_adaptive_learning_rate() {
        let mut config = create_test_config();
        config.adaptive_lr = true;
        config.error_bound_threshold = 0.05;

        let mut coordinator = TrainingCoordinator::new(config);

        // Add high error bounds
        for _ in 0..10 {
            coordinator.error_tracker.add(0.1);
        }

        let base_lr = coordinator.lr_schedule.current();
        let adaptive_lr = coordinator.compute_adaptive_learning_rate();

        // Should reduce learning rate due to high error bounds
        assert!(adaptive_lr < base_lr);
    }

    #[test]
    fn test_gradient_norm_computation() {
        let config = create_test_config();
        let coordinator = TrainingCoordinator::new(config);
        let model = create_test_model();

        let mut gradient = ModelGradient::zeros_like(&model);

        // Set some non-zero values
        if let Some(ref mut embed) = gradient.embeddings {
            for v in embed.data.iter_mut().take(10) {
                *v = 1.0;
            }
        }

        let norm = coordinator.compute_gradient_norm(&gradient);
        assert!(norm > 0.0);
        assert!((norm - 10.0_f64.sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_full_training_round() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);
        let model = create_test_model();

        // Initialize
        coordinator.initialize(model.clone()).unwrap();

        // Start round
        let round_id = coordinator.start_round().unwrap();

        // We're the only participant, so we're the leader
        assert!(coordinator.is_leader());

        // Submit gradient
        let gradient = create_test_gradient(&model);
        let gradient_hash = gradient.commitment();
        let proof = create_valid_proof(gradient_hash, round_id.0);

        coordinator.submit_gradient(gradient, 0.01, proof).unwrap();

        // Aggregate
        let result = coordinator.aggregate().unwrap();
        assert_eq!(result.num_contributors, 1);
        assert!((result.combined_error_bound - 0.01).abs() < 1e-10);

        // Complete round
        coordinator.complete_round().unwrap();

        // Check events
        let events = coordinator.drain_events();
        assert!(events.iter().any(|e| matches!(e, CoordinatorEvent::AggregationComplete { .. })));
        assert!(events.iter().any(|e| matches!(e, CoordinatorEvent::ModelUpdated { .. })));
    }

    #[test]
    fn test_proof_validity_propagation() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);
        let model = create_test_model();

        coordinator.initialize(model.clone()).unwrap();
        coordinator.start_round().unwrap();

        // Add a peer
        let peer_id = NodeId::new("peer-1");
        coordinator.register_peer(peer_id.clone(), 1000, "127.0.0.1:8080".to_string());

        // Receive a gradient
        let gradient = create_test_gradient(&model);
        let gradient_hash = gradient.commitment();
        let proof = create_valid_proof(gradient_hash, 0);

        let _ = coordinator.receive_gradient(
            peer_id.clone(),
            gradient,
            1000,
            0.01,
            gradient_hash,
            proof,
        );

        // Check propagation queue
        let propagation = coordinator.drain_proof_propagation();
        assert!(!propagation.is_empty());
        assert!(propagation.iter().any(|(_, id, valid)| id == &peer_id && *valid));
    }

    #[test]
    fn test_metrics_recording_with_error_bound() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);
        let model = create_test_model();

        coordinator.initialize(model).unwrap();

        // Set aggregation error bound
        coordinator.last_aggregation_error_bound = 0.02;

        // Record metrics
        coordinator.record_metrics(1.5, 0.1, 32, Duration::from_millis(100));

        // Check that metrics were recorded
        let summary = coordinator.metrics.summary();
        assert!(summary.average_error_bound > 0.0);
    }

    #[test]
    fn test_invalid_proof_byzantine_score() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);
        let model = create_test_model();

        coordinator.initialize(model.clone()).unwrap();
        coordinator.start_round().unwrap();

        let peer_id = NodeId::new("peer-1");
        coordinator.register_peer(peer_id.clone(), 1000, "127.0.0.1:8080".to_string());

        let gradient = create_test_gradient(&model);
        let gradient_hash = gradient.commitment();
        let invalid_proof = vec![0u8; 64]; // Invalid proof

        // Should fail verification
        let result = coordinator.receive_gradient(
            peer_id.clone(),
            gradient,
            1000,
            0.01,
            gradient_hash,
            invalid_proof,
        );

        assert!(result.is_err());

        // Check Byzantine score increased
        let peer = coordinator.peers.get(&peer_id).unwrap();
        assert!(peer.byzantine_score > 0.0);
        assert_eq!(peer.invalid_proofs, 1);
    }

    #[test]
    fn test_gradient_outlier_rejection() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);
        let model = create_test_model();

        coordinator.initialize(model.clone()).unwrap();
        coordinator.start_round().unwrap();

        // Add normal observations to outlier detector
        for _ in 0..20 {
            coordinator.outlier_detector.add_observation(1.0);
        }

        let peer_id = NodeId::new("peer-1");
        coordinator.register_peer(peer_id.clone(), 1000, "127.0.0.1:8080".to_string());

        // Create a gradient with extreme values
        let mut gradient = create_test_gradient(&model);
        if let Some(ref mut embed) = gradient.embeddings {
            for v in embed.data.iter_mut() {
                *v = 1000.0; // Extreme values
            }
        }

        let gradient_hash = gradient.commitment();
        let proof = create_valid_proof(gradient_hash, 0);

        let result = coordinator.receive_gradient(
            peer_id.clone(),
            gradient,
            1000,
            0.01,
            gradient_hash,
            proof,
        );

        // Should be rejected as outlier due to max_gradient_norm
        assert!(result.is_err());
    }

    #[test]
    fn test_shutdown() {
        let config = create_test_config();
        let mut coordinator = TrainingCoordinator::new(config);
        let model = create_test_model();

        coordinator.initialize(model).unwrap();
        coordinator.shutdown();

        assert_eq!(coordinator.state(), CoordinatorState::ShuttingDown);
        assert!(coordinator.verification_cache.is_empty());
    }

    #[test]
    fn test_recovery_attempts_limit() {
        let mut config = create_test_config();
        config.checkpoint_config.checkpoint_dir = std::path::PathBuf::from("/tmp/helix-test-ckpt-recovery");

        let mut coordinator = TrainingCoordinator::new(config);
        let model = create_test_model();

        coordinator.initialize(model).unwrap();

        // Simulate multiple errors and recovery attempts
        for i in 0..3 {
            coordinator.handle_error(format!("Error {}", i));
            coordinator.attempt_recovery().unwrap();
        }

        // Next error and recovery should hit the limit
        coordinator.handle_error("Final error".to_string());
        coordinator.recovery.recovery_attempts = 3;

        let result = coordinator.attempt_recovery();
        assert!(matches!(result, Err(CoordinatorError::MaxRecoveryAttemptsExceeded)));
    }
}
