//! Distributed Worker Synchronization.
//!
//! Provides synchronization barriers for distributed training:
//! - Phase barriers (wait for all workers to reach a phase)
//! - Timeout handling with partial progress
//! - Graceful degradation when workers fail to synchronize
//! - Progress tracking and monitoring

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use tokio::sync::{broadcast, oneshot, Notify};
use tokio::time::timeout;

use crate::network::messages::PeerId;

/// Unique identifier for a synchronization barrier.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct BarrierId(pub u64);

impl BarrierId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for BarrierId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Barrier({})", self.0)
    }
}

/// Phase of training that workers synchronize on.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum SyncPhase {
    /// Workers joined and ready.
    Ready,
    /// Model shares distributed.
    SharesDistributed,
    /// Computation started.
    ComputationStarted,
    /// Gradients computed.
    GradientsComputed,
    /// Gradients submitted.
    GradientsSubmitted,
    /// Ready for aggregation.
    AggregationReady,
    /// Aggregation complete.
    AggregationComplete,
    /// Round committed on-chain.
    RoundCommitted,
    /// Custom phase.
    Custom(u32),
}

impl std::fmt::Display for SyncPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready => write!(f, "Ready"),
            Self::SharesDistributed => write!(f, "SharesDistributed"),
            Self::ComputationStarted => write!(f, "ComputationStarted"),
            Self::GradientsComputed => write!(f, "GradientsComputed"),
            Self::GradientsSubmitted => write!(f, "GradientsSubmitted"),
            Self::AggregationReady => write!(f, "AggregationReady"),
            Self::AggregationComplete => write!(f, "AggregationComplete"),
            Self::RoundCommitted => write!(f, "RoundCommitted"),
            Self::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

/// Result of waiting on a barrier.
#[derive(Debug, Clone)]
pub enum BarrierResult {
    /// All expected participants arrived.
    AllArrived {
        participants: Vec<PeerId>,
        duration: Duration,
    },
    /// Barrier released with partial participation.
    PartialParticipation {
        arrived: Vec<PeerId>,
        missing: Vec<PeerId>,
        duration: Duration,
    },
    /// Barrier timed out.
    Timeout {
        arrived: Vec<PeerId>,
        missing: Vec<PeerId>,
        duration: Duration,
    },
    /// Barrier was cancelled.
    Cancelled { reason: String },
}

impl BarrierResult {
    pub fn is_success(&self) -> bool {
        matches!(self, Self::AllArrived { .. })
    }

    pub fn is_partial(&self) -> bool {
        matches!(self, Self::PartialParticipation { .. })
    }

    pub fn arrived_count(&self) -> usize {
        match self {
            Self::AllArrived { participants, .. } => participants.len(),
            Self::PartialParticipation { arrived, .. } => arrived.len(),
            Self::Timeout { arrived, .. } => arrived.len(),
            Self::Cancelled { .. } => 0,
        }
    }
}

/// Configuration for a synchronization barrier.
#[derive(Debug, Clone)]
pub struct BarrierConfig {
    /// Timeout duration.
    pub timeout: Duration,
    /// Minimum participants required (for partial success).
    pub min_participants: usize,
    /// Whether to allow partial participation.
    pub allow_partial: bool,
    /// Minimum fraction of participants for partial success.
    pub min_fraction: f64,
    /// Whether to auto-release on minimum participants.
    pub auto_release_on_min: bool,
}

impl Default for BarrierConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            min_participants: 2,
            allow_partial: true,
            min_fraction: 0.67,
            auto_release_on_min: false,
        }
    }
}

/// State of a participant at a barrier.
#[derive(Debug, Clone)]
struct ParticipantState {
    peer_id: PeerId,
    arrived_at: Instant,
    data: Option<Vec<u8>>,
}

/// Internal state of a barrier.
struct BarrierState {
    id: BarrierId,
    phase: SyncPhase,
    config: BarrierConfig,
    expected: HashSet<PeerId>,
    arrived: HashMap<PeerId, ParticipantState>,
    created_at: Instant,
    released: bool,
    result: Option<BarrierResult>,
}

impl BarrierState {
    fn new(
        id: BarrierId,
        phase: SyncPhase,
        config: BarrierConfig,
        expected: HashSet<PeerId>,
    ) -> Self {
        Self {
            id,
            phase,
            config,
            expected,
            arrived: HashMap::new(),
            created_at: Instant::now(),
            released: false,
            result: None,
        }
    }

    fn is_complete(&self) -> bool {
        self.arrived.len() >= self.expected.len()
    }

    fn has_minimum(&self) -> bool {
        let count = self.arrived.len();
        count >= self.config.min_participants
            && (count as f64 / self.expected.len().max(1) as f64) >= self.config.min_fraction
    }

    fn missing(&self) -> Vec<PeerId> {
        self.expected
            .iter()
            .filter(|p| !self.arrived.contains_key(*p))
            .cloned()
            .collect()
    }
}

/// A synchronization barrier for distributed workers.
pub struct SyncBarrier {
    state: Arc<RwLock<BarrierState>>,
    notify: Arc<Notify>,
    cancel_tx: Option<oneshot::Sender<String>>,
}

impl SyncBarrier {
    /// Creates a new barrier.
    fn new(
        id: BarrierId,
        phase: SyncPhase,
        config: BarrierConfig,
        expected: HashSet<PeerId>,
    ) -> (Self, oneshot::Receiver<String>) {
        let (cancel_tx, cancel_rx) = oneshot::channel();
        let barrier = Self {
            state: Arc::new(RwLock::new(BarrierState::new(id, phase, config, expected))),
            notify: Arc::new(Notify::new()),
            cancel_tx: Some(cancel_tx),
        };
        (barrier, cancel_rx)
    }

    /// Records a participant arriving at the barrier.
    pub fn arrive(&self, peer_id: &PeerId, data: Option<Vec<u8>>) -> Result<usize, String> {
        let mut state = self.state.write();

        if state.released {
            return Err("Barrier already released".to_string());
        }

        if !state.expected.contains(peer_id) {
            return Err(format!("Peer {} not expected at barrier", peer_id));
        }

        if state.arrived.contains_key(peer_id) {
            return Err(format!("Peer {} already arrived", peer_id));
        }

        state.arrived.insert(
            peer_id.clone(),
            ParticipantState {
                peer_id: peer_id.clone(),
                arrived_at: Instant::now(),
                data,
            },
        );

        let count = state.arrived.len();

        // Check if we should release
        let should_release = state.is_complete()
            || (state.config.auto_release_on_min && state.has_minimum());

        drop(state);

        if should_release {
            self.notify.notify_waiters();
        }

        Ok(count)
    }

    /// Removes a participant from the expected set.
    pub fn remove_expected(&self, peer_id: &PeerId) {
        let mut state = self.state.write();
        state.expected.remove(peer_id);
        state.arrived.remove(peer_id);

        if state.is_complete() || state.has_minimum() {
            drop(state);
            self.notify.notify_waiters();
        }
    }

    /// Waits for the barrier to be released.
    pub async fn wait(&self) -> BarrierResult {
        let timeout_duration = {
            let state = self.state.read();
            if state.released {
                return state.result.clone().unwrap_or(BarrierResult::Cancelled {
                    reason: "Already released".to_string(),
                });
            }
            state.config.timeout
        };

        // Wait with timeout
        let result = timeout(timeout_duration, async {
            loop {
                {
                    let state = self.state.read();
                    if state.released {
                        return state.result.clone().unwrap_or(BarrierResult::Cancelled {
                            reason: "Released".to_string(),
                        });
                    }
                    if state.is_complete() {
                        let arrived: Vec<PeerId> = state.arrived.keys().cloned().collect();
                        let duration = state.created_at.elapsed();
                        return BarrierResult::AllArrived {
                            participants: arrived,
                            duration,
                        };
                    }
                }

                self.notify.notified().await;
            }
        })
        .await;

        match result {
            Ok(barrier_result) => {
                // Store result
                let mut state = self.state.write();
                state.released = true;
                state.result = Some(barrier_result.clone());
                barrier_result
            }
            Err(_) => {
                // Timeout - check if partial success
                let mut state = self.state.write();
                state.released = true;

                let arrived: Vec<PeerId> = state.arrived.keys().cloned().collect();
                let missing = state.missing();
                let duration = state.created_at.elapsed();

                let result = if state.config.allow_partial && state.has_minimum() {
                    BarrierResult::PartialParticipation {
                        arrived,
                        missing,
                        duration,
                    }
                } else {
                    BarrierResult::Timeout {
                        arrived,
                        missing,
                        duration,
                    }
                };

                state.result = Some(result.clone());
                result
            }
        }
    }

    /// Cancels the barrier.
    pub fn cancel(mut self, reason: &str) {
        if let Some(tx) = self.cancel_tx.take() {
            let _ = tx.send(reason.to_string());
        }

        let mut state = self.state.write();
        state.released = true;
        state.result = Some(BarrierResult::Cancelled {
            reason: reason.to_string(),
        });

        drop(state);
        self.notify.notify_waiters();
    }

    /// Returns the current progress.
    pub fn progress(&self) -> (usize, usize) {
        let state = self.state.read();
        (state.arrived.len(), state.expected.len())
    }

    /// Returns the barrier ID.
    pub fn id(&self) -> BarrierId {
        self.state.read().id
    }

    /// Returns the phase.
    pub fn phase(&self) -> SyncPhase {
        self.state.read().phase
    }

    /// Returns whether the barrier is released.
    pub fn is_released(&self) -> bool {
        self.state.read().released
    }

    /// Returns the arrived participants.
    pub fn arrived(&self) -> Vec<PeerId> {
        self.state.read().arrived.keys().cloned().collect()
    }

    /// Returns the missing participants.
    pub fn missing(&self) -> Vec<PeerId> {
        self.state.read().missing()
    }
}

/// Event from the barrier manager.
#[derive(Debug, Clone)]
pub enum BarrierEvent {
    /// Barrier created.
    Created {
        barrier_id: BarrierId,
        phase: SyncPhase,
        expected: usize,
    },
    /// Participant arrived.
    ParticipantArrived {
        barrier_id: BarrierId,
        peer_id: PeerId,
        arrived: usize,
        expected: usize,
    },
    /// Barrier released.
    Released {
        barrier_id: BarrierId,
        result: BarrierResult,
    },
    /// Barrier cancelled.
    Cancelled {
        barrier_id: BarrierId,
        reason: String,
    },
}

/// Manages multiple synchronization barriers.
pub struct BarrierManager {
    /// Active barriers by ID.
    barriers: Arc<RwLock<HashMap<BarrierId, Arc<SyncBarrier>>>>,
    /// Barriers by phase.
    phase_barriers: Arc<RwLock<HashMap<SyncPhase, BarrierId>>>,
    /// Next barrier ID.
    next_id: AtomicU64,
    /// Event sender.
    event_tx: broadcast::Sender<BarrierEvent>,
    /// Default configuration.
    default_config: BarrierConfig,
}

impl BarrierManager {
    /// Creates a new barrier manager.
    pub fn new(default_config: BarrierConfig) -> Self {
        let (event_tx, _) = broadcast::channel(1000);
        Self {
            barriers: Arc::new(RwLock::new(HashMap::new())),
            phase_barriers: Arc::new(RwLock::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            event_tx,
            default_config,
        }
    }

    /// Subscribes to barrier events.
    pub fn subscribe(&self) -> broadcast::Receiver<BarrierEvent> {
        self.event_tx.subscribe()
    }

    /// Creates a new barrier for a phase.
    pub fn create_barrier(
        &self,
        phase: SyncPhase,
        expected: HashSet<PeerId>,
        config: Option<BarrierConfig>,
    ) -> Arc<SyncBarrier> {
        let id = BarrierId::new(self.next_id.fetch_add(1, Ordering::SeqCst));
        let config = config.unwrap_or_else(|| self.default_config.clone());
        let expected_count = expected.len();

        let (barrier, _cancel_rx) = SyncBarrier::new(id, phase, config, expected);
        let barrier = Arc::new(barrier);

        {
            let mut barriers = self.barriers.write();
            barriers.insert(id, Arc::clone(&barrier));
        }

        {
            let mut phase_barriers = self.phase_barriers.write();
            phase_barriers.insert(phase, id);
        }

        let _ = self.event_tx.send(BarrierEvent::Created {
            barrier_id: id,
            phase,
            expected: expected_count,
        });

        barrier
    }

    /// Gets a barrier by ID.
    pub fn get_barrier(&self, id: BarrierId) -> Option<Arc<SyncBarrier>> {
        self.barriers.read().get(&id).cloned()
    }

    /// Gets the barrier for a phase.
    pub fn get_phase_barrier(&self, phase: SyncPhase) -> Option<Arc<SyncBarrier>> {
        let phase_barriers = self.phase_barriers.read();
        if let Some(&id) = phase_barriers.get(&phase) {
            self.barriers.read().get(&id).cloned()
        } else {
            None
        }
    }

    /// Records a participant arriving at a phase barrier.
    pub fn arrive_at_phase(
        &self,
        phase: SyncPhase,
        peer_id: &PeerId,
        data: Option<Vec<u8>>,
    ) -> Result<usize, String> {
        let barrier = self
            .get_phase_barrier(phase)
            .ok_or_else(|| format!("No barrier for phase {:?}", phase))?;

        let count = barrier.arrive(peer_id, data)?;
        let (arrived, expected) = barrier.progress();

        let _ = self.event_tx.send(BarrierEvent::ParticipantArrived {
            barrier_id: barrier.id(),
            peer_id: peer_id.clone(),
            arrived,
            expected,
        });

        Ok(count)
    }

    /// Waits for a phase barrier.
    pub async fn wait_for_phase(&self, phase: SyncPhase) -> Result<BarrierResult, String> {
        let barrier = self
            .get_phase_barrier(phase)
            .ok_or_else(|| format!("No barrier for phase {:?}", phase))?;

        let result = barrier.wait().await;

        let _ = self.event_tx.send(BarrierEvent::Released {
            barrier_id: barrier.id(),
            result: result.clone(),
        });

        Ok(result)
    }

    /// Removes a participant from all active barriers.
    pub fn remove_participant(&self, peer_id: &PeerId) {
        let barriers = self.barriers.read();
        for barrier in barriers.values() {
            if !barrier.is_released() {
                barrier.remove_expected(peer_id);
            }
        }
    }

    /// Cancels a barrier.
    pub fn cancel_barrier(&self, id: BarrierId, reason: &str) -> Result<(), String> {
        let barrier = {
            let mut barriers = self.barriers.write();
            barriers.remove(&id)
        };

        if let Some(barrier) = barrier {
            let phase = barrier.phase();
            {
                let mut phase_barriers = self.phase_barriers.write();
                phase_barriers.remove(&phase);
            }

            // Note: barrier.cancel() consumes the barrier, but we have Arc
            // Instead, just mark it as cancelled
            let _ = self.event_tx.send(BarrierEvent::Cancelled {
                barrier_id: id,
                reason: reason.to_string(),
            });

            Ok(())
        } else {
            Err(format!("Barrier {} not found", id))
        }
    }

    /// Cancels all barriers for a phase.
    pub fn cancel_phase(&self, phase: SyncPhase, reason: &str) {
        let id = {
            let phase_barriers = self.phase_barriers.read();
            phase_barriers.get(&phase).copied()
        };

        if let Some(id) = id {
            let _ = self.cancel_barrier(id, reason);
        }
    }

    /// Cleans up released barriers.
    pub fn cleanup(&self) {
        let mut barriers = self.barriers.write();
        let mut phase_barriers = self.phase_barriers.write();

        let released: Vec<BarrierId> = barriers
            .iter()
            .filter(|(_, b)| b.is_released())
            .map(|(id, _)| *id)
            .collect();

        for id in released {
            if let Some(barrier) = barriers.remove(&id) {
                phase_barriers.remove(&barrier.phase());
            }
        }
    }

    /// Returns the count of active barriers.
    pub fn active_count(&self) -> usize {
        self.barriers
            .read()
            .values()
            .filter(|b| !b.is_released())
            .count()
    }
}

/// Synchronization coordinator for distributed training.
pub struct SyncCoordinator {
    /// Barrier manager.
    barrier_manager: BarrierManager,
    /// Active workers.
    workers: Arc<RwLock<HashSet<PeerId>>>,
    /// Current round ID.
    current_round: Arc<RwLock<Option<u64>>>,
}

impl SyncCoordinator {
    /// Creates a new sync coordinator.
    pub fn new(config: BarrierConfig) -> Self {
        Self {
            barrier_manager: BarrierManager::new(config),
            workers: Arc::new(RwLock::new(HashSet::new())),
            current_round: Arc::new(RwLock::new(None)),
        }
    }

    /// Starts a new round.
    pub fn start_round(&self, round_id: u64) {
        *self.current_round.write() = Some(round_id);
    }

    /// Ends the current round.
    pub fn end_round(&self) {
        *self.current_round.write() = None;
        self.barrier_manager.cleanup();
    }

    /// Registers a worker.
    pub fn register_worker(&self, peer_id: PeerId) {
        self.workers.write().insert(peer_id);
    }

    /// Unregisters a worker.
    pub fn unregister_worker(&self, peer_id: &PeerId) {
        self.workers.write().remove(peer_id);
        self.barrier_manager.remove_participant(peer_id);
    }

    /// Creates a phase barrier for all current workers.
    pub fn create_phase_barrier(&self, phase: SyncPhase, config: Option<BarrierConfig>) -> Arc<SyncBarrier> {
        let workers = self.workers.read().clone();
        self.barrier_manager.create_barrier(phase, workers, config)
    }

    /// Creates a phase barrier for specific workers.
    pub fn create_barrier_for(
        &self,
        phase: SyncPhase,
        workers: HashSet<PeerId>,
        config: Option<BarrierConfig>,
    ) -> Arc<SyncBarrier> {
        self.barrier_manager.create_barrier(phase, workers, config)
    }

    /// Records a worker arriving at a phase.
    pub fn arrive(&self, phase: SyncPhase, peer_id: &PeerId) -> Result<usize, String> {
        self.barrier_manager.arrive_at_phase(phase, peer_id, None)
    }

    /// Records a worker arriving with data.
    pub fn arrive_with_data(
        &self,
        phase: SyncPhase,
        peer_id: &PeerId,
        data: Vec<u8>,
    ) -> Result<usize, String> {
        self.barrier_manager
            .arrive_at_phase(phase, peer_id, Some(data))
    }

    /// Waits for a phase barrier.
    pub async fn wait_for(&self, phase: SyncPhase) -> Result<BarrierResult, String> {
        self.barrier_manager.wait_for_phase(phase).await
    }

    /// Subscribes to barrier events.
    pub fn subscribe(&self) -> broadcast::Receiver<BarrierEvent> {
        self.barrier_manager.subscribe()
    }

    /// Returns the barrier manager.
    pub fn barrier_manager(&self) -> &BarrierManager {
        &self.barrier_manager
    }

    /// Returns the current workers.
    pub fn workers(&self) -> HashSet<PeerId> {
        self.workers.read().clone()
    }

    /// Returns the worker count.
    pub fn worker_count(&self) -> usize {
        self.workers.read().len()
    }

    /// Creates a barrier with the specified workers and timeout (convenience method).
    pub fn create_barrier(
        &self,
        phase: SyncPhase,
        workers: Vec<PeerId>,
        timeout: Duration,
    ) -> Arc<SyncBarrier> {
        let config = BarrierConfig {
            timeout,
            ..Default::default()
        };
        let workers_set: HashSet<PeerId> = workers.into_iter().collect();
        self.barrier_manager.create_barrier(phase, workers_set, Some(config))
    }

    /// Arrives at a barrier and waits for all participants.
    pub async fn arrive_and_wait(
        &self,
        phase: SyncPhase,
        peer_id: PeerId,
        timeout: Duration,
    ) -> BarrierResult {
        // Arrive at the barrier
        let _ = self.barrier_manager.arrive_at_phase(phase, &peer_id, None);

        // Wait for the barrier with timeout
        match tokio::time::timeout(timeout, self.barrier_manager.wait_for_phase(phase)).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => BarrierResult::Cancelled {
                reason: "Phase barrier not found".to_string(),
            },
            Err(_) => BarrierResult::Timeout {
                arrived: Vec::new(),
                missing: Vec::new(),
                duration: timeout,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_peer_id(idx: usize) -> PeerId {
        PeerId::from_string(&format!("test-peer-{}", idx))
    }

    #[tokio::test]
    async fn test_barrier_all_arrive() {
        let expected: HashSet<PeerId> = (0..3).map(create_peer_id).collect();
        let config = BarrierConfig {
            timeout: Duration::from_secs(5),
            ..Default::default()
        };

        let (barrier, _) = SyncBarrier::new(
            BarrierId::new(1),
            SyncPhase::Ready,
            config,
            expected.clone(),
        );
        let barrier = Arc::new(barrier);

        // Spawn wait task
        let barrier_clone = Arc::clone(&barrier);
        let wait_task = tokio::spawn(async move { barrier_clone.wait().await });

        // Arrive
        for i in 0..3 {
            let count = barrier.arrive(&create_peer_id(i), None).unwrap();
            assert_eq!(count, i + 1);
        }

        let result = wait_task.await.unwrap();
        assert!(result.is_success());
        assert_eq!(result.arrived_count(), 3);
    }

    #[tokio::test]
    async fn test_barrier_timeout_partial() {
        let expected: HashSet<PeerId> = (0..3).map(create_peer_id).collect();
        let config = BarrierConfig {
            timeout: Duration::from_millis(100),
            min_participants: 2,
            allow_partial: true,
            min_fraction: 0.5,
            ..Default::default()
        };

        let (barrier, _) = SyncBarrier::new(
            BarrierId::new(1),
            SyncPhase::Ready,
            config,
            expected,
        );
        let barrier = Arc::new(barrier);

        // Only 2 arrive
        barrier.arrive(&create_peer_id(0), None).unwrap();
        barrier.arrive(&create_peer_id(1), None).unwrap();

        let result = barrier.wait().await;
        assert!(result.is_partial());
        assert_eq!(result.arrived_count(), 2);
    }

    #[tokio::test]
    async fn test_barrier_manager() {
        let config = BarrierConfig::default();
        let manager = BarrierManager::new(config);

        let expected: HashSet<PeerId> = (0..2).map(create_peer_id).collect();
        let barrier = manager.create_barrier(SyncPhase::Ready, expected.clone(), None);

        assert_eq!(barrier.id().0, 1);
        assert_eq!(manager.active_count(), 1);

        // Arrive via manager
        manager
            .arrive_at_phase(SyncPhase::Ready, &create_peer_id(0), None)
            .unwrap();
        manager
            .arrive_at_phase(SyncPhase::Ready, &create_peer_id(1), None)
            .unwrap();

        let (arrived, expected) = barrier.progress();
        assert_eq!(arrived, 2);
        assert_eq!(expected, 2);
    }

    #[tokio::test]
    async fn test_sync_coordinator() {
        let config = BarrierConfig {
            timeout: Duration::from_secs(5),
            min_participants: 2,
            ..Default::default()
        };
        let coordinator = SyncCoordinator::new(config);

        // Register workers
        coordinator.register_worker(create_peer_id(0));
        coordinator.register_worker(create_peer_id(1));

        assert_eq!(coordinator.worker_count(), 2);

        // Create barrier
        let barrier = coordinator.create_phase_barrier(SyncPhase::Ready, None);

        // Arrive
        coordinator.arrive(SyncPhase::Ready, &create_peer_id(0)).unwrap();
        coordinator.arrive(SyncPhase::Ready, &create_peer_id(1)).unwrap();

        let result = coordinator.wait_for(SyncPhase::Ready).await.unwrap();
        assert!(result.is_success());
    }

    #[test]
    fn test_barrier_remove_expected() {
        let expected: HashSet<PeerId> = (0..3).map(create_peer_id).collect();
        let config = BarrierConfig::default();

        let (barrier, _) = SyncBarrier::new(
            BarrierId::new(1),
            SyncPhase::Ready,
            config,
            expected,
        );

        barrier.arrive(&create_peer_id(0), None).unwrap();
        barrier.arrive(&create_peer_id(1), None).unwrap();

        // Remove expected peer
        barrier.remove_expected(&create_peer_id(2));

        let (arrived, expected) = barrier.progress();
        assert_eq!(arrived, 2);
        assert_eq!(expected, 2);
    }
}
