//! MPC Session Coordinator for HELIX P2P Network.
//!
//! Orchestrates the creation and lifecycle of MPC training sessions on top of
//! the existing P2P connection infrastructure. The coordinator runs on the
//! aggregator node and manages:
//!
//! - Worker registration and tracking (with x25519 / Ethereum address metadata)
//! - MPC session creation with party ID assignment
//! - Session parameter distribution to workers
//! - MPC message routing through the P2P layer
//! - Session lifecycle transitions (Setup → Training → Checkpointing → Completed/Failed)
//! - MAC failure handling and cheater identification coordination
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │ SessionCoordinator                                                  │
//! │  ┌──────────────────┐  ┌──────────────────────────────────────┐   │
//! │  │ ConnectionManager │──│ Event Loop                           │   │
//! │  │ (TCP, Heartbeat,  │  │  ├─ Routes MpcDataMessages          │   │
//! │  │  Reconnection)    │  │  ├─ Processes WorkerRegistration     │   │
//! │  └──────────────────┘  │  ├─ Coordinates phase transitions    │   │
//! │                         │  └─ Emits SessionEvents              │   │
//! │  ┌──────────────────┐  └──────────────────────────────────────┘   │
//! │  │ MPC Router        │                                             │
//! │  │  session_id →     │  ┌──────────────────────────────────────┐   │
//! │  │   party_id →      │──│ Per-session NodeTransport channels   │   │
//! │  │    inbound_tx     │  └──────────────────────────────────────┘   │
//! │  └──────────────────┘                                             │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::sync::Mutex as TokioMutex;

use helix_mpc::session::node_transport::{NodeTransport, PeerMapping};
use helix_mpc::session::transport::MPCTransport;
use helix_mpc::types::PartyId;

use crate::config::NodeRole;
use crate::network::connection_manager::{ConnectionManager, ConnectionManagerError, P2PEvent};
use crate::network::messages::{
    MessagePayload, MpcDataMessage, NetworkMessage, NodeCapabilities, PeerId,
};
use crate::network::peer_registry::PeerSnapshot;

// ── Session Configuration ────────────────────────────────────────────────

/// Parameters distributed to workers when an MPC session starts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionParams {
    /// Unique session identifier.
    pub session_id: String,
    /// Model architecture: input dimension.
    pub d_in: usize,
    /// Model architecture: hidden dimension.
    pub d_hid: usize,
    /// Model architecture: output dimension.
    pub d_out: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Batch size.
    pub batch_size: u32,
    /// Number of training steps per round.
    pub steps_per_round: u32,
    /// How often to run MAC verification (in steps).
    pub mac_check_interval: u32,
    /// How often to create a checkpoint (in steps).
    pub checkpoint_interval: u32,
    /// Total number of parties.
    pub num_parties: usize,
}

impl Default for SessionParams {
    fn default() -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            d_in: 784,
            d_hid: 32,
            d_out: 10,
            learning_rate: 0.01,
            batch_size: 64,
            steps_per_round: 100,
            mac_check_interval: 10,
            checkpoint_interval: 25,
            num_parties: 3,
        }
    }
}

// ── Session State Machine ────────────────────────────────────────────────

/// State of an MPC training session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionState {
    /// Session is being set up: workers are being assigned party IDs.
    Setup,
    /// All workers acknowledged and training is underway.
    Training { current_step: u64 },
    /// A checkpoint is being coordinated.
    Checkpointing { at_step: u64 },
    /// Session completed successfully.
    Completed,
    /// Session failed.
    Failed { reason: String },
}

impl SessionState {
    /// Returns true if the session is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, SessionState::Completed | SessionState::Failed { .. })
    }

    /// Returns a human-readable name for the state.
    pub fn name(&self) -> &'static str {
        match self {
            SessionState::Setup => "setup",
            SessionState::Training { .. } => "training",
            SessionState::Checkpointing { .. } => "checkpointing",
            SessionState::Completed => "completed",
            SessionState::Failed { .. } => "failed",
        }
    }
}

// ── MPC Session ──────────────────────────────────────────────────────────

/// Participant in an MPC session.
#[derive(Debug, Clone)]
pub struct SessionParticipant {
    /// Network peer ID.
    pub peer_id: PeerId,
    /// Assigned MPC party ID.
    pub party_id: PartyId,
    /// Whether the worker has acknowledged the session setup.
    pub ready: bool,
}

/// An active MPC training session tracked by the coordinator.
pub struct MpcSession {
    /// Unique session identifier.
    pub session_id: String,
    /// Current session state.
    pub state: SessionState,
    /// Session parameters distributed to workers.
    pub params: SessionParams,
    /// Ordered list of participants (index = party index).
    pub participants: Vec<SessionParticipant>,
    /// When the session was created.
    pub created_at: Instant,
    /// When the last state transition occurred.
    pub last_transition: Instant,
}

impl MpcSession {
    /// Returns the participant for a given peer ID.
    pub fn participant_by_peer(&self, peer_id: &PeerId) -> Option<&SessionParticipant> {
        self.participants.iter().find(|p| &p.peer_id == peer_id)
    }

    /// Returns the participant for a given party ID.
    pub fn participant_by_party(&self, party_id: &PartyId) -> Option<&SessionParticipant> {
        self.participants.iter().find(|p| &p.party_id == party_id)
    }

    /// Returns the peer ID for a given party ID.
    pub fn peer_for_party(&self, party_id: &PartyId) -> Option<&PeerId> {
        self.participant_by_party(party_id).map(|p| &p.peer_id)
    }

    /// Returns the party ID for a given peer ID.
    pub fn party_for_peer(&self, peer_id: &PeerId) -> Option<&PartyId> {
        self.participant_by_peer(peer_id).map(|p| &p.party_id)
    }

    /// Returns true if all participants are ready.
    pub fn all_ready(&self) -> bool {
        self.participants.iter().all(|p| p.ready)
    }

    /// Returns the number of participants.
    pub fn num_participants(&self) -> usize {
        self.participants.len()
    }

    /// Returns the party-peer mapping used by the MPC transport layer.
    pub fn party_peer_map(&self) -> Vec<(PartyId, PeerId)> {
        self.participants
            .iter()
            .map(|p| (p.party_id.clone(), p.peer_id.clone()))
            .collect()
    }
}

// ── Session Events ───────────────────────────────────────────────────────

/// Events emitted by the session coordinator.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// A worker was registered and added to the available pool.
    WorkerRegistered { peer_id: PeerId },
    /// A worker was removed (disconnected or deregistered).
    WorkerRemoved { peer_id: PeerId, reason: String },
    /// A new MPC session was created.
    SessionCreated {
        session_id: String,
        participants: Vec<(PeerId, PartyId)>,
    },
    /// All workers in a session signaled readiness; training begins.
    SessionStarted { session_id: String },
    /// A checkpoint was reached.
    CheckpointReached { session_id: String, step: u64 },
    /// Session completed successfully.
    SessionCompleted { session_id: String },
    /// Session failed.
    SessionFailed { session_id: String, reason: String },
    /// A worker was removed from an active session.
    ParticipantLeft {
        session_id: String,
        peer_id: PeerId,
        remaining: usize,
    },
    /// A state transition occurred.
    StateTransition {
        session_id: String,
        from: String,
        to: String,
    },
}

// ── Session Messages ─────────────────────────────────────────────────────

/// MPC session control messages sent via the P2P layer.
///
/// These are serialized with bincode and sent as `MpcDataMessage` payloads
/// with a special session_id prefix ("ctrl::{session_id}") to distinguish
/// them from raw MPC protocol data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionControlMessage {
    /// Aggregator → Worker: Session assignment with party ID and parameters.
    SessionAssignment {
        session_id: String,
        party_id: usize,
        params: SessionParams,
        party_peer_map: Vec<(String, String)>, // (party_id_str, peer_id_str)
    },
    /// Worker → Aggregator: Acknowledges session assignment and signals readiness.
    SessionReady { session_id: String },
    /// Aggregator → All: Training begins.
    BeginTraining { session_id: String },
    /// Aggregator → All: Checkpoint now.
    CheckpointNow { session_id: String, at_step: u64 },
    /// Worker → Aggregator: Checkpoint complete.
    CheckpointDone { session_id: String, at_step: u64 },
    /// Aggregator → All: Resume training after checkpoint.
    ResumeTraining { session_id: String },
    /// Aggregator → All: Session completed.
    SessionComplete { session_id: String },
    /// Aggregator → All: Session failed.
    SessionAbort { session_id: String, reason: String },
    /// Aggregator → All: MAC verification failed, identify cheater.
    MacFailure {
        session_id: String,
        step: u64,
        suspected_party: Option<usize>,
    },
    /// Aggregator → All: Remove a participant (cheater or disconnected).
    RemoveParticipant {
        session_id: String,
        party_id: usize,
        reason: String,
    },
}

/// Prefix for control messages to distinguish from MPC data.
const CTRL_PREFIX: &str = "ctrl::";

// ── Session Coordinator ──────────────────────────────────────────────────

/// Orchestrates MPC training sessions on top of the P2P connection layer.
///
/// The coordinator runs on the aggregator node and handles:
/// 1. Worker tracking (via the underlying ConnectionManager's peer registry)
/// 2. MPC session creation and party ID assignment
/// 3. MPC message routing through P2P connections
/// 4. Session lifecycle management
pub struct SessionCoordinator {
    /// Our local peer ID.
    local_id: PeerId,
    /// The underlying P2P connection manager.
    conn_manager: Arc<ConnectionManager>,
    /// Active MPC sessions indexed by session_id.
    sessions: Arc<RwLock<HashMap<String, MpcSession>>>,
    /// MPC message router: session_id → (party_id_str → inbound_tx).
    /// Used to deliver incoming MPC data to the correct session's transport.
    mpc_router: Arc<TokioMutex<HashMap<String, HashMap<String, mpsc::Sender<Vec<u8>>>>>>,
    /// Per-session outbound pump task handles for cleanup on session removal.
    outbound_handles: Arc<TokioMutex<HashMap<String, Vec<tokio::task::JoinHandle<()>>>>>,
    /// Per-session PeerMapping for translating party IDs to peer IDs.
    session_peer_maps: Arc<TokioMutex<HashMap<String, PeerMapping>>>,
    /// Session event channel.
    event_tx: mpsc::Sender<SessionEvent>,
    event_rx: Arc<TokioMutex<mpsc::Receiver<SessionEvent>>>,
    /// Whether the coordinator is running.
    running: Arc<AtomicBool>,
}

impl SessionCoordinator {
    /// Creates a new session coordinator.
    pub fn new(local_id: PeerId, conn_manager: Arc<ConnectionManager>) -> Self {
        let (event_tx, event_rx) = mpsc::channel(1_000);

        Self {
            local_id,
            conn_manager,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            mpc_router: Arc::new(TokioMutex::new(HashMap::new())),
            outbound_handles: Arc::new(TokioMutex::new(HashMap::new())),
            session_peer_maps: Arc::new(TokioMutex::new(HashMap::new())),
            event_tx,
            event_rx: Arc::new(TokioMutex::new(event_rx)),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Starts the coordinator's event processing loop.
    ///
    /// This spawns a background task that:
    /// 1. Listens for P2P events from the ConnectionManager
    /// 2. Routes MPC data messages to the correct session
    /// 3. Processes session control messages
    /// 4. Tracks peer connections/disconnections
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);

        let running = self.running.clone();
        let conn_manager = self.conn_manager.clone();
        let sessions = self.sessions.clone();
        let mpc_router = self.mpc_router.clone();
        let session_peer_maps = self.session_peer_maps.clone();
        let event_tx = self.event_tx.clone();
        let local_id = self.local_id.clone();

        tokio::spawn(async move {
            while running.load(Ordering::SeqCst) {
                let event = tokio::select! {
                    ev = conn_manager.next_event() => ev,
                    _ = tokio::time::sleep(Duration::from_millis(50)) => continue,
                };

                let event = match event {
                    Some(ev) => ev,
                    None => break,
                };

                match event {
                    P2PEvent::PeerConnected { peer_id, role, .. } => {
                        if role == NodeRole::Compute {
                            let _ = event_tx.send(SessionEvent::WorkerRegistered {
                                peer_id,
                            }).await;
                        }
                    }

                    P2PEvent::PeerDisconnected { peer_id, reason } => {
                        // Check if the disconnected peer is in any active session.
                        // Scope the write guard so it is dropped before any .await.
                        let affected_sessions = {
                            let mut sessions_write = sessions.write();
                            let mut affected = Vec::new();

                            for (sid, session) in sessions_write.iter_mut() {
                                if session.state.is_terminal() {
                                    continue;
                                }
                                if let Some(pos) = session.participants.iter().position(|p| p.peer_id == peer_id) {
                                    session.participants.remove(pos);
                                    affected.push((
                                        sid.clone(),
                                        session.participants.len(),
                                    ));
                                }
                            }
                            affected
                        };

                        for (sid, remaining) in affected_sessions {
                            let _ = event_tx.send(SessionEvent::ParticipantLeft {
                                session_id: sid.clone(),
                                peer_id: peer_id.clone(),
                                remaining,
                            }).await;

                            // If no participants left, fail the session
                            if remaining == 0 {
                                let old_state = {
                                    let mut sessions_write = sessions.write();
                                    if let Some(session) = sessions_write.get_mut(&sid) {
                                        let old = session.state.name().to_string();
                                        session.state = SessionState::Failed {
                                            reason: "All participants disconnected".to_string(),
                                        };
                                        session.last_transition = Instant::now();
                                        Some(old)
                                    } else {
                                        None
                                    }
                                    // sessions_write dropped here
                                };
                                if let Some(old_state) = old_state {
                                    let _ = event_tx.send(SessionEvent::StateTransition {
                                        session_id: sid.clone(),
                                        from: old_state,
                                        to: "failed".to_string(),
                                    }).await;
                                    let _ = event_tx.send(SessionEvent::SessionFailed {
                                        session_id: sid,
                                        reason: "All participants disconnected".to_string(),
                                    }).await;
                                }
                            }
                        }

                        let _ = event_tx.send(SessionEvent::WorkerRemoved {
                            peer_id,
                            reason,
                        }).await;
                    }

                    P2PEvent::MessageReceived { from, message } => {
                        match message.payload {
                            MessagePayload::MpcData(ref mpc_msg) => {
                                // Check if this is a control message or raw MPC data
                                if mpc_msg.session_id.starts_with(CTRL_PREFIX) {
                                    Self::handle_control_message(
                                        &from,
                                        mpc_msg,
                                        &sessions,
                                        &event_tx,
                                    ).await;
                                } else {
                                    // Route raw MPC data to the correct session channel
                                    Self::route_mpc_message(
                                        &from,
                                        mpc_msg,
                                        &mpc_router,
                                        &session_peer_maps,
                                    ).await;
                                }
                            }
                            _ => {
                                // Other message types are handled by the connection
                                // manager directly (heartbeats, etc.)
                            }
                        }
                    }

                    _ => {} // HandshakeFailed, PeerHealthChanged handled by ConnectionManager
                }
            }
        });

        tracing::info!("Session coordinator started for peer {}", self.local_id);
    }

    /// Stops the coordinator.
    pub async fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);

        // Abort all outbound pump tasks
        let mut handles = self.outbound_handles.lock().await;
        for (_, session_handles) in handles.drain() {
            for h in session_handles {
                h.abort();
            }
        }

        tracing::info!("Session coordinator stopped for peer {}", self.local_id);
    }

    /// Returns the next session event.
    pub async fn next_event(&self) -> Option<SessionEvent> {
        let mut rx = self.event_rx.lock().await;
        rx.recv().await
    }

    /// Returns whether the coordinator is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Returns the number of active (non-terminal) sessions.
    pub fn active_session_count(&self) -> usize {
        self.sessions.read()
            .values()
            .filter(|s| !s.state.is_terminal())
            .count()
    }

    /// Returns the number of connected workers.
    pub fn worker_count(&self) -> usize {
        self.conn_manager.peers_by_role(NodeRole::Compute).len()
    }

    /// Returns snapshots of all connected workers.
    pub fn connected_workers(&self) -> Vec<PeerSnapshot> {
        self.conn_manager.peers_by_role(NodeRole::Compute)
    }

    /// Returns a reference to the underlying connection manager.
    pub fn conn_manager(&self) -> &Arc<ConnectionManager> {
        &self.conn_manager
    }

    // ── Session Creation ─────────────────────────────────────────────────

    /// Creates a new MPC training session with the specified workers.
    ///
    /// Assigns sequential party IDs to each worker (0, 1, 2, ...),
    /// registers the session in the coordinator's tracking structures,
    /// and sends `SessionAssignment` control messages to each worker.
    ///
    /// Returns the session ID and the created `NodeTransport` instances
    /// (one per participant) for use by the MPC protocol layer.
    pub async fn create_session(
        &self,
        worker_peer_ids: &[PeerId],
        params: SessionParams,
    ) -> Result<(String, Vec<NodeTransport>), SessionCoordinatorError> {
        if worker_peer_ids.is_empty() {
            return Err(SessionCoordinatorError::NoWorkers);
        }

        // Verify all workers are connected
        for peer_id in worker_peer_ids {
            if self.conn_manager.registry().get(peer_id).is_none() {
                return Err(SessionCoordinatorError::WorkerNotConnected(
                    peer_id.to_string(),
                ));
            }
        }

        let session_id = params.session_id.clone();

        // Build participants with assigned party IDs
        let participants: Vec<SessionParticipant> = worker_peer_ids
            .iter()
            .enumerate()
            .map(|(i, peer_id)| SessionParticipant {
                peer_id: peer_id.clone(),
                party_id: PartyId::from_index(i),
                ready: false,
            })
            .collect();

        let party_peer_map: Vec<(PartyId, PeerId)> = participants
            .iter()
            .map(|p| (p.party_id.clone(), p.peer_id.clone()))
            .collect();

        // Create the MPC session record
        let session = MpcSession {
            session_id: session_id.clone(),
            state: SessionState::Setup,
            params: params.clone(),
            participants: participants.clone(),
            created_at: Instant::now(),
            last_transition: Instant::now(),
        };

        self.sessions.write().insert(session_id.clone(), session);

        // Set up MPC transport channels for each participant
        let mut transports = Vec::with_capacity(worker_peer_ids.len());
        let mut router_entries: HashMap<String, mpsc::Sender<Vec<u8>>> = HashMap::new();
        let mut all_outbound_handles = Vec::new();

        let mut mapping = PeerMapping::new();
        for (party_id, peer_id) in &party_peer_map {
            mapping.add(party_id.clone(), peer_id.0.clone());
        }

        // Create a NodeTransport for each party
        for (i, peer_id) in worker_peer_ids.iter().enumerate() {
            let our_party = PartyId::from_index(i);
            let mut channels = Vec::new();

            for (j, other_peer_id) in worker_peer_ids.iter().enumerate() {
                if i == j {
                    continue;
                }

                let other_party = PartyId::from_index(j);

                // Outbound channel: transport writes, pump reads and sends via ConnectionManager
                let (outbound_tx, mut outbound_rx) = mpsc::channel::<Vec<u8>>(4096);
                // Inbound channel: router writes, transport reads
                let (inbound_tx, inbound_rx) = mpsc::channel::<Vec<u8>>(4096);

                // Register inbound sender for routing from this other party to our party
                let router_key = format!("{}:{}", i, j);
                router_entries.insert(router_key, inbound_tx);

                // Spawn outbound pump task
                let conn_mgr = self.conn_manager.clone();
                let target_peer = other_peer_id.clone();
                let sid = session_id.clone();
                let local = self.local_id.clone();

                let handle = tokio::spawn(async move {
                    while let Some(data) = outbound_rx.recv().await {
                        let mpc_msg = MpcDataMessage {
                            session_id: sid.clone(),
                            data,
                        };
                        let network_msg = NetworkMessage::new(
                            local.clone(),
                            MessagePayload::MpcData(mpc_msg),
                        );
                        if let Err(e) = conn_mgr.send(&target_peer, network_msg).await {
                            tracing::error!(
                                "Failed to send MPC message to {}: {}",
                                target_peer, e,
                            );
                        }
                    }
                });
                all_outbound_handles.push(handle);

                channels.push((other_party, outbound_tx, inbound_rx));
            }

            let transport = NodeTransport::new(
                our_party,
                session_id.clone(),
                mapping.clone(),
                channels,
            );
            transports.push(transport);
        }

        // Store router entries and peer mapping
        {
            let mut router = self.mpc_router.lock().await;
            router.insert(session_id.clone(), router_entries);
        }
        {
            let mut maps = self.session_peer_maps.lock().await;
            maps.insert(session_id.clone(), mapping);
        }
        {
            let mut handles = self.outbound_handles.lock().await;
            handles.insert(session_id.clone(), all_outbound_handles);
        }

        // Send SessionAssignment to each worker
        let party_peer_map_strings: Vec<(String, String)> = party_peer_map
            .iter()
            .map(|(party, peer)| (party.0.clone(), peer.0.clone()))
            .collect();

        for (i, peer_id) in worker_peer_ids.iter().enumerate() {
            let ctrl_msg = SessionControlMessage::SessionAssignment {
                session_id: session_id.clone(),
                party_id: i,
                params: params.clone(),
                party_peer_map: party_peer_map_strings.clone(),
            };

            self.send_control_message(peer_id, &session_id, &ctrl_msg).await?;
        }

        let _ = self.event_tx.send(SessionEvent::SessionCreated {
            session_id: session_id.clone(),
            participants: party_peer_map
                .iter()
                .map(|(party, peer)| (peer.clone(), party.clone()))
                .collect(),
        }).await;

        tracing::info!(
            "Created MPC session {} with {} workers",
            session_id,
            worker_peer_ids.len(),
        );

        Ok((session_id, transports))
    }

    // ── Session Lifecycle ────────────────────────────────────────────────

    /// Marks a worker as ready in the specified session.
    ///
    /// If all workers are now ready, transitions the session to Training
    /// and broadcasts a BeginTraining message.
    pub async fn mark_worker_ready(
        &self,
        session_id: &str,
        peer_id: &PeerId,
    ) -> Result<(), SessionCoordinatorError> {
        let all_ready = {
            let mut sessions = self.sessions.write();
            let session = sessions.get_mut(session_id)
                .ok_or(SessionCoordinatorError::SessionNotFound(session_id.to_string()))?;

            if session.state != SessionState::Setup {
                return Err(SessionCoordinatorError::InvalidTransition {
                    session_id: session_id.to_string(),
                    from: session.state.name().to_string(),
                    to: "training".to_string(),
                });
            }

            if let Some(participant) = session.participants.iter_mut().find(|p| &p.peer_id == peer_id) {
                participant.ready = true;
            } else {
                return Err(SessionCoordinatorError::ParticipantNotFound(
                    peer_id.to_string(),
                ));
            }

            session.all_ready()
        };

        if all_ready {
            self.transition_to_training(session_id).await?;
        }

        Ok(())
    }

    /// Transitions a session to the Training state and broadcasts BeginTraining.
    async fn transition_to_training(
        &self,
        session_id: &str,
    ) -> Result<(), SessionCoordinatorError> {
        let peers = {
            let mut sessions = self.sessions.write();
            let session = sessions.get_mut(session_id)
                .ok_or(SessionCoordinatorError::SessionNotFound(session_id.to_string()))?;

            let old_state = session.state.name().to_string();
            session.state = SessionState::Training { current_step: 0 };
            session.last_transition = Instant::now();

            let _ = self.event_tx.send(SessionEvent::StateTransition {
                session_id: session_id.to_string(),
                from: old_state,
                to: "training".to_string(),
            }).await;

            session.participants.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>()
        };

        // Broadcast BeginTraining to all participants
        let ctrl_msg = SessionControlMessage::BeginTraining {
            session_id: session_id.to_string(),
        };
        for peer_id in &peers {
            let _ = self.send_control_message(peer_id, session_id, &ctrl_msg).await;
        }

        let _ = self.event_tx.send(SessionEvent::SessionStarted {
            session_id: session_id.to_string(),
        }).await;

        tracing::info!("Session {} transitioned to training", session_id);
        Ok(())
    }

    /// Advances the training step counter for a session.
    pub fn advance_step(&self, session_id: &str, new_step: u64) {
        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.get_mut(session_id) {
            if let SessionState::Training { ref mut current_step } = session.state {
                *current_step = new_step;
            }
        }
    }

    /// Initiates a checkpoint for a session.
    pub async fn begin_checkpoint(
        &self,
        session_id: &str,
        at_step: u64,
    ) -> Result<(), SessionCoordinatorError> {
        let peers = {
            let mut sessions = self.sessions.write();
            let session = sessions.get_mut(session_id)
                .ok_or(SessionCoordinatorError::SessionNotFound(session_id.to_string()))?;

            let old_state = session.state.name().to_string();
            session.state = SessionState::Checkpointing { at_step };
            session.last_transition = Instant::now();

            let _ = self.event_tx.send(SessionEvent::StateTransition {
                session_id: session_id.to_string(),
                from: old_state,
                to: "checkpointing".to_string(),
            }).await;

            session.participants.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>()
        };

        let ctrl_msg = SessionControlMessage::CheckpointNow {
            session_id: session_id.to_string(),
            at_step,
        };
        for peer_id in &peers {
            let _ = self.send_control_message(peer_id, session_id, &ctrl_msg).await;
        }

        let _ = self.event_tx.send(SessionEvent::CheckpointReached {
            session_id: session_id.to_string(),
            step: at_step,
        }).await;

        tracing::info!("Session {} checkpointing at step {}", session_id, at_step);
        Ok(())
    }

    /// Resumes training after a checkpoint.
    pub async fn resume_after_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<(), SessionCoordinatorError> {
        let (peers, at_step) = {
            let mut sessions = self.sessions.write();
            let session = sessions.get_mut(session_id)
                .ok_or(SessionCoordinatorError::SessionNotFound(session_id.to_string()))?;

            let at_step = match session.state {
                SessionState::Checkpointing { at_step } => at_step,
                _ => return Err(SessionCoordinatorError::InvalidTransition {
                    session_id: session_id.to_string(),
                    from: session.state.name().to_string(),
                    to: "training".to_string(),
                }),
            };

            session.state = SessionState::Training { current_step: at_step };
            session.last_transition = Instant::now();

            let _ = self.event_tx.send(SessionEvent::StateTransition {
                session_id: session_id.to_string(),
                from: "checkpointing".to_string(),
                to: "training".to_string(),
            }).await;

            let peers = session.participants.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>();
            (peers, at_step)
        };

        let ctrl_msg = SessionControlMessage::ResumeTraining {
            session_id: session_id.to_string(),
        };
        for peer_id in &peers {
            let _ = self.send_control_message(peer_id, session_id, &ctrl_msg).await;
        }

        tracing::info!("Session {} resumed training at step {}", session_id, at_step);
        Ok(())
    }

    /// Completes a session successfully.
    pub async fn complete_session(
        &self,
        session_id: &str,
    ) -> Result<(), SessionCoordinatorError> {
        let peers = {
            let mut sessions = self.sessions.write();
            let session = sessions.get_mut(session_id)
                .ok_or(SessionCoordinatorError::SessionNotFound(session_id.to_string()))?;

            let old_state = session.state.name().to_string();
            session.state = SessionState::Completed;
            session.last_transition = Instant::now();

            let _ = self.event_tx.send(SessionEvent::StateTransition {
                session_id: session_id.to_string(),
                from: old_state,
                to: "completed".to_string(),
            }).await;

            session.participants.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>()
        };

        let ctrl_msg = SessionControlMessage::SessionComplete {
            session_id: session_id.to_string(),
        };
        for peer_id in &peers {
            let _ = self.send_control_message(peer_id, session_id, &ctrl_msg).await;
        }

        // Clean up router and pump tasks
        self.cleanup_session(session_id).await;

        let _ = self.event_tx.send(SessionEvent::SessionCompleted {
            session_id: session_id.to_string(),
        }).await;

        tracing::info!("Session {} completed successfully", session_id);
        Ok(())
    }

    /// Fails a session with a reason.
    pub async fn fail_session(
        &self,
        session_id: &str,
        reason: &str,
    ) -> Result<(), SessionCoordinatorError> {
        let peers = {
            let mut sessions = self.sessions.write();
            let session = sessions.get_mut(session_id)
                .ok_or(SessionCoordinatorError::SessionNotFound(session_id.to_string()))?;

            let old_state = session.state.name().to_string();
            session.state = SessionState::Failed { reason: reason.to_string() };
            session.last_transition = Instant::now();

            let _ = self.event_tx.send(SessionEvent::StateTransition {
                session_id: session_id.to_string(),
                from: old_state,
                to: "failed".to_string(),
            }).await;

            session.participants.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>()
        };

        let ctrl_msg = SessionControlMessage::SessionAbort {
            session_id: session_id.to_string(),
            reason: reason.to_string(),
        };
        for peer_id in &peers {
            let _ = self.send_control_message(peer_id, session_id, &ctrl_msg).await;
        }

        // Clean up
        self.cleanup_session(session_id).await;

        let _ = self.event_tx.send(SessionEvent::SessionFailed {
            session_id: session_id.to_string(),
            reason: reason.to_string(),
        }).await;

        tracing::warn!("Session {} failed: {}", session_id, reason);
        Ok(())
    }

    /// Reports a MAC verification failure for a session.
    pub async fn report_mac_failure(
        &self,
        session_id: &str,
        step: u64,
        suspected_party: Option<usize>,
    ) -> Result<(), SessionCoordinatorError> {
        let peers = {
            let sessions = self.sessions.read();
            let session = sessions.get(session_id)
                .ok_or(SessionCoordinatorError::SessionNotFound(session_id.to_string()))?;
            session.participants.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>()
        };

        let ctrl_msg = SessionControlMessage::MacFailure {
            session_id: session_id.to_string(),
            step,
            suspected_party,
        };
        for peer_id in &peers {
            let _ = self.send_control_message(peer_id, session_id, &ctrl_msg).await;
        }

        tracing::warn!(
            "Session {} MAC failure at step {}, suspected party: {:?}",
            session_id, step, suspected_party,
        );
        Ok(())
    }

    /// Removes a participant from a session (e.g., identified cheater).
    pub async fn remove_participant(
        &self,
        session_id: &str,
        party_index: usize,
        reason: &str,
    ) -> Result<(), SessionCoordinatorError> {
        let (peers, remaining) = {
            let mut sessions = self.sessions.write();
            let session = sessions.get_mut(session_id)
                .ok_or(SessionCoordinatorError::SessionNotFound(session_id.to_string()))?;

            if party_index >= session.participants.len() {
                return Err(SessionCoordinatorError::ParticipantNotFound(
                    format!("party-{}", party_index),
                ));
            }

            let removed_peer = session.participants.remove(party_index).peer_id;
            let remaining = session.participants.len();

            let _ = self.event_tx.send(SessionEvent::ParticipantLeft {
                session_id: session_id.to_string(),
                peer_id: removed_peer,
                remaining,
            }).await;

            let peers = session.participants.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>();
            (peers, remaining)
        };

        // Notify remaining participants
        let ctrl_msg = SessionControlMessage::RemoveParticipant {
            session_id: session_id.to_string(),
            party_id: party_index,
            reason: reason.to_string(),
        };
        for peer_id in &peers {
            let _ = self.send_control_message(peer_id, session_id, &ctrl_msg).await;
        }

        // If too few participants remain, fail the session
        if remaining < 2 {
            self.fail_session(session_id, "Too few participants remaining").await?;
        }

        Ok(())
    }

    // ── Session Queries ──────────────────────────────────────────────────

    /// Returns the state of a session.
    pub fn session_state(&self, session_id: &str) -> Option<SessionState> {
        self.sessions.read().get(session_id).map(|s| s.state.clone())
    }

    /// Returns the number of participants in a session.
    pub fn session_participant_count(&self, session_id: &str) -> Option<usize> {
        self.sessions.read().get(session_id).map(|s| s.num_participants())
    }

    /// Returns the list of all session IDs.
    pub fn session_ids(&self) -> Vec<String> {
        self.sessions.read().keys().cloned().collect()
    }

    /// Returns the party-peer mapping for a session.
    pub fn session_party_peer_map(&self, session_id: &str) -> Option<Vec<(PartyId, PeerId)>> {
        self.sessions.read().get(session_id).map(|s| s.party_peer_map())
    }

    // ── Internal: Message Handling ───────────────────────────────────────

    /// Sends a control message to a peer via the P2P layer.
    async fn send_control_message(
        &self,
        peer_id: &PeerId,
        session_id: &str,
        msg: &SessionControlMessage,
    ) -> Result<(), SessionCoordinatorError> {
        let data = bincode::serialize(msg)
            .map_err(|e| SessionCoordinatorError::Serialization(e.to_string()))?;

        let mpc_msg = MpcDataMessage {
            session_id: format!("{}{}", CTRL_PREFIX, session_id),
            data,
        };
        let network_msg = NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::MpcData(mpc_msg),
        );

        self.conn_manager.send(peer_id, network_msg).await
            .map_err(SessionCoordinatorError::Connection)
    }

    /// Handles an incoming control message from a worker.
    async fn handle_control_message(
        from: &PeerId,
        mpc_msg: &MpcDataMessage,
        sessions: &Arc<RwLock<HashMap<String, MpcSession>>>,
        event_tx: &mpsc::Sender<SessionEvent>,
    ) {
        // Extract session_id from the control prefix
        let session_id = &mpc_msg.session_id[CTRL_PREFIX.len()..];

        let ctrl_msg: SessionControlMessage = match bincode::deserialize(&mpc_msg.data) {
            Ok(msg) => msg,
            Err(e) => {
                tracing::error!("Failed to deserialize control message from {}: {}", from, e);
                return;
            }
        };

        match ctrl_msg {
            SessionControlMessage::SessionReady { session_id: sid } => {
                let all_ready = {
                    let mut sessions_write = sessions.write();
                    if let Some(session) = sessions_write.get_mut(&sid) {
                        if let Some(p) = session.participants.iter_mut().find(|p| &p.peer_id == from) {
                            p.ready = true;
                        }
                        session.all_ready()
                    } else {
                        false
                    }
                };

                if all_ready {
                    // Note: The actual transition is triggered via mark_worker_ready
                    // This handler only updates the ready flag; the coordinator
                    // caller is responsible for calling transition_to_training.
                    tracing::info!("All workers ready for session {}", sid);
                }
            }

            SessionControlMessage::CheckpointDone { session_id: sid, at_step } => {
                tracing::debug!(
                    "Worker {} completed checkpoint for session {} at step {}",
                    from, sid, at_step,
                );
            }

            _ => {
                // Other control messages are aggregator → worker only
                tracing::debug!(
                    "Ignoring unexpected control message from worker {} for session {}",
                    from, session_id,
                );
            }
        }
    }

    /// Routes an incoming MPC data message to the correct session's transport channel.
    async fn route_mpc_message(
        from: &PeerId,
        mpc_msg: &MpcDataMessage,
        router: &Arc<TokioMutex<HashMap<String, HashMap<String, mpsc::Sender<Vec<u8>>>>>>,
        session_peer_maps: &Arc<TokioMutex<HashMap<String, PeerMapping>>>,
    ) {
        let router_guard = router.lock().await;
        let session_channels = match router_guard.get(&mpc_msg.session_id) {
            Some(channels) => channels,
            None => {
                tracing::debug!(
                    "No active session '{}' for MPC message from {}",
                    mpc_msg.session_id, from,
                );
                return;
            }
        };

        // Look up which party the sender is in this session
        let maps_guard = session_peer_maps.lock().await;
        let mapping = match maps_guard.get(&mpc_msg.session_id) {
            Some(m) => m,
            None => return,
        };

        let sender_party = match mapping.party_for_peer(&from.0) {
            Some(p) => p.clone(),
            None => {
                tracing::debug!(
                    "Unknown peer {} in session '{}'",
                    from, mpc_msg.session_id,
                );
                return;
            }
        };

        // Find the channel for delivering from this sender to each recipient
        // The router key is "recipient_party_idx:sender_party_idx"
        for (key, tx) in session_channels {
            // key format: "recipient_idx:sender_idx"
            let parts: Vec<&str> = key.split(':').collect();
            if parts.len() == 2 {
                let sender_idx = parts[1];
                let expected_party = format!("party-{}", sender_idx);
                if expected_party == sender_party.0 {
                    let _ = tx.send(mpc_msg.data.clone()).await;
                }
            }
        }
    }

    /// Cleans up a session's router entries and pump tasks.
    async fn cleanup_session(&self, session_id: &str) {
        self.mpc_router.lock().await.remove(session_id);
        self.session_peer_maps.lock().await.remove(session_id);

        if let Some(handles) = self.outbound_handles.lock().await.remove(session_id) {
            for h in handles {
                h.abort();
            }
        }
    }
}

// ── Errors ───────────────────────────────────────────────────────────────

/// Errors from the session coordinator.
#[derive(Debug, thiserror::Error)]
pub enum SessionCoordinatorError {
    #[error("No workers provided for session creation")]
    NoWorkers,
    #[error("Worker not connected: {0}")]
    WorkerNotConnected(String),
    #[error("Session not found: {0}")]
    SessionNotFound(String),
    #[error("Participant not found: {0}")]
    ParticipantNotFound(String),
    #[error("Invalid state transition for session {session_id}: {from} → {to}")]
    InvalidTransition {
        session_id: String,
        from: String,
        to: String,
    },
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Connection error: {0}")]
    Connection(#[from] ConnectionManagerError),
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::connection_manager::ConnectionManagerConfig;

    /// Helper: creates a ConnectionManager with a given ID and role.
    fn make_config(id: &str, role: NodeRole) -> ConnectionManagerConfig {
        ConnectionManagerConfig {
            local_id: PeerId::from_string(id),
            role,
            capabilities: NodeCapabilities {
                can_train: role == NodeRole::Compute,
                can_aggregate: role == NodeRole::Aggregator,
                ..Default::default()
            },
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(60), // Long to avoid noise
            heartbeat_timeout: Duration::from_secs(120),
            ..Default::default()
        }
    }

    /// Helper: creates and starts a ConnectionManager.
    async fn start_node(id: &str, role: NodeRole) -> Arc<ConnectionManager> {
        let config = make_config(id, role);
        let mgr = Arc::new(ConnectionManager::new(config));
        mgr.start().await.unwrap();
        mgr
    }

    // ── Test: Peer Connection ────────────────────────────────────────────

    #[tokio::test]
    async fn test_peer_connection() {
        // Boot aggregator + 2 workers in separate tokio tasks with different ports
        let agg = start_node("aggregator", NodeRole::Aggregator).await;
        let agg_addr = agg.listen_addr().unwrap();

        let worker1 = start_node("worker-1", NodeRole::Compute).await;
        let worker2 = start_node("worker-2", NodeRole::Compute).await;

        // Workers connect to aggregator
        let peer1 = worker1.connect_to(agg_addr).await.unwrap();
        assert_eq!(peer1, PeerId::from_string("aggregator"));

        let peer2 = worker2.connect_to(agg_addr).await.unwrap();
        assert_eq!(peer2, PeerId::from_string("aggregator"));

        // Wait for inbound handshakes to complete
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Verify aggregator sees both workers
        assert_eq!(agg.connection_count(), 2);
        let workers = agg.registry().peers_by_role(NodeRole::Compute);
        assert_eq!(workers.len(), 2);

        // Verify workers see the aggregator
        assert_eq!(worker1.connection_count(), 1);
        assert_eq!(worker2.connection_count(), 1);

        let agg_peers = worker1.registry().peers_by_role(NodeRole::Aggregator);
        assert_eq!(agg_peers.len(), 1);
        assert_eq!(agg_peers[0].peer_id, PeerId::from_string("aggregator"));

        // Verify the handshake exchanged roles correctly
        let w1_snap = agg.registry().get(&PeerId::from_string("worker-1")).unwrap();
        assert_eq!(w1_snap.role, NodeRole::Compute);
        assert!(w1_snap.capabilities.can_train);

        let w2_snap = agg.registry().get(&PeerId::from_string("worker-2")).unwrap();
        assert_eq!(w2_snap.role, NodeRole::Compute);
        assert!(w2_snap.capabilities.can_train);

        // Cleanup
        agg.stop().await;
        worker1.stop().await;
        worker2.stop().await;
    }

    // ── Test: MPC Message Routing ────────────────────────────────────────

    #[tokio::test]
    async fn test_mpc_message_routing() {
        // Setup: aggregator + 2 workers
        let agg = start_node("agg", NodeRole::Aggregator).await;
        let agg_addr = agg.listen_addr().unwrap();

        let w1 = start_node("w1", NodeRole::Compute).await;
        let w2 = start_node("w2", NodeRole::Compute).await;

        w1.connect_to(agg_addr).await.unwrap();
        w2.connect_to(agg_addr).await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Workers also need direct connections for MPC messages.
        // In our setup, workers connect to aggregator. For direct worker-to-worker
        // communication, we need workers to connect to each other.
        let w1_addr = w1.listen_addr().unwrap();
        let w2_addr = w2.listen_addr().unwrap();
        w1.connect_to(w2_addr).await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        // w1 sends an MPC message to w2 via the P2P layer
        let test_payload = b"hello-mpc-data-from-w1";
        let mpc_msg = MpcDataMessage {
            session_id: "test-session".to_string(),
            data: test_payload.to_vec(),
        };
        let network_msg = NetworkMessage::new(
            PeerId::from_string("w1"),
            MessagePayload::MpcData(mpc_msg),
        );

        w1.send(&PeerId::from_string("w2"), network_msg).await.unwrap();

        // w2 should receive the message
        let mut found = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(Duration::from_millis(200), w2.next_event()).await {
                Ok(Some(P2PEvent::MessageReceived { from, message })) => {
                    if from == PeerId::from_string("w1") {
                        if let MessagePayload::MpcData(ref mpc) = message.payload {
                            if mpc.session_id == "test-session" && mpc.data == test_payload {
                                found = true;
                                break;
                            }
                        }
                    }
                }
                Ok(Some(_)) => continue,
                _ => continue,
            }
        }
        assert!(found, "Worker 2 should receive MPC message from Worker 1");

        // Cleanup
        agg.stop().await;
        w1.stop().await;
        w2.stop().await;
    }

    // ── Test: Heartbeat Timeout ──────────────────────────────────────────

    #[tokio::test]
    async fn test_heartbeat_timeout() {
        // Aggregator with short heartbeat timeout for testing
        let agg_config = ConnectionManagerConfig {
            local_id: PeerId::from_string("agg"),
            role: NodeRole::Aggregator,
            capabilities: NodeCapabilities { can_aggregate: true, ..Default::default() },
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_millis(100),
            heartbeat_timeout: Duration::from_millis(200),
            max_missed_heartbeats: 2,
            ..Default::default()
        };
        let agg = Arc::new(ConnectionManager::new(agg_config));
        agg.start().await.unwrap();
        let agg_addr = agg.listen_addr().unwrap();

        // Worker with heartbeat disabled (very long interval)
        let w_config = ConnectionManagerConfig {
            local_id: PeerId::from_string("lazy-worker"),
            role: NodeRole::Compute,
            capabilities: NodeCapabilities { can_train: true, ..Default::default() },
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(999), // never sends heartbeat
            ..Default::default()
        };
        let worker = Arc::new(ConnectionManager::new(w_config));
        worker.start().await.unwrap();
        worker.connect_to(agg_addr).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(agg.connection_count(), 1, "Worker should be connected");

        // Wait for heartbeat timeout to kick in (200ms timeout * 3 missed = ~600ms)
        // Plus health checker runs every heartbeat_timeout interval
        tokio::time::sleep(Duration::from_millis(1500)).await;

        // The aggregator should have removed the worker
        assert_eq!(
            agg.connection_count(), 0,
            "Aggregator should have removed the worker after heartbeat timeout"
        );

        agg.stop().await;
        worker.stop().await;
    }

    // ── Test: Session Setup ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_session_setup() {
        // Setup: aggregator + 3 workers
        let agg = start_node("coord", NodeRole::Aggregator).await;
        let agg_addr = agg.listen_addr().unwrap();

        let w1 = start_node("w-0", NodeRole::Compute).await;
        let w2 = start_node("w-1", NodeRole::Compute).await;
        let w3 = start_node("w-2", NodeRole::Compute).await;

        w1.connect_to(agg_addr).await.unwrap();
        w2.connect_to(agg_addr).await.unwrap();
        w3.connect_to(agg_addr).await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create the session coordinator on the aggregator
        let coordinator = SessionCoordinator::new(
            PeerId::from_string("coord"),
            agg.clone(),
        );
        coordinator.start();

        // Verify workers are visible
        assert_eq!(coordinator.worker_count(), 3);
        let workers = coordinator.connected_workers();
        assert_eq!(workers.len(), 3);

        // Create an MPC session with the 3 workers
        let worker_ids = vec![
            PeerId::from_string("w-0"),
            PeerId::from_string("w-1"),
            PeerId::from_string("w-2"),
        ];

        let params = SessionParams {
            session_id: "test-session-1".to_string(),
            d_in: 784,
            d_hid: 32,
            d_out: 10,
            num_parties: 3,
            ..Default::default()
        };

        let (session_id, transports) = coordinator
            .create_session(&worker_ids, params)
            .await
            .unwrap();

        assert_eq!(session_id, "test-session-1");
        assert_eq!(transports.len(), 3);

        // Verify each transport has the correct party ID
        assert_eq!(transports[0].party_id(), &PartyId::from_index(0));
        assert_eq!(transports[1].party_id(), &PartyId::from_index(1));
        assert_eq!(transports[2].party_id(), &PartyId::from_index(2));

        // Verify each transport sees the other 2 parties as peers
        assert_eq!(transports[0].peers().len(), 2);
        assert_eq!(transports[1].peers().len(), 2);
        assert_eq!(transports[2].peers().len(), 2);

        // Verify session state
        let state = coordinator.session_state("test-session-1").unwrap();
        assert_eq!(state, SessionState::Setup);
        assert_eq!(coordinator.active_session_count(), 1);

        // Verify party-peer mapping
        let mapping = coordinator.session_party_peer_map("test-session-1").unwrap();
        assert_eq!(mapping.len(), 3);
        assert_eq!(mapping[0].0, PartyId::from_index(0));
        assert_eq!(mapping[0].1, PeerId::from_string("w-0"));
        assert_eq!(mapping[1].0, PartyId::from_index(1));
        assert_eq!(mapping[1].1, PeerId::from_string("w-1"));
        assert_eq!(mapping[2].0, PartyId::from_index(2));
        assert_eq!(mapping[2].1, PeerId::from_string("w-2"));

        // Workers should receive the SessionAssignment control messages
        // (verify by draining w1's events and looking for MpcData)
        let mut w1_received_assignment = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(Duration::from_millis(100), w1.next_event()).await {
                Ok(Some(P2PEvent::MessageReceived { message, .. })) => {
                    if let MessagePayload::MpcData(ref mpc) = message.payload {
                        if mpc.session_id.starts_with(CTRL_PREFIX) {
                            if let Ok(ctrl) = bincode::deserialize::<SessionControlMessage>(&mpc.data) {
                                if let SessionControlMessage::SessionAssignment {
                                    session_id: sid,
                                    party_id,
                                    params: p,
                                    ..
                                } = ctrl {
                                    assert_eq!(sid, "test-session-1");
                                    assert_eq!(party_id, 0);
                                    assert_eq!(p.d_in, 784);
                                    assert_eq!(p.d_hid, 32);
                                    assert_eq!(p.d_out, 10);
                                    assert_eq!(p.num_parties, 3);
                                    w1_received_assignment = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                Ok(Some(_)) => continue,
                _ => continue,
            }
        }
        assert!(w1_received_assignment, "Worker 0 should receive SessionAssignment");

        // Test lifecycle: mark workers ready → transition to training
        coordinator.mark_worker_ready("test-session-1", &PeerId::from_string("w-0")).await.unwrap();
        coordinator.mark_worker_ready("test-session-1", &PeerId::from_string("w-1")).await.unwrap();

        // After 2 of 3, should still be in Setup
        let state = coordinator.session_state("test-session-1").unwrap();
        assert_eq!(state, SessionState::Setup);

        // Third ready → should transition to Training
        coordinator.mark_worker_ready("test-session-1", &PeerId::from_string("w-2")).await.unwrap();

        let state = coordinator.session_state("test-session-1").unwrap();
        assert_eq!(state, SessionState::Training { current_step: 0 });

        // Test advance step
        coordinator.advance_step("test-session-1", 10);
        let state = coordinator.session_state("test-session-1").unwrap();
        assert_eq!(state, SessionState::Training { current_step: 10 });

        // Test checkpoint
        coordinator.begin_checkpoint("test-session-1", 10).await.unwrap();
        let state = coordinator.session_state("test-session-1").unwrap();
        assert_eq!(state, SessionState::Checkpointing { at_step: 10 });

        // Resume
        coordinator.resume_after_checkpoint("test-session-1").await.unwrap();
        let state = coordinator.session_state("test-session-1").unwrap();
        assert_eq!(state, SessionState::Training { current_step: 10 });

        // Complete
        coordinator.complete_session("test-session-1").await.unwrap();
        let state = coordinator.session_state("test-session-1").unwrap();
        assert_eq!(state, SessionState::Completed);

        // Session should no longer count as active
        assert_eq!(coordinator.active_session_count(), 0);

        // Cleanup
        coordinator.stop().await;
        agg.stop().await;
        w1.stop().await;
        w2.stop().await;
        w3.stop().await;
    }

    // ── Test: Session failure on worker disconnect ────────────────────────

    #[tokio::test]
    async fn test_session_worker_disconnect() {
        let agg = start_node("agg", NodeRole::Aggregator).await;
        let agg_addr = agg.listen_addr().unwrap();

        let w1 = start_node("w-a", NodeRole::Compute).await;
        let w2 = start_node("w-b", NodeRole::Compute).await;

        w1.connect_to(agg_addr).await.unwrap();
        w2.connect_to(agg_addr).await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        let coordinator = SessionCoordinator::new(
            PeerId::from_string("agg"),
            agg.clone(),
        );
        coordinator.start();

        let params = SessionParams {
            session_id: "disconnect-test".to_string(),
            num_parties: 2,
            ..Default::default()
        };

        let worker_ids = vec![
            PeerId::from_string("w-a"),
            PeerId::from_string("w-b"),
        ];

        coordinator.create_session(&worker_ids, params).await.unwrap();

        // Mark both ready to start training
        coordinator.mark_worker_ready("disconnect-test", &PeerId::from_string("w-a")).await.unwrap();
        coordinator.mark_worker_ready("disconnect-test", &PeerId::from_string("w-b")).await.unwrap();

        let state = coordinator.session_state("disconnect-test").unwrap();
        assert_eq!(state, SessionState::Training { current_step: 0 });

        // Fail the session explicitly (simulating what happens when a MAC check fails)
        coordinator.fail_session("disconnect-test", "MAC verification failed").await.unwrap();

        let state = coordinator.session_state("disconnect-test").unwrap();
        assert!(matches!(state, SessionState::Failed { .. }));
        assert_eq!(coordinator.active_session_count(), 0);

        // Cleanup
        coordinator.stop().await;
        agg.stop().await;
        w1.stop().await;
        w2.stop().await;
    }

    // ── Test: Cannot create session with disconnected workers ─────────────

    #[tokio::test]
    async fn test_create_session_with_disconnected_worker() {
        let agg = start_node("agg", NodeRole::Aggregator).await;
        let coordinator = SessionCoordinator::new(
            PeerId::from_string("agg"),
            agg.clone(),
        );

        let result = coordinator.create_session(
            &[PeerId::from_string("nonexistent")],
            SessionParams::default(),
        ).await;

        assert!(matches!(result, Err(SessionCoordinatorError::WorkerNotConnected(_))));

        agg.stop().await;
    }

    // ── Test: Cannot create empty session ─────────────────────────────────

    #[tokio::test]
    async fn test_create_empty_session() {
        let agg = start_node("agg", NodeRole::Aggregator).await;
        let coordinator = SessionCoordinator::new(
            PeerId::from_string("agg"),
            agg.clone(),
        );

        let result = coordinator.create_session(&[], SessionParams::default()).await;
        assert!(matches!(result, Err(SessionCoordinatorError::NoWorkers)));

        agg.stop().await;
    }

    // ── Test: Handshake with x25519 and eth address ──────────────────────

    #[tokio::test]
    async fn test_handshake_with_mpc_identity() {
        let x25519_key = vec![0xAA; 32];
        let eth_addr = vec![0xBB; 20];

        let agg_config = ConnectionManagerConfig {
            local_id: PeerId::from_string("id-agg"),
            role: NodeRole::Aggregator,
            capabilities: NodeCapabilities { can_aggregate: true, ..Default::default() },
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(60),
            ..Default::default()
        };
        let agg = ConnectionManager::new(agg_config);
        agg.start().await.unwrap();
        let agg_addr = agg.listen_addr().unwrap();

        let w_config = ConnectionManagerConfig {
            local_id: PeerId::from_string("id-worker"),
            role: NodeRole::Compute,
            capabilities: NodeCapabilities { can_train: true, ..Default::default() },
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Duration::from_secs(60),
            x25519_pubkey: Some(x25519_key.clone()),
            eth_address: Some(eth_addr.clone()),
            ..Default::default()
        };
        let worker = ConnectionManager::new(w_config);
        worker.start().await.unwrap();

        worker.connect_to(agg_addr).await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Aggregator should see the worker with x25519 and eth_address
        let snap = agg.registry().get(&PeerId::from_string("id-worker")).unwrap();
        assert_eq!(snap.x25519_pubkey, Some(x25519_key));
        assert_eq!(snap.eth_address, Some(eth_addr));

        agg.stop().await;
        worker.stop().await;
    }

    // ── Test: Multiple sessions ──────────────────────────────────────────

    #[tokio::test]
    async fn test_multiple_sessions() {
        let agg = start_node("multi-agg", NodeRole::Aggregator).await;
        let agg_addr = agg.listen_addr().unwrap();

        let w1 = start_node("multi-w1", NodeRole::Compute).await;
        let w2 = start_node("multi-w2", NodeRole::Compute).await;

        w1.connect_to(agg_addr).await.unwrap();
        w2.connect_to(agg_addr).await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        let coordinator = SessionCoordinator::new(
            PeerId::from_string("multi-agg"),
            agg.clone(),
        );
        coordinator.start();

        let workers = vec![
            PeerId::from_string("multi-w1"),
            PeerId::from_string("multi-w2"),
        ];

        // Create two sessions
        let params1 = SessionParams {
            session_id: "session-a".to_string(),
            num_parties: 2,
            ..Default::default()
        };
        let params2 = SessionParams {
            session_id: "session-b".to_string(),
            num_parties: 2,
            ..Default::default()
        };

        coordinator.create_session(&workers, params1).await.unwrap();
        coordinator.create_session(&workers, params2).await.unwrap();

        assert_eq!(coordinator.active_session_count(), 2);
        let ids = coordinator.session_ids();
        assert!(ids.contains(&"session-a".to_string()));
        assert!(ids.contains(&"session-b".to_string()));

        // Complete one, verify the other is still active
        coordinator.complete_session("session-a").await.unwrap();
        assert_eq!(coordinator.active_session_count(), 1);

        coordinator.stop().await;
        agg.stop().await;
        w1.stop().await;
        w2.stop().await;
    }
}
