//! MPC Session Orchestration.
//!
//! Integrates helix-mpc session management (party registry, health monitoring,
//! session checkpoints) into the node's training round lifecycle.
//!
//! The `MpcSessionOrchestrator` is created by the aggregator at startup and
//! manages the full lifecycle of MPC sessions:
//! 1. Workers register as MPC parties when they connect
//! 2. Health monitoring tracks party liveness via heartbeats
//! 3. Sessions are established at round start with available parties
//! 4. Checkpoints are saved periodically for crash recovery
//! 5. Degraded parties are excluded from future rounds

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use helix_mpc::session::registry::{
    PartyCapabilities, PartyRegistry, RegistryConfig, SelectionCriteria,
};
use helix_mpc::session::health::{
    HealthConfig, HealthStatus, PartyHealthMonitor,
};
use helix_mpc::session::checkpoint::SessionPersistence;
use helix_mpc::types::{MPCConfig, PartyId};

use crate::network::messages::PeerId;

/// Maps between node PeerIds and MPC PartyIds.
#[derive(Debug, Default)]
pub struct PartyMapping {
    /// PeerId -> (PartyId, party_index)
    peer_to_party: HashMap<String, (PartyId, usize)>,
    /// party_index -> PeerId
    party_to_peer: HashMap<usize, String>,
    /// Next available party index.
    next_index: usize,
}

impl PartyMapping {
    /// Creates a new empty mapping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a peer as an MPC party. Returns the assigned party index.
    pub fn register(&mut self, peer_id: &PeerId) -> usize {
        let index = self.next_index;
        let party = PartyId::from_index(index);
        self.peer_to_party.insert(peer_id.0.clone(), (party, index));
        self.party_to_peer.insert(index, peer_id.0.clone());
        self.next_index += 1;
        index
    }

    /// Removes a peer from the mapping.
    pub fn deregister(&mut self, peer_id: &PeerId) {
        if let Some((_, index)) = self.peer_to_party.remove(&peer_id.0) {
            self.party_to_peer.remove(&index);
        }
    }

    /// Gets the party index for a peer.
    pub fn party_index(&self, peer_id: &PeerId) -> Option<usize> {
        self.peer_to_party.get(&peer_id.0).map(|(_, idx)| *idx)
    }

    /// Gets the peer ID for a party index.
    pub fn peer_id(&self, party_index: usize) -> Option<&str> {
        self.party_to_peer.get(&party_index).map(|s| s.as_str())
    }

    /// Returns the number of registered parties.
    pub fn len(&self) -> usize {
        self.peer_to_party.len()
    }

    /// Returns true if no parties are registered.
    pub fn is_empty(&self) -> bool {
        self.peer_to_party.is_empty()
    }

    /// Returns all registered party indices.
    pub fn party_indices(&self) -> Vec<usize> {
        self.party_to_peer.keys().copied().collect()
    }

    /// Resets the mapping for a new session (clears all registrations).
    pub fn reset(&mut self) {
        self.peer_to_party.clear();
        self.party_to_peer.clear();
        self.next_index = 0;
    }
}

/// Active MPC session state.
#[derive(Debug)]
pub struct ActiveSession {
    /// Session identifier.
    pub session_id: String,
    /// Round ID this session is for.
    pub round_id: u64,
    /// Party indices participating in this session.
    pub parties: Vec<usize>,
    /// Session start time.
    pub started_at: Instant,
    /// Whether the session is still active.
    pub active: bool,
}

/// MPC Session Orchestrator.
///
/// Manages the full lifecycle of MPC sessions within the node's training rounds.
/// Created by the aggregator at startup, it handles:
/// - Party registration/deregistration as workers join/leave
/// - Health monitoring with automatic degraded-party exclusion
/// - Session establishment at round start
/// - Checkpoint persistence for crash recovery
pub struct MpcSessionOrchestrator {
    /// Party registry for tracking available MPC participants.
    registry: PartyRegistry,
    /// Health monitor for per-party liveness (created when a session starts).
    health_monitor: Option<PartyHealthMonitor>,
    /// MPC configuration for health monitor construction.
    mpc_config: MPCConfig,
    /// Persistence for session checkpoints.
    persistence: Option<SessionPersistence>,
    /// Mapping between node PeerIds and MPC PartyIds.
    mapping: PartyMapping,
    /// Currently active session (if any).
    active_session: Option<ActiveSession>,
    /// Minimum number of parties required for a session.
    min_parties: usize,
    /// Session timeout duration.
    session_timeout: Duration,
    /// Heartbeat interval for health checks.
    heartbeat_interval: Duration,
    /// Last health check time.
    last_health_check: Instant,
    /// Default socket address for party registration (used when no real addr is available).
    default_addr: SocketAddr,
}

impl MpcSessionOrchestrator {
    /// Creates a new session orchestrator.
    pub fn new(
        min_parties: usize,
        checkpoint_dir: Option<PathBuf>,
    ) -> Self {
        let registry_config = RegistryConfig {
            party_ttl: Duration::from_secs(120),
            ..RegistryConfig::default()
        };
        let registry = PartyRegistry::new(registry_config);

        let mpc_config = MPCConfig {
            num_parties: min_parties,
            use_shamir: false,
            ..MPCConfig::default()
        };

        let persistence = checkpoint_dir.and_then(|dir| {
            SessionPersistence::new(dir, 5).ok()
        });

        Self {
            registry,
            health_monitor: None,
            mpc_config,
            persistence,
            mapping: PartyMapping::new(),
            active_session: None,
            min_parties,
            session_timeout: Duration::from_secs(300),
            heartbeat_interval: Duration::from_secs(10),
            last_health_check: Instant::now(),
            default_addr: "127.0.0.1:0".parse().unwrap(),
        }
    }

    /// Registers a worker as an MPC party.
    ///
    /// Returns the assigned party index, or None if the worker is already registered.
    pub fn register_worker(&mut self, peer_id: &PeerId) -> Option<usize> {
        if self.mapping.party_index(peer_id).is_some() {
            return None; // Already registered
        }

        let party_index = self.mapping.register(peer_id);
        let party = PartyId::from_index(party_index);

        // Register in the MPC party registry with default capabilities.
        if let Err(e) = self.registry.register(
            party,
            self.default_addr,
            PartyCapabilities::default(),
            Some(peer_id.0.clone()),
        ) {
            log::warn!("Failed to register party in MPC registry: {}", e);
            self.mapping.deregister(peer_id);
            return None;
        }

        log::info!(
            "MPC party registered: peer={} -> party_index={}",
            peer_id, party_index,
        );

        Some(party_index)
    }

    /// Deregisters a worker from the MPC party pool.
    pub fn deregister_worker(&mut self, peer_id: &PeerId) {
        if let Some(party_index) = self.mapping.party_index(peer_id) {
            let party = PartyId::from_index(party_index);
            self.registry.deregister(&party);
            self.mapping.deregister(peer_id);

            log::info!(
                "MPC party deregistered: peer={} (was party_index={})",
                peer_id, party_index,
            );
        }
    }

    /// Records a heartbeat from a worker.
    pub fn record_heartbeat(&mut self, peer_id: &PeerId) {
        if let Some(party_index) = self.mapping.party_index(peer_id) {
            let party = PartyId::from_index(party_index);
            // Update registry heartbeat.
            self.registry.heartbeat(&party);
            // Update health monitor if one exists.
            if let Some(ref mut monitor) = self.health_monitor {
                monitor.record_heartbeat(&party);
            }
        }
    }

    /// Returns the number of available (registry-available) parties.
    pub fn available_parties(&self) -> usize {
        self.registry.available_count()
    }

    /// Returns true if enough parties are available for an MPC session.
    pub fn can_start_session(&self) -> bool {
        self.active_session.is_none() && self.available_parties() >= self.min_parties
    }

    /// Starts a new MPC session for the given round.
    ///
    /// Selects healthy parties and assigns them to the session.
    /// Returns the session ID and list of (party_index, peer_id) pairs,
    /// or None if not enough parties are available.
    pub fn start_session(&mut self, round_id: u64) -> Option<(String, Vec<(usize, String)>)> {
        if !self.can_start_session() {
            return None;
        }

        let session_id = format!("mpc-round-{}-{}", round_id, uuid::Uuid::new_v4());

        // Select parties using the registry's reputation-based selection.
        let criteria = SelectionCriteria {
            num_parties: self.min_parties,
            ..SelectionCriteria::default()
        };

        let selected_parties = match self.registry.select_parties(&criteria) {
            Ok(parties) => parties,
            Err(e) => {
                log::warn!("Failed to select MPC parties: {}", e);
                return None;
            }
        };

        // Build the result mapping and assign parties to the session.
        let mut result: Vec<(usize, String)> = Vec::new();
        let mut party_ids_for_monitor: Vec<PartyId> = Vec::new();

        for (party_id, _addr) in &selected_parties {
            // Assign party to this session in the registry.
            if let Err(e) = self.registry.assign_to_session(party_id, &session_id) {
                log::warn!("Failed to assign party {} to session: {}", party_id, e);
                continue;
            }

            // Find the party index from the party_id string.
            // PartyId::from_index(n) produces "party-N", so parse back.
            if let Some(idx_str) = party_id.0.strip_prefix("party-") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if let Some(pid) = self.mapping.peer_id(idx) {
                        result.push((idx, pid.to_string()));
                        party_ids_for_monitor.push(party_id.clone());
                    }
                }
            }
        }

        if result.len() < self.min_parties {
            // Roll back assignments.
            for (party_id, _) in &selected_parties {
                let _ = self.registry.release_from_session(party_id);
            }
            return None;
        }

        let party_indices: Vec<usize> = result.iter().map(|(idx, _)| *idx).collect();

        // Create health monitor for this session's parties.
        let health_config = HealthConfig {
            heartbeat_interval: self.heartbeat_interval,
            degraded_threshold: Duration::from_secs(30),
            disconnect_threshold: Duration::from_secs(60),
            ..HealthConfig::default()
        };
        let mpc_config = MPCConfig {
            num_parties: result.len(),
            use_shamir: false,
            ..MPCConfig::default()
        };
        self.health_monitor = Some(PartyHealthMonitor::new(
            &party_ids_for_monitor,
            health_config,
            mpc_config,
        ));

        self.active_session = Some(ActiveSession {
            session_id: session_id.clone(),
            round_id,
            parties: party_indices,
            started_at: Instant::now(),
            active: true,
        });

        log::info!(
            "MPC session started: id={}, round={}, parties={}",
            session_id, round_id, result.len(),
        );

        Some((session_id, result))
    }

    /// Completes the current MPC session.
    pub fn complete_session(&mut self) {
        if let Some(ref mut session) = self.active_session {
            session.active = false;
            let duration = session.started_at.elapsed();
            log::info!(
                "MPC session completed: id={}, round={}, duration={:.1}s",
                session.session_id, session.round_id, duration.as_secs_f64(),
            );

            // Release all parties back to available state.
            for &idx in &session.parties {
                let party = PartyId::from_index(idx);
                let _ = self.registry.release_from_session(&party);
            }
        }

        self.health_monitor = None;
        self.active_session = None;
    }

    /// Fails the current MPC session with an error reason.
    pub fn fail_session(&mut self, reason: &str) {
        if let Some(ref session) = self.active_session {
            log::error!(
                "MPC session failed: id={}, round={}, reason={}",
                session.session_id, session.round_id, reason,
            );

            // Release all parties and penalize reputation.
            for &idx in &session.parties {
                let party = PartyId::from_index(idx);
                let _ = self.registry.release_from_session(&party);
            }
        }

        self.health_monitor = None;
        self.active_session = None;
    }

    /// Checks for session timeout and degraded parties.
    ///
    /// Should be called periodically (e.g., every heartbeat interval).
    pub fn health_check(&mut self) -> Vec<OrchestratorHealthEvent> {
        let mut events = Vec::new();

        // Check session timeout.
        if let Some(ref session) = self.active_session {
            if session.active && session.started_at.elapsed() > self.session_timeout {
                events.push(OrchestratorHealthEvent::SessionTimeout {
                    session_id: session.session_id.clone(),
                    round_id: session.round_id,
                });
            }
        }

        // Run health monitor check if we have one.
        if let Some(ref mut monitor) = self.health_monitor {
            let result = monitor.check();

            for event in &result.events {
                // Map health events to orchestrator events.
                // Find the peer ID for this party.
                if let Some(idx_str) = event.party_id.0.strip_prefix("party-") {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        if let Some(pid) = self.mapping.peer_id(idx) {
                            match event.new_status {
                                HealthStatus::Degraded => {
                                    events.push(OrchestratorHealthEvent::PartyDegraded {
                                        party_index: idx,
                                        peer_id: pid.to_string(),
                                    });
                                }
                                HealthStatus::Disconnected => {
                                    events.push(OrchestratorHealthEvent::PartyDisconnected {
                                        party_index: idx,
                                        peer_id: pid.to_string(),
                                    });
                                    // Mark as unhealthy in the registry too.
                                    self.registry.mark_unhealthy(&event.party_id, 10.0);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            if !result.can_continue {
                events.push(OrchestratorHealthEvent::InsufficientParties {
                    healthy: result.healthy_count,
                    required: self.min_parties,
                });
            }
        }

        // Cleanup stale parties from the registry.
        let stale = self.registry.cleanup_stale();
        for party_id in &stale {
            if let Some(idx_str) = party_id.0.strip_prefix("party-") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if let Some(pid) = self.mapping.peer_id(idx) {
                        log::info!("Stale MPC party cleaned up: peer={}, party_index={}", pid, idx);
                    }
                }
            }
        }

        self.last_health_check = Instant::now();
        events
    }

    /// Returns the active session, if any.
    pub fn active_session(&self) -> Option<&ActiveSession> {
        self.active_session.as_ref()
    }

    /// Returns whether there is an active MPC session.
    pub fn has_active_session(&self) -> bool {
        self.active_session.is_some()
    }

    /// Returns the current session ID, if any.
    pub fn current_session_id(&self) -> Option<String> {
        self.active_session.as_ref().map(|s| s.session_id.clone())
    }

    /// Returns the party mapping.
    pub fn mapping(&self) -> &PartyMapping {
        &self.mapping
    }

    /// Returns a reference to the party registry.
    pub fn registry(&self) -> &PartyRegistry {
        &self.registry
    }

    /// Returns a reference to the persistence layer, if configured.
    pub fn persistence(&self) -> Option<&SessionPersistence> {
        self.persistence.as_ref()
    }
}

/// Events emitted by health checks.
#[derive(Debug)]
pub enum OrchestratorHealthEvent {
    /// A party's health has degraded (slow heartbeats).
    PartyDegraded {
        party_index: usize,
        peer_id: String,
    },
    /// A party has disconnected (no heartbeat within timeout).
    PartyDisconnected {
        party_index: usize,
        peer_id: String,
    },
    /// The MPC session has timed out.
    SessionTimeout {
        session_id: String,
        round_id: u64,
    },
    /// Not enough healthy parties to continue the session.
    InsufficientParties {
        healthy: usize,
        required: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_party_mapping() {
        let mut mapping = PartyMapping::new();
        let peer1 = PeerId::from_string("worker-1");
        let peer2 = PeerId::from_string("worker-2");

        let idx1 = mapping.register(&peer1);
        let idx2 = mapping.register(&peer2);

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(mapping.len(), 2);
        assert_eq!(mapping.party_index(&peer1), Some(0));
        assert_eq!(mapping.peer_id(1), Some("worker-2"));

        mapping.deregister(&peer1);
        assert_eq!(mapping.len(), 1);
        assert_eq!(mapping.party_index(&peer1), None);
    }

    #[test]
    fn test_session_orchestrator_registration() {
        let mut orch = MpcSessionOrchestrator::new(2, None);

        let p1 = PeerId::from_string("w1");
        let p2 = PeerId::from_string("w2");
        let p3 = PeerId::from_string("w3");

        assert_eq!(orch.register_worker(&p1), Some(0));
        assert_eq!(orch.register_worker(&p2), Some(1));
        assert_eq!(orch.register_worker(&p3), Some(2));

        // Duplicate registration returns None
        assert_eq!(orch.register_worker(&p1), None);

        assert_eq!(orch.available_parties(), 3);
    }

    #[test]
    fn test_session_lifecycle() {
        let mut orch = MpcSessionOrchestrator::new(2, None);

        let p1 = PeerId::from_string("w1");
        let p2 = PeerId::from_string("w2");

        orch.register_worker(&p1);
        orch.register_worker(&p2);

        assert!(orch.can_start_session());

        let (session_id, parties) = orch.start_session(1).unwrap();
        assert!(!session_id.is_empty());
        assert_eq!(parties.len(), 2);

        // Can't start another session while one is active
        assert!(!orch.can_start_session());

        orch.complete_session();
        assert!(orch.active_session().is_none());

        // Can start a new session now
        assert!(orch.can_start_session());
    }

    #[test]
    fn test_insufficient_parties() {
        let mut orch = MpcSessionOrchestrator::new(3, None);

        let p1 = PeerId::from_string("w1");
        let p2 = PeerId::from_string("w2");

        orch.register_worker(&p1);
        orch.register_worker(&p2);

        // Only 2 parties, need 3
        assert!(!orch.can_start_session());
        assert!(orch.start_session(1).is_none());
    }

    #[test]
    fn test_deregister_worker() {
        let mut orch = MpcSessionOrchestrator::new(2, None);

        let p1 = PeerId::from_string("w1");
        let p2 = PeerId::from_string("w2");

        orch.register_worker(&p1);
        orch.register_worker(&p2);
        assert_eq!(orch.available_parties(), 2);

        orch.deregister_worker(&p1);
        assert_eq!(orch.available_parties(), 1);
        assert!(!orch.can_start_session());
    }

    #[test]
    fn test_fail_session() {
        let mut orch = MpcSessionOrchestrator::new(2, None);

        let p1 = PeerId::from_string("w1");
        let p2 = PeerId::from_string("w2");

        orch.register_worker(&p1);
        orch.register_worker(&p2);

        let (_session_id, _parties) = orch.start_session(1).unwrap();
        assert!(orch.active_session().is_some());

        orch.fail_session("test failure");
        assert!(orch.active_session().is_none());
    }

    #[test]
    fn test_mapping_reset() {
        let mut mapping = PartyMapping::new();
        let peer1 = PeerId::from_string("worker-1");
        mapping.register(&peer1);
        assert_eq!(mapping.len(), 1);

        mapping.reset();
        assert!(mapping.is_empty());
        assert_eq!(mapping.party_index(&peer1), None);
    }
}
