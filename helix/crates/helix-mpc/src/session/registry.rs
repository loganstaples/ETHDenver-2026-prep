//! Dynamic party registration and discovery for distributed MPC sessions.
//!
//! The [`PartyRegistry`] bridges node-level peer discovery (mDNS, gossip, manual
//! bootstrap) with MPC session formation. When a node discovers a peer that is
//! MPC-capable, it registers the peer here. When a new MPC session needs to be
//! formed, the registry provides a set of available parties.
//!
//! # Lifecycle
//!
//! 1. Aggregator creates a `PartyRegistry`.
//! 2. As peers are discovered via the node's network layer, they call
//!    [`PartyRegistry::register`].
//! 3. When an MPC session is needed, the aggregator calls
//!    [`PartyRegistry::select_parties`] to get a quorum.
//! 4. Parties that disconnect or fail health checks are removed via
//!    [`PartyRegistry::deregister`] or automatically via TTL expiry.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

/// Capabilities a party advertises during registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartyCapabilities {
    /// Whether this party can perform training computations.
    pub can_train: bool,
    /// Whether this party can generate Beaver triples.
    pub can_generate_triples: bool,
    /// Maximum model size (in parameters) this party can handle.
    pub max_model_params: u64,
    /// Available memory in bytes.
    pub available_memory: u64,
    /// Protocol version supported.
    pub protocol_version: u32,
}

impl Default for PartyCapabilities {
    fn default() -> Self {
        Self {
            can_train: true,
            can_generate_triples: true,
            max_model_params: 1_000_000,
            available_memory: 1_073_741_824, // 1 GB
            protocol_version: 1,
        }
    }
}

/// Status of a registered party.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartyStatus {
    /// Party is available for new sessions.
    Available,
    /// Party is currently in an active MPC session.
    InSession,
    /// Party failed a health check and is temporarily unavailable.
    Unhealthy,
    /// Party is being drained (will not accept new sessions).
    Draining,
}

/// A registered party in the registry.
#[derive(Debug, Clone)]
pub struct RegisteredParty {
    /// The party's ID.
    pub party_id: PartyId,
    /// Network address for MPC transport.
    pub addr: SocketAddr,
    /// Node peer ID (from the node's peer discovery, if available).
    pub node_peer_id: Option<String>,
    /// Advertised capabilities.
    pub capabilities: PartyCapabilities,
    /// Current status.
    pub status: PartyStatus,
    /// When this party was first registered.
    pub registered_at: Instant,
    /// Last heartbeat or activity timestamp.
    pub last_seen: Instant,
    /// Current session ID, if in a session.
    pub current_session: Option<String>,
    /// Cumulative reputation score (higher is better).
    pub reputation: f64,
}

/// Configuration for the party registry.
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// How long a party can be idle before being considered stale.
    pub party_ttl: Duration,
    /// Minimum reputation to be eligible for session selection.
    pub min_reputation: f64,
    /// Maximum number of registered parties.
    pub max_parties: usize,
    /// How often to run the cleanup sweep.
    pub cleanup_interval: Duration,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            party_ttl: Duration::from_secs(120),
            min_reputation: 0.0,
            max_parties: 256,
            cleanup_interval: Duration::from_secs(30),
        }
    }
}

/// Criteria for selecting parties for an MPC session.
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    /// Number of parties needed.
    pub num_parties: usize,
    /// Minimum threshold for t-of-n.
    pub threshold: usize,
    /// Minimum protocol version required.
    pub min_protocol_version: u32,
    /// Required capabilities.
    pub require_training: bool,
    /// Minimum reputation.
    pub min_reputation: f64,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            num_parties: 3,
            threshold: 2,
            min_protocol_version: 1,
            require_training: true,
            min_reputation: 0.0,
        }
    }
}

/// Dynamic party registry for MPC session formation.
///
/// Thread-safe — all methods take `&self` and use internal locking.
pub struct PartyRegistry {
    parties: Arc<RwLock<HashMap<PartyId, RegisteredParty>>>,
    config: RegistryConfig,
}

impl PartyRegistry {
    /// Creates a new party registry.
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            parties: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Registers a new party or updates an existing one.
    pub fn register(
        &self,
        party_id: PartyId,
        addr: SocketAddr,
        capabilities: PartyCapabilities,
        node_peer_id: Option<String>,
    ) -> MPCResult<()> {
        let mut parties = self.parties.write();

        if parties.len() >= self.config.max_parties && !parties.contains_key(&party_id) {
            return Err(MPCError::SessionError(format!(
                "registry full: {} parties registered (max {})",
                parties.len(),
                self.config.max_parties,
            )));
        }

        let now = Instant::now();

        if let Some(existing) = parties.get_mut(&party_id) {
            // Update existing registration.
            existing.addr = addr;
            existing.capabilities = capabilities;
            existing.last_seen = now;
            if existing.status == PartyStatus::Unhealthy {
                existing.status = PartyStatus::Available;
            }
            if node_peer_id.is_some() {
                existing.node_peer_id = node_peer_id;
            }
        } else {
            parties.insert(
                party_id.clone(),
                RegisteredParty {
                    party_id,
                    addr,
                    node_peer_id,
                    capabilities,
                    status: PartyStatus::Available,
                    registered_at: now,
                    last_seen: now,
                    current_session: None,
                    reputation: 100.0,
                },
            );
        }

        Ok(())
    }

    /// Removes a party from the registry.
    pub fn deregister(&self, party_id: &PartyId) -> bool {
        self.parties.write().remove(party_id).is_some()
    }

    /// Updates a party's heartbeat timestamp.
    pub fn heartbeat(&self, party_id: &PartyId) -> bool {
        if let Some(party) = self.parties.write().get_mut(party_id) {
            party.last_seen = Instant::now();
            true
        } else {
            false
        }
    }

    /// Marks a party as assigned to a session.
    pub fn assign_to_session(&self, party_id: &PartyId, session_id: &str) -> MPCResult<()> {
        let mut parties = self.parties.write();
        let party = parties.get_mut(party_id).ok_or_else(|| {
            MPCError::UnknownParty(party_id.clone())
        })?;

        if party.status != PartyStatus::Available {
            return Err(MPCError::SessionError(format!(
                "party {} is not available (status: {:?})",
                party_id, party.status,
            )));
        }

        party.status = PartyStatus::InSession;
        party.current_session = Some(session_id.to_string());
        Ok(())
    }

    /// Releases a party from its current session.
    pub fn release_from_session(&self, party_id: &PartyId) -> MPCResult<()> {
        let mut parties = self.parties.write();
        let party = parties.get_mut(party_id).ok_or_else(|| {
            MPCError::UnknownParty(party_id.clone())
        })?;

        party.status = PartyStatus::Available;
        party.current_session = None;
        Ok(())
    }

    /// Marks a party as unhealthy.
    pub fn mark_unhealthy(&self, party_id: &PartyId, penalty: f64) {
        if let Some(party) = self.parties.write().get_mut(party_id) {
            party.status = PartyStatus::Unhealthy;
            party.reputation = (party.reputation - penalty).max(0.0);
        }
    }

    /// Adjusts a party's reputation.
    pub fn adjust_reputation(&self, party_id: &PartyId, delta: f64) {
        if let Some(party) = self.parties.write().get_mut(party_id) {
            party.reputation = (party.reputation + delta).max(0.0).min(1000.0);
        }
    }

    /// Selects parties for a new MPC session based on the given criteria.
    ///
    /// Returns a vector of `(PartyId, SocketAddr)` sorted by reputation
    /// (highest first). Returns an error if not enough eligible parties exist.
    pub fn select_parties(
        &self,
        criteria: &SelectionCriteria,
    ) -> MPCResult<Vec<(PartyId, SocketAddr)>> {
        let parties = self.parties.read();

        let mut eligible: Vec<&RegisteredParty> = parties
            .values()
            .filter(|p| {
                p.status == PartyStatus::Available
                    && p.reputation >= criteria.min_reputation
                    && p.capabilities.protocol_version >= criteria.min_protocol_version
                    && (!criteria.require_training || p.capabilities.can_train)
                    && p.last_seen.elapsed() < self.config.party_ttl
            })
            .collect();

        if eligible.len() < criteria.num_parties {
            return Err(MPCError::InsufficientParties {
                required: criteria.num_parties,
                available: eligible.len(),
            });
        }

        // Sort by reputation (highest first), then by registration time (oldest first).
        eligible.sort_by(|a, b| {
            b.reputation
                .partial_cmp(&a.reputation)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.registered_at.cmp(&b.registered_at))
        });

        Ok(eligible
            .iter()
            .take(criteria.num_parties)
            .map(|p| (p.party_id.clone(), p.addr))
            .collect())
    }

    /// Removes stale parties that haven't been seen within the TTL.
    pub fn cleanup_stale(&self) -> Vec<PartyId> {
        let mut parties = self.parties.write();
        let ttl = self.config.party_ttl;
        let stale: Vec<PartyId> = parties
            .values()
            .filter(|p| p.last_seen.elapsed() > ttl && p.status != PartyStatus::InSession)
            .map(|p| p.party_id.clone())
            .collect();

        for id in &stale {
            parties.remove(id);
        }

        stale
    }

    /// Returns the number of registered parties.
    pub fn count(&self) -> usize {
        self.parties.read().len()
    }

    /// Returns the number of available parties.
    pub fn available_count(&self) -> usize {
        self.parties
            .read()
            .values()
            .filter(|p| p.status == PartyStatus::Available)
            .count()
    }

    /// Returns a snapshot of all registered parties (for diagnostics).
    pub fn snapshot(&self) -> Vec<RegisteredParty> {
        self.parties.read().values().cloned().collect()
    }

    /// Gets info for a specific party.
    pub fn get(&self, party_id: &PartyId) -> Option<RegisteredParty> {
        self.parties.read().get(party_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_addr(port: u16) -> SocketAddr {
        use std::net::{IpAddr, Ipv4Addr};
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
    }

    #[test]
    fn test_register_and_count() {
        let registry = PartyRegistry::new(RegistryConfig::default());

        registry
            .register(
                PartyId::from_index(0),
                test_addr(9000),
                PartyCapabilities::default(),
                None,
            )
            .unwrap();

        registry
            .register(
                PartyId::from_index(1),
                test_addr(9001),
                PartyCapabilities::default(),
                Some("peer-abc".into()),
            )
            .unwrap();

        assert_eq!(registry.count(), 2);
        assert_eq!(registry.available_count(), 2);
    }

    #[test]
    fn test_deregister() {
        let registry = PartyRegistry::new(RegistryConfig::default());
        let pid = PartyId::from_index(0);

        registry
            .register(pid.clone(), test_addr(9000), PartyCapabilities::default(), None)
            .unwrap();
        assert_eq!(registry.count(), 1);

        assert!(registry.deregister(&pid));
        assert_eq!(registry.count(), 0);
        assert!(!registry.deregister(&pid)); // Already gone.
    }

    #[test]
    fn test_heartbeat() {
        let registry = PartyRegistry::new(RegistryConfig::default());
        let pid = PartyId::from_index(0);

        assert!(!registry.heartbeat(&pid)); // Not registered yet.

        registry
            .register(pid.clone(), test_addr(9000), PartyCapabilities::default(), None)
            .unwrap();

        assert!(registry.heartbeat(&pid));
    }

    #[test]
    fn test_select_parties() {
        let registry = PartyRegistry::new(RegistryConfig::default());

        for i in 0..5 {
            registry
                .register(
                    PartyId::from_index(i),
                    test_addr(9000 + i as u16),
                    PartyCapabilities::default(),
                    None,
                )
                .unwrap();
        }

        let selected = registry
            .select_parties(&SelectionCriteria {
                num_parties: 3,
                ..Default::default()
            })
            .unwrap();

        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_select_insufficient() {
        let registry = PartyRegistry::new(RegistryConfig::default());

        registry
            .register(
                PartyId::from_index(0),
                test_addr(9000),
                PartyCapabilities::default(),
                None,
            )
            .unwrap();

        let result = registry.select_parties(&SelectionCriteria {
            num_parties: 3,
            ..Default::default()
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_session_assignment() {
        let registry = PartyRegistry::new(RegistryConfig::default());
        let pid = PartyId::from_index(0);

        registry
            .register(pid.clone(), test_addr(9000), PartyCapabilities::default(), None)
            .unwrap();

        registry.assign_to_session(&pid, "session-1").unwrap();

        // Should not be available for another session.
        let result = registry.select_parties(&SelectionCriteria {
            num_parties: 1,
            ..Default::default()
        });
        assert!(result.is_err());

        // Release and re-select.
        registry.release_from_session(&pid).unwrap();
        let selected = registry
            .select_parties(&SelectionCriteria {
                num_parties: 1,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_reputation_filtering() {
        let registry = PartyRegistry::new(RegistryConfig::default());
        let pid = PartyId::from_index(0);

        registry
            .register(pid.clone(), test_addr(9000), PartyCapabilities::default(), None)
            .unwrap();

        // Tank reputation.
        registry.mark_unhealthy(&pid, 200.0);

        // Not selectable even after recovery if min_reputation is high.
        // First make it available again.
        registry.register(pid.clone(), test_addr(9000), PartyCapabilities::default(), None).unwrap();

        let result = registry.select_parties(&SelectionCriteria {
            num_parties: 1,
            min_reputation: 50.0,
            ..Default::default()
        });
        // Reputation was 100 - 200 = 0 (clamped), so it's below 50.
        assert!(result.is_err());
    }

    #[test]
    fn test_cleanup_stale() {
        let config = RegistryConfig {
            party_ttl: Duration::from_millis(1),
            ..Default::default()
        };
        let registry = PartyRegistry::new(config);

        registry
            .register(
                PartyId::from_index(0),
                test_addr(9000),
                PartyCapabilities::default(),
                None,
            )
            .unwrap();

        std::thread::sleep(Duration::from_millis(5));

        let stale = registry.cleanup_stale();
        assert_eq!(stale.len(), 1);
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_max_parties() {
        let config = RegistryConfig {
            max_parties: 2,
            ..Default::default()
        };
        let registry = PartyRegistry::new(config);

        registry
            .register(PartyId::from_index(0), test_addr(9000), PartyCapabilities::default(), None)
            .unwrap();
        registry
            .register(PartyId::from_index(1), test_addr(9001), PartyCapabilities::default(), None)
            .unwrap();

        // Third should fail.
        let result = registry.register(
            PartyId::from_index(2),
            test_addr(9002),
            PartyCapabilities::default(),
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_update_existing_party() {
        let registry = PartyRegistry::new(RegistryConfig::default());
        let pid = PartyId::from_index(0);

        registry
            .register(pid.clone(), test_addr(9000), PartyCapabilities::default(), None)
            .unwrap();

        // Re-register with different address (update).
        registry
            .register(pid.clone(), test_addr(9999), PartyCapabilities::default(), None)
            .unwrap();

        assert_eq!(registry.count(), 1); // Still just one.
        let info = registry.get(&pid).unwrap();
        assert_eq!(info.addr.port(), 9999);
    }
}
