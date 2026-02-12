//! Node-integrated MPC transport.
//!
//! Bridges the node's peer discovery and TCP/TLS connections with the MPC
//! transport layer. Instead of MPC sessions establishing their own TCP
//! connections, [`NodeTransport`] reuses the node's existing connection pool.
//!
//! # Design
//!
//! The node's `ConnectionPool` manages TCP connections to all known peers.
//! `NodeTransport` wraps these connections with the [`MPCTransport`] trait,
//! adding MPC-specific framing and session multiplexing.
//!
//! Each MPC message is tagged with a `session_id` so multiple MPC sessions
//! can share the same underlying TCP connections.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │  Node (helix-node)                               │
//! │  ┌────────────┐  ┌────────────────────────────┐  │
//! │  │  Discovery  │  │  Connection Manager        │  │
//! │  │  (mDNS,     │──▶  (TCP/TLS pool)           │  │
//! │  │   gossip)   │  │                            │  │
//! │  └────────────┘  └──────┬─────────────────────┘  │
//! │                          │                        │
//! │  ┌──────────────────────▼───────────────────────┐│
//! │  │  NodeTransport (implements MPCTransport)      ││
//! │  │  - session_id tagging                         ││
//! │  │  - party_id ↔ node_peer_id mapping            ││
//! │  │  - message routing                            ││
//! │  └──────────────────────────────────────────────┘│
//! └──────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Mutex as TokioMutex};

use crate::error::{MPCError, MPCResult};
use crate::session::transport::MPCTransport;
use crate::types::PartyId;

/// An MPC message tagged with session ID for multiplexing over shared connections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggedMessage {
    /// Which MPC session this message belongs to.
    pub session_id: String,
    /// The sender's MPC party ID.
    pub sender: PartyId,
    /// The raw MPC payload.
    pub payload: Vec<u8>,
}

/// Maps between MPC party IDs and node peer IDs.
#[derive(Debug, Clone)]
pub struct PeerMapping {
    /// MPC party ID → node peer ID.
    party_to_peer: HashMap<PartyId, String>,
    /// Node peer ID → MPC party ID.
    peer_to_party: HashMap<String, PartyId>,
}

impl PeerMapping {
    pub fn new() -> Self {
        Self {
            party_to_peer: HashMap::new(),
            peer_to_party: HashMap::new(),
        }
    }

    /// Adds a mapping between party and peer IDs.
    pub fn add(&mut self, party_id: PartyId, peer_id: String) {
        self.party_to_peer.insert(party_id.clone(), peer_id.clone());
        self.peer_to_party.insert(peer_id, party_id);
    }

    /// Looks up the node peer ID for an MPC party.
    pub fn peer_for_party(&self, party_id: &PartyId) -> Option<&String> {
        self.party_to_peer.get(party_id)
    }

    /// Looks up the MPC party ID for a node peer.
    pub fn party_for_peer(&self, peer_id: &str) -> Option<&PartyId> {
        self.peer_to_party.get(peer_id)
    }
}

/// Trait for sending/receiving raw bytes to/from node peers.
///
/// This is the bridge interface that the node's connection manager implements.
/// `NodeTransport` uses this to send/receive MPC messages through the node's
/// existing connections.
#[async_trait]
pub trait NodeConnectionBridge: Send + Sync {
    /// Sends raw bytes to a node peer.
    async fn send_to_peer(&self, peer_id: &str, data: &[u8]) -> Result<(), String>;

    /// Returns a receiver for incoming messages from a specific peer.
    /// The node's receive loop routes MPC-tagged messages here.
    async fn recv_from_peer(&self, peer_id: &str) -> Result<Vec<u8>, String>;
}

/// MPC transport that routes through the node's network layer.
///
/// Implements [`MPCTransport`] by mapping MPC party IDs to node peer IDs
/// and sending/receiving through the node's connection pool.
pub struct NodeTransport {
    /// Our MPC party ID.
    party: PartyId,
    /// List of peer party IDs.
    peers_list: Vec<PartyId>,
    /// Session ID for message tagging.
    session_id: String,
    /// Party ↔ peer mapping.
    mapping: PeerMapping,
    /// Per-peer outgoing channels (party_id → sender).
    senders: HashMap<String, mpsc::Sender<Vec<u8>>>,
    /// Per-peer incoming channels (party_id → receiver).
    receivers: HashMap<String, TokioMutex<mpsc::Receiver<Vec<u8>>>>,
}

impl NodeTransport {
    /// Creates a new NodeTransport backed by message channels.
    ///
    /// The caller (typically the node's MPC integration layer) is responsible
    /// for connecting these channels to the actual node connections.
    ///
    /// # Arguments
    ///
    /// * `party` - Our MPC party ID
    /// * `session_id` - MPC session identifier
    /// * `mapping` - Party ↔ node peer mapping
    /// * `channels` - Pre-created channels for each peer `(party_id, tx, rx)`
    pub fn new(
        party: PartyId,
        session_id: String,
        mapping: PeerMapping,
        channels: Vec<(PartyId, mpsc::Sender<Vec<u8>>, mpsc::Receiver<Vec<u8>>)>,
    ) -> Self {
        let mut peers_list = Vec::new();
        let mut senders = HashMap::new();
        let mut receivers = HashMap::new();

        for (peer_party, tx, rx) in channels {
            peers_list.push(peer_party.clone());
            senders.insert(peer_party.0.clone(), tx);
            receivers.insert(peer_party.0.clone(), TokioMutex::new(rx));
        }

        peers_list.sort();

        Self {
            party,
            peers_list,
            session_id,
            mapping,
            senders,
            receivers,
        }
    }

    /// Creates a pair of NodeTransports connected by in-memory channels.
    ///
    /// Useful for testing the node transport interface without a real node.
    pub fn create_pair(
        party_a: PartyId,
        party_b: PartyId,
        session_id: &str,
    ) -> (Self, Self) {
        let (a_to_b_tx, a_to_b_rx) = mpsc::channel(4096);
        let (b_to_a_tx, b_to_a_rx) = mpsc::channel(4096);

        let mut mapping_a = PeerMapping::new();
        mapping_a.add(party_b.clone(), format!("peer-{}", party_b));

        let mut mapping_b = PeerMapping::new();
        mapping_b.add(party_a.clone(), format!("peer-{}", party_a));

        let transport_a = Self::new(
            party_a.clone(),
            session_id.to_string(),
            mapping_a,
            vec![(party_b.clone(), a_to_b_tx, b_to_a_rx)],
        );

        let transport_b = Self::new(
            party_b,
            session_id.to_string(),
            mapping_b,
            vec![(party_a, b_to_a_tx, a_to_b_rx)],
        );

        (transport_a, transport_b)
    }

    /// Creates a mesh of NodeTransports for N parties, connected by channels.
    ///
    /// Returns one transport per party, all wired together.
    pub fn create_mesh(parties: &[PartyId], session_id: &str) -> Vec<Self> {
        let n = parties.len();

        // Create channels for each pair.
        let mut tx_map: HashMap<(usize, usize), mpsc::Sender<Vec<u8>>> = HashMap::new();
        let mut rx_map: HashMap<(usize, usize), mpsc::Receiver<Vec<u8>>> = HashMap::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let (tx, rx) = mpsc::channel(4096);
                    tx_map.insert((i, j), tx);
                    rx_map.insert((i, j), rx);
                }
            }
        }

        let mut transports = Vec::with_capacity(n);

        for i in 0..n {
            let mut mapping = PeerMapping::new();
            let mut channels = Vec::new();

            for j in 0..n {
                if i != j {
                    mapping.add(parties[j].clone(), format!("peer-{}", parties[j]));

                    let tx = tx_map.remove(&(i, j)).unwrap();
                    let rx = rx_map.remove(&(j, i)).unwrap();
                    channels.push((parties[j].clone(), tx, rx));
                }
            }

            transports.push(Self::new(
                parties[i].clone(),
                session_id.to_string(),
                mapping,
                channels,
            ));
        }

        transports
    }

    /// Returns the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the peer mapping.
    pub fn mapping(&self) -> &PeerMapping {
        &self.mapping
    }
}

#[async_trait]
impl MPCTransport for NodeTransport {
    async fn send(&self, party: &PartyId, msg: &[u8]) -> MPCResult<()> {
        let tx = self.senders.get(&party.0).ok_or_else(|| {
            MPCError::CommunicationError(format!(
                "no channel to party {} in session {}",
                party, self.session_id
            ))
        })?;

        // Tag the message with session info for multiplexing.
        let tagged = TaggedMessage {
            session_id: self.session_id.clone(),
            sender: self.party.clone(),
            payload: msg.to_vec(),
        };

        let envelope = bincode::serialize(&tagged).map_err(|e| {
            MPCError::CommunicationError(format!("failed to serialize tagged message: {}", e))
        })?;

        tx.send(envelope).await.map_err(|e| {
            MPCError::CommunicationError(format!(
                "send to {} failed in session {}: {}",
                party, self.session_id, e
            ))
        })
    }

    async fn recv(&self, party: &PartyId) -> MPCResult<Vec<u8>> {
        let rx_mutex = self.receivers.get(&party.0).ok_or_else(|| {
            MPCError::CommunicationError(format!(
                "no channel from party {} in session {}",
                party, self.session_id
            ))
        })?;

        let mut rx = rx_mutex.lock().await;
        let envelope = rx.recv().await.ok_or_else(|| {
            MPCError::CommunicationError(format!(
                "channel from {} closed in session {}",
                party, self.session_id
            ))
        })?;

        let tagged: TaggedMessage = bincode::deserialize(&envelope).map_err(|e| {
            MPCError::CommunicationError(format!(
                "failed to deserialize tagged message: {}",
                e
            ))
        })?;

        if tagged.session_id != self.session_id {
            return Err(MPCError::CommunicationError(format!(
                "session mismatch: expected {}, got {}",
                self.session_id, tagged.session_id
            )));
        }

        Ok(tagged.payload)
    }

    async fn broadcast(&self, msg: &[u8]) -> MPCResult<()> {
        for peer in &self.peers_list {
            self.send(peer, msg).await?;
        }
        Ok(())
    }

    fn party_id(&self) -> &PartyId {
        &self.party
    }

    fn peers(&self) -> Vec<PartyId> {
        self.peers_list.clone()
    }
}

/// Incoming MPC message router.
///
/// The node's receive loop calls `route_message()` when it receives a message
/// tagged as MPC. The router dispatches to the correct session's transport.
pub struct MPCMessageRouter {
    /// Session ID → per-peer sender channels.
    sessions: Arc<TokioMutex<HashMap<String, HashMap<String, mpsc::Sender<Vec<u8>>>>>>,
}

impl MPCMessageRouter {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(TokioMutex::new(HashMap::new())),
        }
    }

    /// Registers sender channels for a session.
    pub async fn register_session(
        &self,
        session_id: String,
        party_senders: HashMap<String, mpsc::Sender<Vec<u8>>>,
    ) {
        self.sessions
            .lock()
            .await
            .insert(session_id, party_senders);
    }

    /// Removes a session from the router.
    pub async fn remove_session(&self, session_id: &str) {
        self.sessions.lock().await.remove(session_id);
    }

    /// Routes an incoming message to the correct session and party channel.
    pub async fn route_message(&self, raw: &[u8]) -> MPCResult<()> {
        let tagged: TaggedMessage = bincode::deserialize(raw).map_err(|e| {
            MPCError::CommunicationError(format!("failed to deserialize MPC message: {}", e))
        })?;

        let sessions = self.sessions.lock().await;
        let session_channels = sessions.get(&tagged.session_id).ok_or_else(|| {
            MPCError::SessionError(format!(
                "no active session {} for incoming MPC message",
                tagged.session_id
            ))
        })?;

        let sender_key = tagged.sender.0.clone();
        let tx = session_channels.get(&sender_key).ok_or_else(|| {
            MPCError::CommunicationError(format!(
                "no channel for party {} in session {}",
                tagged.sender, tagged.session_id
            ))
        })?;

        tx.send(raw.to_vec()).await.map_err(|e| {
            MPCError::CommunicationError(format!(
                "failed to route message from {} in session {}: {}",
                tagged.sender, tagged.session_id, e
            ))
        })
    }

    /// Returns the number of active sessions.
    pub async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[tokio::test]
    async fn test_node_transport_pair() {
        let pa = PartyId::from_index(0);
        let pb = PartyId::from_index(1);

        let (ta, tb) = NodeTransport::create_pair(pa.clone(), pb.clone(), "test-session");

        ta.send(&pb, b"hello from a").await.unwrap();
        let msg = tb.recv(&pa).await.unwrap();
        assert_eq!(msg, b"hello from a");

        tb.send(&pa, b"hello from b").await.unwrap();
        let msg = ta.recv(&pb).await.unwrap();
        assert_eq!(msg, b"hello from b");
    }

    #[tokio::test]
    async fn test_node_transport_mesh() {
        let parties = test_parties(3);
        let transports = NodeTransport::create_mesh(&parties, "mesh-session");

        // Party 0 broadcasts.
        transports[0].broadcast(b"broadcast").await.unwrap();

        // Parties 1 and 2 receive.
        let msg1 = transports[1].recv(&parties[0]).await.unwrap();
        let msg2 = transports[2].recv(&parties[0]).await.unwrap();
        assert_eq!(msg1, b"broadcast");
        assert_eq!(msg2, b"broadcast");
    }

    #[tokio::test]
    async fn test_node_transport_session_id() {
        let pa = PartyId::from_index(0);
        let pb = PartyId::from_index(1);

        let (ta, _tb) = NodeTransport::create_pair(pa.clone(), pb.clone(), "session-1");
        assert_eq!(ta.session_id(), "session-1");
    }

    #[tokio::test]
    async fn test_peer_mapping() {
        let mut mapping = PeerMapping::new();
        let party = PartyId::from_index(0);

        mapping.add(party.clone(), "peer-abc".to_string());

        assert_eq!(mapping.peer_for_party(&party), Some(&"peer-abc".to_string()));
        assert_eq!(mapping.party_for_peer("peer-abc"), Some(&party));
        assert_eq!(mapping.party_for_peer("unknown"), None);
    }

    #[tokio::test]
    async fn test_message_router() {
        let router = MPCMessageRouter::new();

        let (tx, mut rx) = mpsc::channel(100);
        let mut senders = HashMap::new();
        senders.insert("party-0".to_string(), tx);

        router
            .register_session("session-1".to_string(), senders)
            .await;

        assert_eq!(router.session_count().await, 1);

        // Create a tagged message.
        let tagged = TaggedMessage {
            session_id: "session-1".to_string(),
            sender: PartyId::from_index(0),
            payload: b"test payload".to_vec(),
        };
        let raw = bincode::serialize(&tagged).unwrap();

        router.route_message(&raw).await.unwrap();

        let received = rx.recv().await.unwrap();
        assert!(!received.is_empty());

        // Remove session.
        router.remove_session("session-1").await;
        assert_eq!(router.session_count().await, 0);
    }

    #[tokio::test]
    async fn test_router_unknown_session() {
        let router = MPCMessageRouter::new();

        let tagged = TaggedMessage {
            session_id: "nonexistent".to_string(),
            sender: PartyId::from_index(0),
            payload: b"test".to_vec(),
        };
        let raw = bincode::serialize(&tagged).unwrap();

        let result = router.route_message(&raw).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_node_transport_peers() {
        let parties = test_parties(3);
        let transports = NodeTransport::create_mesh(&parties, "test");

        assert_eq!(transports[0].party_id(), &parties[0]);
        assert_eq!(transports[0].peers().len(), 2);
    }
}
