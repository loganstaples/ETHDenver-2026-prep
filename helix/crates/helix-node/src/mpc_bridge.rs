//! MPC ↔ Node P2P Bridge.
//!
//! Bridges the MPC transport layer with the node's P2P networking layer so that
//! MPC sessions communicate through already-established peer connections instead
//! of creating their own TCP connections.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Node                                                           │
//! │  ┌──────────────┐   ┌──────────────────────────────────────┐   │
//! │  │ ConnectionPool│──▶│ NodeMpcBridge                        │   │
//! │  │ (TCP/TLS)     │   │  ├─ send: wraps in MpcDataMessage   │   │
//! │  │               │◀──│  ├─ recv: routes from event loop    │   │
//! │  └──────────────┘   │  └─ sessions: per-session channels   │   │
//! │                      └──────────┬───────────────────────────┘   │
//! │                                 │                               │
//! │  ┌──────────────────────────────▼──────────────────────────┐   │
//! │  │ NodeTransport (per MPC session, implements MPCTransport) │   │
//! │  │  - session_id tagging via TaggedMessage                  │   │
//! │  │  - party_id ↔ peer_id mapping via PeerMapping            │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! 1. Create a `NodeMpcBridge` with a reference to the node's `ConnectionPool`
//! 2. When starting an MPC session, call `create_session()` to get `NodeTransport`
//!    instances for each party
//! 3. Feed incoming `MpcDataMessage` events to `handle_incoming()` — the bridge
//!    routes them to the correct session's transport channels
//! 4. When the session completes, call `remove_session()` to clean up

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::sync::Mutex as TokioMutex;

use helix_mpc::session::node_transport::{
    MPCMessageRouter, NodeTransport, PeerMapping, TaggedMessage,
};
use helix_mpc::types::PartyId;

use crate::network::messages::{
    MpcDataMessage, MessagePayload, NetworkMessage, PeerId,
};
use crate::network::transport::ConnectionPool;

/// Bridge between the node's P2P network and MPC sessions.
///
/// Manages the lifecycle of MPC sessions over existing P2P connections:
/// - Creates per-session transport channels
/// - Routes outgoing MPC messages through the node's ConnectionPool
/// - Dispatches incoming MPC messages to the correct session
pub struct NodeMpcBridge {
    /// Our node's peer ID.
    local_peer_id: PeerId,
    /// Reference to the node's connection pool for sending messages.
    pool: Arc<ConnectionPool>,
    /// MPC message router — dispatches incoming messages to session channels.
    router: MPCMessageRouter,
    /// Per-session outbound pump handles.
    /// Each session has a background task that reads from outbound channels
    /// and sends via the connection pool.
    outbound_handles: TokioMutex<HashMap<String, Vec<tokio::task::JoinHandle<()>>>>,
    /// Per-session peer mapping (session_id → PeerMapping).
    session_mappings: TokioMutex<HashMap<String, PeerMapping>>,
}

impl NodeMpcBridge {
    /// Creates a new MPC bridge.
    pub fn new(local_peer_id: PeerId, pool: Arc<ConnectionPool>) -> Self {
        Self {
            local_peer_id,
            pool,
            router: MPCMessageRouter::new(),
            outbound_handles: TokioMutex::new(HashMap::new()),
            session_mappings: TokioMutex::new(HashMap::new()),
        }
    }

    /// Creates MPC transport instances for a new session.
    ///
    /// For each party in the session, this creates a `NodeTransport` backed by
    /// channels. Outbound messages are pumped to the `ConnectionPool` by
    /// background tasks. Inbound messages arrive via `handle_incoming()`.
    ///
    /// # Arguments
    ///
    /// * `session_id` - Unique session identifier
    /// * `our_party` - Our MPC party ID in this session
    /// * `party_peer_map` - Mapping of MPC party IDs to node peer IDs for all
    ///   parties in this session (including ourselves)
    ///
    /// # Returns
    ///
    /// A `NodeTransport` for our party, ready to use with MPC protocols.
    pub async fn create_session(
        &self,
        session_id: &str,
        our_party: &PartyId,
        party_peer_map: &[(PartyId, PeerId)],
    ) -> NodeTransport {
        let mut mapping = PeerMapping::new();
        let mut channels = Vec::new();
        // Channels the router will use to deliver incoming messages to our transport
        let mut router_senders: HashMap<String, mpsc::Sender<Vec<u8>>> = HashMap::new();
        let mut outbound_handles = Vec::new();

        for (party_id, peer_id) in party_peer_map {
            if party_id == our_party {
                continue; // Skip ourselves
            }

            mapping.add(party_id.clone(), peer_id.0.clone());

            // Outbound channel: our transport writes here, pump task reads and sends via pool
            let (outbound_tx, mut outbound_rx) = mpsc::channel::<Vec<u8>>(4096);
            // Inbound channel: router writes here, our transport reads
            let (inbound_tx, inbound_rx) = mpsc::channel::<Vec<u8>>(4096);

            // Register the inbound sender so the router can deliver messages from this party
            router_senders.insert(party_id.0.clone(), inbound_tx);

            // Spawn outbound pump: reads tagged messages from the channel,
            // wraps them in MpcDataMessage, and sends through the connection pool
            let pool = self.pool.clone();
            let target_peer = peer_id.clone();
            let sid = session_id.to_string();
            let local_peer = self.local_peer_id.clone();

            let handle = tokio::spawn(async move {
                while let Some(data) = outbound_rx.recv().await {
                    let mpc_msg = MpcDataMessage {
                        session_id: sid.clone(),
                        data,
                    };
                    let network_msg = NetworkMessage::new(
                        local_peer.clone(),
                        MessagePayload::MpcData(mpc_msg),
                    );
                    if let Err(e) = pool.send(&target_peer, network_msg).await {
                        tracing::error!(
                            "Failed to send MPC message to peer {}: {}",
                            target_peer, e
                        );
                    }
                }
            });
            outbound_handles.push(handle);

            channels.push((party_id.clone(), outbound_tx, inbound_rx));
        }

        // Register session channels with the router for incoming message dispatch
        self.router
            .register_session(session_id.to_string(), router_senders)
            .await;

        // Store the session's outbound handles and mapping
        self.outbound_handles
            .lock()
            .await
            .insert(session_id.to_string(), outbound_handles);
        self.session_mappings
            .lock()
            .await
            .insert(session_id.to_string(), mapping.clone());

        NodeTransport::new(
            our_party.clone(),
            session_id.to_string(),
            mapping,
            channels,
        )
    }

    /// Handles an incoming MPC message from the P2P network.
    ///
    /// Called by the node's event loop when it receives an `MpcDataMessage`.
    /// Routes the message to the correct session's transport channel.
    ///
    /// # Arguments
    ///
    /// * `from_peer` - The node peer ID that sent this message
    /// * `mpc_msg` - The MPC data message containing session_id and payload
    pub async fn handle_incoming(
        &self,
        from_peer: &PeerId,
        mpc_msg: &MpcDataMessage,
    ) -> Result<(), String> {
        // Look up which party this peer corresponds to in this session
        let mappings = self.session_mappings.lock().await;
        let mapping = mappings.get(&mpc_msg.session_id).ok_or_else(|| {
            format!(
                "No active MPC session '{}' for incoming message from peer {}",
                mpc_msg.session_id, from_peer
            )
        })?;

        let sender_party = mapping
            .party_for_peer(&from_peer.0)
            .ok_or_else(|| {
                format!(
                    "Unknown peer {} in MPC session '{}'",
                    from_peer, mpc_msg.session_id
                )
            })?
            .clone();
        drop(mappings);

        // The data in mpc_msg.data is the raw tagged message bytes.
        // Route it through the MPCMessageRouter to the correct session channel.
        self.router
            .route_message(&mpc_msg.data)
            .await
            .map_err(|e| format!("MPC router error: {}", e))
    }

    /// Removes a session and cleans up its resources.
    ///
    /// Aborts outbound pump tasks and removes router registrations.
    pub async fn remove_session(&self, session_id: &str) {
        // Remove from router
        self.router.remove_session(session_id).await;

        // Abort outbound pump tasks
        if let Some(handles) = self.outbound_handles.lock().await.remove(session_id) {
            for handle in handles {
                handle.abort();
            }
        }

        // Remove mapping
        self.session_mappings.lock().await.remove(session_id);

        tracing::info!("MPC session '{}' removed from bridge", session_id);
    }

    /// Returns the number of active sessions.
    pub async fn active_session_count(&self) -> usize {
        self.router.session_count().await
    }

    /// Returns a reference to the connection pool.
    pub fn pool(&self) -> &Arc<ConnectionPool> {
        &self.pool
    }

    /// Returns our local peer ID.
    pub fn local_peer_id(&self) -> &PeerId {
        &self.local_peer_id
    }
}

/// Creates a mesh of `NodeMpcBridge` + `NodeTransport` instances for testing.
///
/// Given N nodes (each with a peer ID and connection pool), this creates:
/// - One `NodeMpcBridge` per node
/// - One `NodeTransport` per node for the given session
///
/// All transports are wired together through the bridges so that MPC messages
/// flow over the nodes' P2P connections.
pub async fn create_bridge_mesh(
    session_id: &str,
    nodes: &[(PeerId, Arc<ConnectionPool>)],
) -> Vec<(NodeMpcBridge, NodeTransport)> {
    let n = nodes.len();

    // Assign MPC party IDs to each node
    let party_peer_map: Vec<(PartyId, PeerId)> = nodes
        .iter()
        .enumerate()
        .map(|(i, (peer_id, _))| (PartyId::from_index(i), peer_id.clone()))
        .collect();

    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let bridge = NodeMpcBridge::new(nodes[i].0.clone(), nodes[i].1.clone());
        let our_party = PartyId::from_index(i);
        let transport = bridge
            .create_session(session_id, &our_party, &party_peer_map)
            .await;
        results.push((bridge, transport));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use helix_mpc::session::transport::MPCTransport;

    #[tokio::test]
    async fn test_bridge_create_and_remove_session() {
        // Create a mock pool (we won't actually send over it in this test)
        let local_id = PeerId::from_string("node-0");
        let transport = Arc::new(
            crate::network::transport::TcpTransport::new(
                local_id.clone(),
                crate::network::transport::TransportConfig {
                    listen_addr: "127.0.0.1:0".parse().unwrap(),
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let pool = Arc::new(ConnectionPool::new(transport));

        let bridge = NodeMpcBridge::new(local_id, pool);

        // Create a session with 2 other parties
        let party_peer_map = vec![
            (PartyId::from_index(0), PeerId::from_string("node-0")),
            (PartyId::from_index(1), PeerId::from_string("node-1")),
            (PartyId::from_index(2), PeerId::from_string("node-2")),
        ];

        let transport = bridge
            .create_session("test-session", &PartyId::from_index(0), &party_peer_map)
            .await;

        assert_eq!(transport.party_id(), &PartyId::from_index(0));
        assert_eq!(transport.peers().len(), 2);
        assert_eq!(bridge.active_session_count().await, 1);

        bridge.remove_session("test-session").await;
        assert_eq!(bridge.active_session_count().await, 0);
    }

    #[tokio::test]
    async fn test_bridge_rejects_unknown_session() {
        let local_id = PeerId::from_string("node-0");
        let transport = Arc::new(
            crate::network::transport::TcpTransport::new(
                local_id.clone(),
                crate::network::transport::TransportConfig {
                    listen_addr: "127.0.0.1:0".parse().unwrap(),
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let pool = Arc::new(ConnectionPool::new(transport));

        let bridge = NodeMpcBridge::new(local_id, pool);

        let result = bridge
            .handle_incoming(
                &PeerId::from_string("node-1"),
                &MpcDataMessage {
                    session_id: "nonexistent".to_string(),
                    data: vec![1, 2, 3],
                },
            )
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No active MPC session"));
    }
}
