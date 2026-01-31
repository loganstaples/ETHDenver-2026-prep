//! mDNS-based Peer Discovery for HELIX Network.
//!
//! Provides automatic peer discovery on local networks using multicast DNS.
//! This is intended for local demos and development; production should use DHT.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use parking_lot::RwLock;
use tokio::sync::mpsc;

use super::messages::{NodeCapabilities, PeerId, PeerInfo};

/// Service type for HELIX nodes.
const HELIX_SERVICE_TYPE: &str = "_helix._tcp.local.";

/// mDNS discovery configuration.
#[derive(Debug, Clone)]
pub struct MdnsConfig {
    /// Service name (unique per node).
    pub service_name: String,
    /// Port to advertise.
    pub port: u16,
    /// Time to live for announcements.
    pub ttl_secs: u32,
    /// Discovery interval.
    pub discovery_interval_secs: u64,
    /// Enable discovery.
    pub enabled: bool,
}

impl Default for MdnsConfig {
    fn default() -> Self {
        Self {
            service_name: format!("helix-node-{}", uuid::Uuid::new_v4()),
            port: 9000,
            ttl_secs: 300,
            discovery_interval_secs: 30,
            enabled: true,
        }
    }
}

/// Event from mDNS discovery.
#[derive(Debug, Clone)]
pub enum MdnsEvent {
    /// New peer discovered.
    Discovered(PeerInfo),
    /// Peer updated.
    Updated(PeerInfo),
    /// Peer disappeared.
    Lost(PeerId),
}

/// mDNS-based peer discovery.
pub struct MdnsDiscovery {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: MdnsConfig,
    /// Discovered peers.
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
    /// Event sender.
    event_tx: Option<mpsc::Sender<MdnsEvent>>,
    /// Running state.
    running: Arc<std::sync::atomic::AtomicBool>,
    /// Our capabilities.
    capabilities: NodeCapabilities,
}

impl MdnsDiscovery {
    /// Creates a new mDNS discovery instance.
    pub fn new(local_id: PeerId, config: MdnsConfig, capabilities: NodeCapabilities) -> Self {
        Self {
            local_id,
            config,
            peers: Arc::new(RwLock::new(HashMap::new())),
            event_tx: None,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            capabilities,
        }
    }

    /// Starts mDNS discovery and returns an event receiver.
    pub fn start(&mut self) -> Result<mpsc::Receiver<MdnsEvent>, MdnsError> {
        if !self.config.enabled {
            return Err(MdnsError::Disabled);
        }

        let (event_tx, event_rx) = mpsc::channel(1000);
        self.event_tx = Some(event_tx.clone());
        self.running.store(true, std::sync::atomic::Ordering::SeqCst);

        // Create mDNS daemon
        let mdns = ServiceDaemon::new()
            .map_err(|e| MdnsError::DaemonFailed(e.to_string()))?;

        // Register our service
        let service_info = self.create_service_info()?;
        mdns.register(service_info)
            .map_err(|e| MdnsError::RegistrationFailed(e.to_string()))?;

        // Start browse
        let receiver = mdns.browse(HELIX_SERVICE_TYPE)
            .map_err(|e| MdnsError::BrowseFailed(e.to_string()))?;

        let peers = self.peers.clone();
        let running = self.running.clone();
        let local_id = self.local_id.clone();
        let service_name = self.config.service_name.clone();

        // Spawn discovery loop
        tokio::spawn(async move {
            Self::discovery_loop(receiver, peers, event_tx, running, local_id, service_name).await;
        });

        Ok(event_rx)
    }

    /// Stops mDNS discovery.
    pub fn stop(&self) {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Returns all discovered peers.
    pub fn get_peers(&self) -> Vec<PeerInfo> {
        self.peers.read().values().cloned().collect()
    }

    /// Returns a specific peer by ID.
    pub fn get_peer(&self, peer_id: &PeerId) -> Option<PeerInfo> {
        self.peers.read()
            .values()
            .find(|p| p.id == *peer_id)
            .cloned()
    }

    fn create_service_info(&self) -> Result<ServiceInfo, MdnsError> {
        let hostname = hostname::get()
            .map_err(|e| MdnsError::HostnameFailed(e.to_string()))?
            .to_string_lossy()
            .to_string();

        let _full_name = format!("{}.{}", self.config.service_name, HELIX_SERVICE_TYPE);

        let mut properties: HashMap<String, String> = HashMap::new();
        properties.insert("peer_id".to_string(), self.local_id.0.clone());
        properties.insert("can_train".to_string(), self.capabilities.can_train.to_string());
        properties.insert("can_aggregate".to_string(), self.capabilities.can_aggregate.to_string());
        properties.insert("can_prove".to_string(), self.capabilities.can_prove.to_string());
        properties.insert("gpu_memory_mb".to_string(), self.capabilities.gpu_memory_mb.to_string());

        ServiceInfo::new(
            HELIX_SERVICE_TYPE,
            &self.config.service_name,
            &hostname,
            (),
            self.config.port,
            properties,
        ).map_err(|e| MdnsError::ServiceInfoFailed(e.to_string()))
    }

    async fn discovery_loop(
        receiver: mdns_sd::Receiver<ServiceEvent>,
        peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
        event_tx: mpsc::Sender<MdnsEvent>,
        running: Arc<std::sync::atomic::AtomicBool>,
        local_id: PeerId,
        local_service_name: String,
    ) {
        while running.load(std::sync::atomic::Ordering::SeqCst) {
            // Use tokio::task::spawn_blocking for the blocking recv
            let event_result = tokio::task::spawn_blocking({
                let receiver = receiver.clone();
                move || receiver.recv_timeout(Duration::from_secs(1))
            }).await;

            match event_result {
                Ok(Ok(event)) => {
                    match event {
                        ServiceEvent::ServiceResolved(info) => {
                            // Skip ourselves
                            if info.get_fullname().contains(&local_service_name) {
                                continue;
                            }

                            if let Some(peer_info) = Self::parse_service_info(&info) {
                                // Skip if it's us
                                if peer_info.id == local_id {
                                    continue;
                                }

                                let is_new = !peers.read().contains_key(info.get_fullname());
                                peers.write().insert(info.get_fullname().to_string(), peer_info.clone());

                                let event = if is_new {
                                    MdnsEvent::Discovered(peer_info)
                                } else {
                                    MdnsEvent::Updated(peer_info)
                                };

                                if event_tx.send(event).await.is_err() {
                                    break;
                                }
                            }
                        }
                        ServiceEvent::ServiceRemoved(_, full_name) => {
                            let removed_peer = {
                                peers.write().remove(&full_name)
                            };
                            if let Some(peer_info) = removed_peer {
                                let _ = event_tx.send(MdnsEvent::Lost(peer_info.id)).await;
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Err(_)) => {
                    // Timeout, continue
                }
                Err(_) => {
                    // Task error
                    break;
                }
            }
        }
    }

    fn parse_service_info(info: &ServiceInfo) -> Option<PeerInfo> {
        let properties = info.get_properties();

        let peer_id_str = properties.get("peer_id")?.val_str();
        let peer_id = PeerId::from_string(peer_id_str);

        // Get address from resolved IPs
        let addr = info.get_addresses()
            .iter()
            .next()
            .map(|ip| format!("{}:{}", ip, info.get_port()))?;

        let capabilities = NodeCapabilities {
            can_train: properties.get("can_train")
                .and_then(|v| v.val_str().parse().ok())
                .unwrap_or(false),
            can_aggregate: properties.get("can_aggregate")
                .and_then(|v| v.val_str().parse().ok())
                .unwrap_or(false),
            can_prove: properties.get("can_prove")
                .and_then(|v| v.val_str().parse().ok())
                .unwrap_or(false),
            gpu_memory_mb: properties.get("gpu_memory_mb")
                .and_then(|v| v.val_str().parse().ok())
                .unwrap_or(0),
            cpu_cores: 0,
            storage_gb: 0,
        };

        Some(PeerInfo {
            id: peer_id,
            address: addr,
            capabilities,
            last_seen: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            reputation: 0,
        })
    }
}

/// mDNS discovery errors.
#[derive(Debug, thiserror::Error)]
pub enum MdnsError {
    #[error("mDNS discovery is disabled")]
    Disabled,
    #[error("Failed to create mDNS daemon: {0}")]
    DaemonFailed(String),
    #[error("Failed to register service: {0}")]
    RegistrationFailed(String),
    #[error("Failed to browse for services: {0}")]
    BrowseFailed(String),
    #[error("Failed to get hostname: {0}")]
    HostnameFailed(String),
    #[error("Failed to create service info: {0}")]
    ServiceInfoFailed(String),
}

/// DHT-based peer discovery stub (for production use).
/// This would use libp2p-kad or similar for distributed hash table discovery.
#[cfg(feature = "dht")]
pub mod dht {
    use super::*;

    /// DHT discovery configuration.
    #[derive(Debug, Clone)]
    pub struct DhtConfig {
        /// Bootstrap nodes.
        pub bootstrap_nodes: Vec<String>,
        /// DHT protocol name.
        pub protocol_name: String,
        /// Replication factor.
        pub replication_factor: usize,
    }

    impl Default for DhtConfig {
        fn default() -> Self {
            Self {
                bootstrap_nodes: Vec::new(),
                protocol_name: "/helix/kad/1.0.0".to_string(),
                replication_factor: 20,
            }
        }
    }

    /// DHT-based peer discovery.
    pub struct DhtDiscovery {
        config: DhtConfig,
    }

    impl DhtDiscovery {
        pub fn new(config: DhtConfig) -> Self {
            Self { config }
        }

        // DHT implementation would go here using libp2p-kad
    }
}

/// Combined discovery that uses mDNS locally and DHT for production.
pub struct CombinedDiscovery {
    /// mDNS discovery.
    mdns: Option<MdnsDiscovery>,
    /// Discovered peers from all sources.
    peers: Arc<RwLock<HashMap<PeerId, PeerInfo>>>,
}

impl CombinedDiscovery {
    /// Creates a new combined discovery.
    pub fn new(local_id: PeerId, mdns_config: MdnsConfig, capabilities: NodeCapabilities) -> Self {
        let mdns = if mdns_config.enabled {
            Some(MdnsDiscovery::new(local_id, mdns_config, capabilities))
        } else {
            None
        };

        Self {
            mdns,
            peers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Starts discovery.
    pub fn start(&mut self) -> Result<mpsc::Receiver<MdnsEvent>, MdnsError> {
        if let Some(ref mut mdns) = self.mdns {
            let rx = mdns.start()?;

            // Spawn peer aggregation
            let peers = self.peers.clone();
            let mdns_peers = mdns.peers.clone();

            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(5)).await;

                    // Merge mDNS peers
                    let mdns_list = mdns_peers.read().clone();
                    for (_, peer) in mdns_list {
                        peers.write().insert(peer.id.clone(), peer);
                    }
                }
            });

            Ok(rx)
        } else {
            // Return a dummy receiver if mDNS is disabled
            let (_, rx) = mpsc::channel(1);
            Ok(rx)
        }
    }

    /// Stops discovery.
    pub fn stop(&self) {
        if let Some(ref mdns) = self.mdns {
            mdns.stop();
        }
    }

    /// Returns all discovered peers.
    pub fn get_peers(&self) -> Vec<PeerInfo> {
        self.peers.read().values().cloned().collect()
    }

    /// Adds a peer manually (from bootstrap or config).
    pub fn add_peer(&self, peer: PeerInfo) {
        self.peers.write().insert(peer.id.clone(), peer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mdns_config_default() {
        let config = MdnsConfig::default();
        assert!(config.enabled);
        assert_eq!(config.port, 9000);
    }

    #[test]
    fn test_combined_discovery_disabled() {
        let local_id = PeerId::random();
        let mut config = MdnsConfig::default();
        config.enabled = false;

        let discovery = CombinedDiscovery::new(local_id, config, NodeCapabilities::default());
        assert!(discovery.mdns.is_none());
    }

    #[test]
    fn test_manual_peer_add() {
        let local_id = PeerId::random();
        let mut config = MdnsConfig::default();
        config.enabled = false;

        let discovery = CombinedDiscovery::new(local_id, config, NodeCapabilities::default());

        let peer = PeerInfo {
            id: PeerId::random(),
            address: "127.0.0.1:9001".to_string(),
            capabilities: NodeCapabilities::default(),
            last_seen: 0,
            reputation: 0,
        };

        discovery.add_peer(peer.clone());
        assert_eq!(discovery.get_peers().len(), 1);
    }
}
