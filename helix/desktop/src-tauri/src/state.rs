use parking_lot::RwLock;
use serde::Serialize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

#[derive(Debug, Serialize, Clone)]
pub struct NodeStatus {
    pub running: bool,
    pub role: String,
    pub uptime_secs: u64,
    pub peer_id: String,
    pub connected_peers: u32,
}

#[derive(Debug, Serialize, Clone)]
pub struct TrainingStatus {
    pub active: bool,
    pub current_round: Option<u64>,
    pub phase: String,
    pub progress: f32,
    pub rounds_completed: u64,
    pub proofs_generated: u64,
    pub total_earned: f64,
    pub session_earned: f64,
    pub model_name: String,
    pub current_loss: Option<f64>,
    pub current_step: Option<u64>,
    pub total_steps: Option<u64>,
}

#[derive(Debug, Serialize, Clone)]
pub struct SystemMetrics {
    pub cpu_usage_percent: f32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub gpu_usage_percent: Option<f32>,
    pub gpu_memory_used_mb: Option<u64>,
    pub gpu_memory_total_mb: Option<u64>,
}

#[derive(Debug, Serialize, Clone)]
pub struct PeerInfo {
    pub id: String,
    pub address: String,
    pub role: String,
    pub reputation: i32,
    pub last_seen_secs_ago: u64,
}

#[derive(Debug, Serialize, serde::Deserialize, Clone, Default)]
pub struct NodeConfig {
    pub listen_address: String,
    pub aggregator_address: String,
    pub rpc_port: u16,
    pub cpu_threads: u32,
    pub gpu_memory_limit_mb: u32,
    pub max_concurrent_tasks: u32,
    pub local_epochs: u32,
    pub batch_size: u32,
    pub generate_proofs: bool,
    pub max_error_bound: f64,
    pub use_tls: bool,
}

#[derive(Debug, Serialize, Clone)]
pub struct ActivityEvent {
    pub id: u64,
    pub timestamp: String,
    pub message: String,
    pub event_type: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct SessionInfo {
    pub start_time: String,
    pub duration_secs: u64,
    pub earnings_rate_per_hour: f64,
}

pub struct AppState {
    pub node_running: AtomicBool,
    pub start_time: RwLock<Option<Instant>>,
    pub peer_id: RwLock<String>,
    pub config: RwLock<NodeConfig>,
    pub training_status: RwLock<TrainingStatus>,
    pub peers: RwLock<Vec<PeerInfo>>,
    pub system_metrics: RwLock<SystemMetrics>,
    pub activity_log: RwLock<Vec<ActivityEvent>>,
    pub activity_counter: AtomicU64,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            node_running: AtomicBool::new(false),
            start_time: RwLock::new(None),
            peer_id: RwLock::new(String::from("0x0000...0000")),
            config: RwLock::new(NodeConfig {
                listen_address: "0.0.0.0:9000".to_string(),
                aggregator_address: "127.0.0.1:9000".to_string(),
                rpc_port: 9002,
                cpu_threads: 4,
                gpu_memory_limit_mb: 8192,
                max_concurrent_tasks: 1,
                local_epochs: 5,
                batch_size: 32,
                generate_proofs: true,
                max_error_bound: 0.1,
                use_tls: false,
            }),
            training_status: RwLock::new(TrainingStatus {
                active: false,
                current_round: None,
                phase: "Idle".to_string(),
                progress: 0.0,
                rounds_completed: 0,
                proofs_generated: 0,
                total_earned: 0.0,
                session_earned: 0.0,
                model_name: String::new(),
                current_loss: None,
                current_step: None,
                total_steps: None,
            }),
            peers: RwLock::new(Vec::new()),
            system_metrics: RwLock::new(SystemMetrics {
                cpu_usage_percent: 0.0,
                memory_used_mb: 0,
                memory_total_mb: 0,
                gpu_usage_percent: None,
                gpu_memory_used_mb: None,
                gpu_memory_total_mb: None,
            }),
            activity_log: RwLock::new(Vec::new()),
            activity_counter: AtomicU64::new(0),
        }
    }

    pub fn is_running(&self) -> bool {
        self.node_running.load(Ordering::Relaxed)
    }

    pub fn uptime_secs(&self) -> u64 {
        self.start_time
            .read()
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0)
    }

    pub fn push_activity(&self, message: String, event_type: &str) {
        let id = self.activity_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let event = ActivityEvent {
            id,
            timestamp: chrono::Utc::now().to_rfc3339(),
            message,
            event_type: event_type.to_string(),
        };
        let mut log = self.activity_log.write();
        log.push(event);
        // Keep at most 200 events
        if log.len() > 200 {
            let excess = log.len() - 200;
            log.drain(0..excess);
        }
    }
}
