export interface NodeStatus {
  running: boolean;
  role: string;
  uptime_secs: number;
  peer_id: string;
  connected_peers: number;
}

export interface TrainingStatus {
  active: boolean;
  current_round: number | null;
  phase: string;
  progress: number;
  rounds_completed: number;
  proofs_generated: number;
  total_earned: number;
  session_earned: number;
}

export interface SystemMetrics {
  cpu_usage_percent: number;
  memory_used_mb: number;
  memory_total_mb: number;
  gpu_usage_percent: number | null;
  gpu_memory_used_mb: number | null;
  gpu_memory_total_mb: number | null;
}

export interface PeerInfo {
  id: string;
  address: string;
  role: string;
  reputation: number;
  last_seen_secs_ago: number;
}

export interface NodeConfig {
  listen_address: string;
  aggregator_address: string;
  rpc_port: number;
  cpu_threads: number;
  gpu_memory_limit_mb: number;
  max_concurrent_tasks: number;
  local_epochs: number;
  batch_size: number;
  generate_proofs: boolean;
  max_error_bound: number;
  use_tls: boolean;
}

export interface ActivityEvent {
  id: string;
  timestamp: string;
  message: string;
  type: "info" | "success" | "warning" | "error";
}
