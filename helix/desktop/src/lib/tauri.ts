import { invoke } from "@tauri-apps/api/core";
import type {
  NodeStatus,
  TrainingStatus,
  SystemMetrics,
  PeerInfo,
  NodeConfig,
} from "./types";

export async function getNodeStatus(): Promise<NodeStatus> {
  return invoke("get_node_status");
}

export async function startNode(): Promise<void> {
  return invoke("start_node");
}

export async function stopNode(): Promise<void> {
  return invoke("stop_node");
}

export async function getSystemMetrics(): Promise<SystemMetrics> {
  return invoke("get_system_metrics");
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  return invoke("get_training_status");
}

export async function getPeers(): Promise<PeerInfo[]> {
  return invoke("get_peers");
}

export async function getConfig(): Promise<NodeConfig> {
  return invoke("get_config");
}

export async function setConfig(config: NodeConfig): Promise<void> {
  return invoke("set_config", { config });
}
