use crate::state::{AppState, NodeStatus};
use std::sync::atomic::Ordering;
use std::time::Instant;
use tauri::State;

#[tauri::command]
pub fn get_node_status(state: State<'_, AppState>) -> NodeStatus {
    NodeStatus {
        running: state.is_running(),
        role: "Compute".to_string(),
        uptime_secs: state.uptime_secs(),
        peer_id: state.peer_id.read().clone(),
        connected_peers: state.peers.read().len() as u32,
    }
}

#[tauri::command]
pub fn start_node(state: State<'_, AppState>) -> Result<(), String> {
    if state.is_running() {
        return Err("Node is already running".to_string());
    }

    state.node_running.store(true, Ordering::Relaxed);
    *state.start_time.write() = Some(Instant::now());

    // TODO: Phase 2 — spawn actual helix-node worker via NodeManager
    Ok(())
}

#[tauri::command]
pub fn stop_node(state: State<'_, AppState>) -> Result<(), String> {
    if !state.is_running() {
        return Err("Node is not running".to_string());
    }

    state.node_running.store(false, Ordering::Relaxed);
    *state.start_time.write() = None;

    // TODO: Phase 2 — send shutdown signal to NodeManager
    Ok(())
}
