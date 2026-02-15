use crate::simulation;
use crate::state::{AppState, NodeStatus};
use std::sync::atomic::Ordering;
use std::time::Instant;
use tauri::{AppHandle, State};

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
pub fn start_node(
    state: State<'_, AppState>,
    app_handle: AppHandle,
) -> Result<(), String> {
    if state.is_running() {
        return Err("Node is already running".to_string());
    }

    // Generate random peer_id
    let peer_id = format!(
        "0x{:04x}{:04x}...{:04x}",
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>(),
    );
    *state.peer_id.write() = peer_id;

    state.node_running.store(true, Ordering::Relaxed);
    *state.start_time.write() = Some(Instant::now());

    state.push_activity("Node started — contributing compute".to_string(), "success");

    // Launch demo simulation
    simulation::start_simulation(app_handle);

    Ok(())
}

#[tauri::command]
pub fn stop_node(state: State<'_, AppState>) -> Result<(), String> {
    if !state.is_running() {
        return Err("Node is not running".to_string());
    }

    state.node_running.store(false, Ordering::Relaxed);
    *state.start_time.write() = None;

    // Reset training state
    {
        let mut ts = state.training_status.write();
        ts.active = false;
        ts.phase = "Idle".to_string();
        ts.progress = 0.0;
        ts.current_round = None;
        ts.current_loss = None;
        ts.current_step = None;
        ts.total_steps = None;
        ts.model_name = String::new();
    }

    // Clear peers
    state.peers.write().clear();

    state.push_activity("Node stopped".to_string(), "info");

    Ok(())
}
