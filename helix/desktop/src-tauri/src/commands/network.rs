use crate::state::{AppState, PeerInfo};
use tauri::State;

#[tauri::command]
pub fn get_peers(state: State<'_, AppState>) -> Vec<PeerInfo> {
    state.peers.read().clone()
}
