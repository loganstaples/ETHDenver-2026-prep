use crate::state::{AppState, NodeConfig};
use tauri::State;

#[tauri::command]
pub fn get_config(state: State<'_, AppState>) -> NodeConfig {
    state.config.read().clone()
}

#[tauri::command]
pub fn set_config(state: State<'_, AppState>, config: NodeConfig) -> Result<(), String> {
    *state.config.write() = config;
    // TODO: persist to ~/.helix/desktop-config.json
    Ok(())
}
