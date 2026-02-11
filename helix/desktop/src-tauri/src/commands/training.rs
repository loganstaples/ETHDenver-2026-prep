use crate::state::{AppState, TrainingStatus};
use tauri::State;

#[tauri::command]
pub fn get_training_status(state: State<'_, AppState>) -> TrainingStatus {
    state.training_status.read().clone()
}
