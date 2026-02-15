use crate::state::{ActivityEvent, AppState};
use tauri::State;

#[tauri::command]
pub fn get_activity_log(
    state: State<'_, AppState>,
    since_id: Option<u64>,
) -> Vec<ActivityEvent> {
    let log = state.activity_log.read();
    match since_id {
        Some(id) => log.iter().filter(|e| e.id > id).cloned().collect(),
        None => log.iter().rev().take(50).rev().cloned().collect(),
    }
}
