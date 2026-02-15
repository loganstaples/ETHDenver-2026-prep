use crate::state::{AppState, SessionInfo};
use tauri::State;

#[tauri::command]
pub fn get_session_info(state: State<'_, AppState>) -> SessionInfo {
    let uptime = state.uptime_secs();
    let training = state.training_status.read();
    let rate = if uptime > 0 {
        (training.session_earned / uptime as f64) * 3600.0
    } else {
        0.0
    };

    SessionInfo {
        start_time: state
            .start_time
            .read()
            .map(|_| chrono::Utc::now().to_rfc3339())
            .unwrap_or_default(),
        duration_secs: uptime,
        earnings_rate_per_hour: rate,
    }
}
