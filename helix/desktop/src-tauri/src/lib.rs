mod commands;
mod simulation;
mod state;

use state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app_state = AppState::new();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            commands::node::get_node_status,
            commands::node::start_node,
            commands::node::stop_node,
            commands::system::get_system_metrics,
            commands::training::get_training_status,
            commands::network::get_peers,
            commands::config::get_config,
            commands::config::set_config,
            commands::activity::get_activity_log,
            commands::session::get_session_info,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
