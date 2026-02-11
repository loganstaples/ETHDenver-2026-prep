use crate::state::{AppState, SystemMetrics};
use sysinfo::System;
use tauri::State;

#[tauri::command]
pub fn get_system_metrics(state: State<'_, AppState>) -> SystemMetrics {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_usage = sys.global_cpu_usage();
    let memory_used = sys.used_memory() / 1_048_576;
    let memory_total = sys.total_memory() / 1_048_576;

    let metrics = SystemMetrics {
        cpu_usage_percent: cpu_usage,
        memory_used_mb: memory_used,
        memory_total_mb: memory_total,
        gpu_usage_percent: None,
        gpu_memory_used_mb: None,
        gpu_memory_total_mb: None,
    };

    *state.system_metrics.write() = metrics.clone();
    metrics
}
