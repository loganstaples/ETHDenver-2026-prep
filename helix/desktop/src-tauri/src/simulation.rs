use crate::state::AppState;
use std::sync::atomic::Ordering;
use tauri::AppHandle;
use tauri::Manager;
use tokio::time::{sleep, Duration};

/// Starts a background simulation task that cycles training rounds,
/// generates activity events, and varies metrics while the node is running.
pub fn start_simulation(app_handle: AppHandle) {
    tokio::spawn(async move {
        let state = app_handle.state::<AppState>();

        // Give a brief startup delay
        sleep(Duration::from_millis(500)).await;

        // Spawn initial peers
        spawn_peers(&state);
        state.push_activity("Connected to aggregator".to_string(), "success");
        state.push_activity(
            format!(
                "Discovered {} peers on the network",
                state.peers.read().len()
            ),
            "info",
        );

        let mut round_number: u64 = 1;

        loop {
            if !state.is_running() {
                break;
            }

            // Start a new training round
            let total_steps: u64 = 50 + (rand::random::<u64>() % 51); // 50-100 steps
            let model_name = pick_model_name(round_number);

            {
                let mut ts = state.training_status.write();
                ts.active = true;
                ts.current_round = Some(round_number);
                ts.model_name = model_name.clone();
                ts.total_steps = Some(total_steps);
                ts.current_step = Some(0);
            }

            state.push_activity(
                format!("Round #{round_number} assigned — model: {model_name}"),
                "info",
            );

            // Phase 1: Assigned (brief)
            set_phase(&state, "Assigned", 0.0);
            if !wait_running(&state, 1000, 2000).await {
                break;
            }

            // Phase 2: Training
            set_phase(&state, "Training", 0.05);
            state.push_activity(
                format!("Training started — {total_steps} steps"),
                "info",
            );

            let mut loss = 2.5 + rand::random::<f64>() * 1.5; // start loss 2.5-4.0
            for step in 1..=total_steps {
                if !state.is_running() {
                    break;
                }

                loss *= 0.97 + rand::random::<f64>() * 0.02; // decay with jitter
                let progress = 0.05 + (step as f32 / total_steps as f32) * 0.45;

                {
                    let mut ts = state.training_status.write();
                    ts.current_step = Some(step);
                    ts.current_loss = Some(loss);
                    ts.progress = progress;
                }

                sleep(Duration::from_millis(80 + rand::random::<u64>() % 120)).await;
            }

            if !state.is_running() {
                break;
            }

            state.push_activity(
                format!("Training complete — loss: {:.4}", loss),
                "success",
            );

            // Phase 3: Proving
            set_phase(&state, "Proving", 0.55);
            state.push_activity("Generating ZK proof...".to_string(), "info");
            if !wait_running(&state, 3000, 6000).await {
                break;
            }
            state.push_activity("Proof generated (1856 bytes)".to_string(), "success");

            // Phase 4: Submitted
            set_phase(&state, "Submitted", 0.80);
            state.push_activity("Proof submitted to coordinator".to_string(), "info");
            if !wait_running(&state, 1000, 3000).await {
                break;
            }

            // Phase 5: Verified
            set_phase(&state, "Verified", 1.0);
            let earned = 0.5 + rand::random::<f64>() * 2.0;

            {
                let mut ts = state.training_status.write();
                ts.rounds_completed += 1;
                ts.proofs_generated += 1;
                ts.total_earned += earned;
                ts.session_earned += earned;
            }

            state.push_activity(
                format!("Round #{round_number} verified — earned {earned:.3} HLX"),
                "earn",
            );

            // Brief pause before next round
            if !wait_running(&state, 2000, 5000).await {
                break;
            }

            // Occasionally churn peers
            if rand::random::<u8>() % 3 == 0 {
                churn_peers(&state);
            }

            round_number += 1;
        }
    });
}

fn set_phase(state: &AppState, phase: &str, progress: f32) {
    let mut ts = state.training_status.write();
    ts.phase = phase.to_string();
    ts.progress = progress;
}

/// Wait for a random duration between min_ms and max_ms, returning false if node stops.
async fn wait_running(state: &AppState, min_ms: u64, max_ms: u64) -> bool {
    let duration = min_ms + rand::random::<u64>() % (max_ms - min_ms);
    let steps = (duration / 200).max(1);
    for _ in 0..steps {
        if !state.node_running.load(Ordering::Relaxed) {
            return false;
        }
        sleep(Duration::from_millis(200)).await;
    }
    true
}

fn spawn_peers(state: &AppState) {
    let mut peers = state.peers.write();
    peers.clear();
    let count = 8 + rand::random::<usize>() % 8; // 8-15 peers
    for i in 0..count {
        peers.push(crate::state::PeerInfo {
            id: format!(
                "0x{:04x}{:04x}...{:04x}",
                rand::random::<u16>(),
                rand::random::<u16>(),
                rand::random::<u16>(),
            ),
            address: format!("10.0.{}.{}:9000", rand::random::<u8>(), rand::random::<u8>()),
            role: if i == 0 {
                "Aggregator".to_string()
            } else {
                "Compute".to_string()
            },
            reputation: 80 + (rand::random::<i32>() % 21), // 80-100
            last_seen_secs_ago: rand::random::<u64>() % 5,
        });
    }
}

fn churn_peers(state: &AppState) {
    let mut peers = state.peers.write();
    // Remove a random peer (if >3)
    if peers.len() > 3 {
        let idx = rand::random::<usize>() % peers.len();
        let removed = peers.remove(idx);
        state.push_activity(
            format!("Peer disconnected: {}", removed.id),
            "warning",
        );
    }
    // Add a new peer
    let new_peer = crate::state::PeerInfo {
        id: format!(
            "0x{:04x}{:04x}...{:04x}",
            rand::random::<u16>(),
            rand::random::<u16>(),
            rand::random::<u16>(),
        ),
        address: format!("10.0.{}.{}:9000", rand::random::<u8>(), rand::random::<u8>()),
        role: "Compute".to_string(),
        reputation: 80 + (rand::random::<i32>() % 21),
        last_seen_secs_ago: 0,
    };
    state.push_activity(
        format!("New peer connected: {}", new_peer.id),
        "info",
    );
    peers.push(new_peer);

    // Update last_seen for existing peers
    for peer in peers.iter_mut() {
        peer.last_seen_secs_ago = rand::random::<u64>() % 8;
    }
}

fn pick_model_name(round: u64) -> String {
    let models = [
        "gpt-helix-7b",
        "llama-3.1-8b",
        "mistral-7b-v0.3",
        "phi-3-mini",
        "qwen2-7b",
        "gemma-2-9b",
    ];
    models[(round as usize) % models.len()].to_string()
}
