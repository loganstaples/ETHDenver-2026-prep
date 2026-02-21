//! Terminal display utilities for the HELIX MPC training demo.
//!
//! Provides colorful, structured output for each phase of the demo using
//! ANSI escape codes via the `colored` crate and progress bars via `indicatif`.

use std::io::Write;
use std::time::Duration;

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

// ============================================================================
// Banner
// ============================================================================

/// Prints the HELIX ASCII art banner.
pub fn banner() {
    let art = r#"
    ██╗  ██╗███████╗██╗     ██╗██╗  ██╗
    ██║  ██║██╔════╝██║     ██║╚██╗██╔╝
    ███████║█████╗  ██║     ██║ ╚███╔╝
    ██╔══██║██╔══╝  ██║     ██║ ██╔██╗
    ██║  ██║███████╗███████╗██║██╔╝ ██╗
    ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝
"#;
    println!("{}", art.bright_cyan().bold());
    println!(
        "    {}",
        "Trustless Distributed ML Training".white().bold()
    );
    println!(
        "    {}",
        "MPC-Primary Architecture | SPDZ MAC Verification".dimmed()
    );
    println!(
        "    {}",
        "ETHDenver 2026".dimmed()
    );
    println!();
}

// ============================================================================
// Phase Headings
// ============================================================================

/// Prints a major phase heading with a number and description.
pub fn phase(title: &str) {
    println!();
    let separator = "=".repeat(60);
    println!("  {}", separator.bright_blue().bold());
    println!("  {}", title.bright_blue().bold());
    println!("  {}", separator.bright_blue().bold());
    println!();
}

/// Prints a sub-phase description.
pub fn subphase(msg: &str) {
    println!("  {} {}", ">>>".bright_blue(), msg.white());
}

// ============================================================================
// Status Messages
// ============================================================================

/// Prints a success message with green checkmark.
pub fn success(msg: &str) {
    println!("  {} {}", "[OK]".bright_green().bold(), msg);
}

/// Prints an informational message.
pub fn info(msg: &str) {
    println!("  {} {}", "[..]".bright_cyan(), msg);
}

/// Prints a metric/statistic.
pub fn metric(label: &str, value: &str) {
    println!(
        "  {} {}: {}",
        "[**]".bright_yellow(),
        label,
        value.bright_white().bold()
    );
}

/// Prints a warning message.
pub fn warn(msg: &str) {
    println!("  {} {}", "[!!]".bright_yellow().bold(), msg.yellow());
}

/// Prints an error/alert message.
pub fn alert(msg: &str) {
    println!("  {} {}", "[XX]".bright_red().bold(), msg.bright_red());
}

// ============================================================================
// Training Step Display
// ============================================================================

/// Prints a single-line in-place training step update.
///
/// Uses carriage return to overwrite the previous line for a clean
/// scrolling effect during training.
pub fn step_update(step: usize, total_steps: usize, loss: f64, workers: usize, mac_ok: bool) {
    let mac_status = if mac_ok {
        "MAC OK".bright_green().to_string()
    } else {
        "MAC FAIL".bright_red().bold().to_string()
    };

    let pct = (step as f64 / total_steps as f64 * 100.0) as usize;
    let bar_width = 30;
    let filled = (pct * bar_width / 100).min(bar_width);
    let empty = bar_width - filled;
    let bar = format!(
        "[{}{}]",
        "#".repeat(filled),
        "-".repeat(empty)
    );

    print!(
        "\r  {} Step {}/{} | Loss: {:.6} | Workers: {} | {} | {}%   ",
        bar.bright_blue(),
        format!("{:>4}", step).bright_white(),
        total_steps,
        loss,
        workers,
        mac_status,
        pct,
    );
    let _ = std::io::stdout().flush();
}

/// Finishes the step update line (moves to next line).
pub fn step_update_finish() {
    println!();
}

// ============================================================================
// Checkpoint Display
// ============================================================================

/// Prints a checkpoint announcement with commitment hash.
pub fn checkpoint(step: u64, commitment_hex: &str) {
    println!();
    println!(
        "  {} Checkpoint at step {}",
        "[CP]".bright_magenta().bold(),
        format!("{}", step).bright_white().bold(),
    );
    println!(
        "       Pedersen commitment: {}",
        if commitment_hex.len() > 20 {
            format!("{}...{}", &commitment_hex[..10], &commitment_hex[commitment_hex.len() - 10..])
        } else {
            commitment_hex.to_string()
        }
        .dimmed()
    );
}

/// Prints a MAC verification success message.
pub fn mac_verified(step: u64) {
    println!(
        "  {} MAC verification passed at step {} (information-theoretic integrity confirmed)",
        "[MAC]".bright_green().bold(),
        step,
    );
}

/// Prints ZK proof generation start notice.
pub fn zk_proof_start(step: u64) {
    println!(
        "  {} Generating StateTransitionCircuit proof at step {}...",
        "[ZK]".bright_magenta().bold(),
        step,
    );
}

/// Prints ZK proof generation success.
pub fn zk_proof_success(step: u64, proof_size: usize, gen_time_ms: u64, verified: bool) {
    let verify_status = if verified { ", self-verified" } else { "" };
    println!(
        "  {} ZK proof generated at step {} ({} bytes, {:.1}s{})",
        "[ZK]".bright_green().bold(),
        step,
        proof_size,
        gen_time_ms as f64 / 1000.0,
        verify_status,
    );
}

/// Prints ZK proof generation failure.
pub fn zk_proof_failed(step: u64, error: &str) {
    println!(
        "  {} ZK proof generation failed at step {}: {}",
        "[ZK]".bright_red().bold(),
        step,
        error,
    );
}

/// Prints ZK prover initialization notice.
pub fn zk_prover_init(num_weights: usize, k: u32) {
    println!(
        "  {} Initializing ZK prover (num_weights={}, k={}, SRS setup + keygen)...",
        "[ZK]".bright_magenta().bold(),
        num_weights,
        k,
    );
}

/// Prints ZK prover initialization success.
pub fn zk_prover_init_done(elapsed_ms: u64) {
    println!(
        "  {} ZK prover initialized ({:.1}s)",
        "[ZK]".bright_green().bold(),
        elapsed_ms as f64 / 1000.0,
    );
}

// ============================================================================
// Training Signal Display
// ============================================================================

/// Prints a notice that training was paused at the given step.
pub fn training_paused(step: usize) {
    println!();
    println!(
        "  {} Training paused at step {}",
        "[PAUSED]".bright_yellow().bold(),
        format!("{}", step).bright_white().bold(),
    );
    println!(
        "       {}",
        "Checkpoint saved — training can be resumed later".yellow()
    );
    println!();
}

/// Prints a notice that training was stopped at the given step.
pub fn training_stopped(step: usize) {
    println!();
    println!(
        "  {} Training stopped at step {}",
        "[STOPPED]".bright_red().bold(),
        format!("{}", step).bright_white().bold(),
    );
    println!(
        "       {}",
        "Training terminated permanently — proceeding to settlement".red()
    );
    println!();
}

// ============================================================================
// Cheater Detection Display
// ============================================================================

/// Prints a dramatic cheater detection announcement.
pub fn cheater_detected(party: usize, step: u64) {
    println!();
    let separator = "!".repeat(60);
    println!("  {}", separator.bright_red().bold());
    println!(
        "  {}  CHEATER DETECTED: Party {} at step {}",
        "ALERT".bright_red().bold(),
        format!("{}", party).bright_red().bold(),
        step,
    );
    println!("  {}", separator.bright_red().bold());
    println!(
        "       {}",
        "SPDZ MAC verification failed - pairwise identification complete".bright_red()
    );
    println!(
        "       {}",
        "Corrupted party excluded from further training".yellow()
    );
    println!();
}

/// Prints cheater simulation notice.
pub fn cheater_simulated(party: usize, step: usize) {
    println!();
    println!(
        "  {} Injecting corruption into Party {} at step {} (simulated attack)",
        "[SIM]".bright_yellow().bold(),
        party,
        step,
    );
}

// ============================================================================
// Progress Bars
// ============================================================================

/// Creates a progress bar for Beaver triple generation.
pub fn triple_progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "  {spinner:.green} Generating Beaver triples [{bar:40.cyan/blue}] {pos}/{len} ({eta})"
        )
        .unwrap()
        .progress_chars("##-"),
    );
    pb
}

/// Creates a progress bar for generic operations.
pub fn generic_progress_bar(total: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            &format!("  {{spinner:.green}} {} [{{bar:40.cyan/blue}}] {{pos}}/{{len}} ({{eta}})", msg)
        )
        .unwrap()
        .progress_chars("##-"),
    );
    pb
}

// ============================================================================
// Loss Curve
// ============================================================================

/// Draws an ASCII loss curve from training loss history.
pub fn loss_curve(losses: &[f64]) {
    if losses.is_empty() {
        println!("  (no loss data)");
        return;
    }

    let height = 14;
    let width = 60.min(losses.len());
    if width == 0 {
        return;
    }

    let max_loss = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_loss = losses.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = (max_loss - min_loss).max(0.001);

    let step_size = (losses.len() as f64 / width as f64).ceil() as usize;

    println!();
    println!(
        "  {}",
        "Loss Convergence Curve".white().bold()
    );
    println!("  {:.4} |", max_loss);

    for row in 0..height {
        let threshold = max_loss - (row as f64 / (height - 1) as f64) * range;
        let mut line = String::new();
        for col in 0..width {
            let idx = col * step_size;
            if idx < losses.len() {
                let val = losses[idx];
                if val >= threshold {
                    line.push('#');
                } else {
                    line.push(' ');
                }
            } else {
                line.push(' ');
            }
        }

        if row == height / 2 {
            println!(
                "  {:.4} |{}|",
                min_loss + range / 2.0,
                line.bright_cyan(),
            );
        } else {
            println!("         |{}|", line.bright_cyan());
        }
    }

    println!(
        "  {:.4} |{}|",
        min_loss,
        "-".repeat(width).dimmed(),
    );
    if width > 10 {
        println!(
            "         {}{}",
            "step 0".dimmed(),
            format!("{:>width$}", format!("step {}", losses.len()), width = width - 4).dimmed(),
        );
    }
    println!();
}

// ============================================================================
// On-Chain Display
// ============================================================================

/// Prints a successful on-chain checkpoint submission.
pub fn chain_checkpoint_submitted(step: u64, tx_hash: &str, gas_used: u64, signers: usize) {
    println!(
        "  {} Checkpoint step {} submitted on-chain ({} signers, {} gas)",
        "[TX]".bright_green().bold(),
        format!("{}", step).bright_white().bold(),
        signers,
        format!("{}", gas_used).dimmed(),
    );
    println!(
        "       tx: {}",
        tx_hash.dimmed(),
    );
}

// ============================================================================
// Summary
// ============================================================================

/// On-chain settlement stats for the summary display.
pub struct ChainStats {
    pub checkpoints_submitted: usize,
    pub total_gas: u64,
    pub contract_address: String,
    pub job_completed: bool,
}

/// Holds all the stats for the final summary.
pub struct DemoSummary {
    pub total_time: Duration,
    pub num_workers: usize,
    pub num_steps: usize,
    pub loss_start: f64,
    pub loss_end: f64,
    pub final_accuracy: f64,
    pub triples_generated: usize,
    pub mac_checks_passed: usize,
    pub cheater_detected: bool,
    pub cheater_party: Option<usize>,
    pub zk_proofs_generated: usize,
    pub zk_proofs_verified: usize,
    pub zk_total_proving_time_ms: u64,
    pub chain_stats: Option<ChainStats>,
}

/// Prints the final demo summary with all metrics.
pub fn summary(stats: &DemoSummary) {
    println!();
    let separator = "=".repeat(60);
    println!("  {}", separator.bright_green());
    println!(
        "  {}",
        "HELIX Demo Complete".bright_green().bold()
    );
    println!("  {}", separator.bright_green());
    println!();

    println!("  {}", "What was demonstrated:".white().bold());
    println!(
        "    {} MPC training: {} workers, weights never reconstructed",
        "[OK]".bright_green(),
        stats.num_workers,
    );
    println!(
        "    {} SPDZ MAC verification: {} checks passed (information-theoretic security)",
        "[OK]".bright_green(),
        stats.mac_checks_passed,
    );
    println!(
        "    {} Beaver triple multiplication: {} triples consumed",
        "[OK]".bright_green(),
        stats.triples_generated,
    );
    println!(
        "    {} Loss convergence: {:.4} -> {:.4} ({:.1}% reduction)",
        "[OK]".bright_green(),
        stats.loss_start,
        stats.loss_end,
        if stats.loss_start > 1e-10 {
            (1.0 - stats.loss_end / stats.loss_start) * 100.0
        } else {
            0.0
        },
    );
    println!(
        "    {} MNIST accuracy: {:.1}% on test set",
        "[OK]".bright_green(),
        stats.final_accuracy * 100.0,
    );

    if stats.cheater_detected {
        println!(
            "    {} Cheater detection: Party {} identified and excluded via pairwise MAC check",
            "[OK]".bright_green(),
            stats.cheater_party.unwrap_or(0),
        );
    }

    if stats.zk_proofs_generated > 0 {
        println!(
            "    {} ZK proofs: {} generated, {} verified ({:.1}s total proving time)",
            "[OK]".bright_green(),
            stats.zk_proofs_generated,
            stats.zk_proofs_verified,
            stats.zk_total_proving_time_ms as f64 / 1000.0,
        );
    }

    if let Some(ref cs) = stats.chain_stats {
        println!(
            "    {} On-chain settlement: {} checkpoints submitted to HelixCoordinatorV4",
            "[OK]".bright_green(),
            cs.checkpoints_submitted,
        );
        println!(
            "    {} Multi-party attestation: all {} workers signed each checkpoint (ECDSA/EIP-191)",
            "[OK]".bright_green(),
            stats.num_workers,
        );
        if cs.job_completed {
            println!(
                "    {} Training finalized on-chain (contract: {})",
                "[OK]".bright_green(),
                &cs.contract_address[..20.min(cs.contract_address.len())],
            );
        }
    }

    println!();
    println!("  {}", "Key metrics:".white().bold());
    println!(
        "    Training steps:   {}",
        format!("{}", stats.num_steps).bright_cyan(),
    );
    println!(
        "    MPC workers:      {}",
        format!("{}", stats.num_workers).bright_cyan(),
    );
    println!(
        "    Total time:       {}",
        format!("{:.1}s", stats.total_time.as_secs_f64()).bright_cyan(),
    );
    let steps_per_sec = stats.num_steps as f64 / stats.total_time.as_secs_f64();
    println!(
        "    Throughput:       {}",
        format!("{:.1} steps/sec", steps_per_sec).bright_cyan(),
    );
    println!(
        "    Final accuracy:   {}",
        format!("{:.1}%", stats.final_accuracy * 100.0).bright_cyan(),
    );
    if stats.zk_proofs_generated > 0 {
        println!(
            "    ZK proofs:        {}",
            format!(
                "{} generated ({:.1}s avg)",
                stats.zk_proofs_generated,
                if stats.zk_proofs_generated > 0 {
                    stats.zk_total_proving_time_ms as f64 / stats.zk_proofs_generated as f64 / 1000.0
                } else {
                    0.0
                },
            ).bright_cyan(),
        );
    }

    if let Some(ref cs) = stats.chain_stats {
        println!(
            "    On-chain TXs:     {}",
            format!("{} checkpoint submissions", cs.checkpoints_submitted).bright_cyan(),
        );
        println!(
            "    Total gas:        {}",
            format!("{} ({:.4} ETH @ 1 gwei)", cs.total_gas, cs.total_gas as f64 / 1e9).bright_cyan(),
        );
    }

    println!();
    println!("  {}", "Security properties:".white().bold());
    println!(
        "    {} Information-theoretic: SPDZ MACs cannot be broken with unlimited compute",
        "*".bright_yellow(),
    );
    println!(
        "    {} Individual cheater identification via pairwise MAC verification",
        "*".bright_yellow(),
    );
    println!(
        "    {} Zero weight leakage: additive secret sharing throughout training",
        "*".bright_yellow(),
    );
    println!(
        "    {} Self-healing: remove cheater, continue training with remaining parties",
        "*".bright_yellow(),
    );
    if stats.chain_stats.is_some() {
        println!(
            "    {} On-chain settlement: multi-party attestation with minimal gas (~50-80K per checkpoint)",
            "*".bright_yellow(),
        );
    }
    println!();
}
