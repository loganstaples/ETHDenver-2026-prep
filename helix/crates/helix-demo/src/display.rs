//! Terminal display utilities for the HELIX demo.

use std::time::Duration;

// ANSI color codes
const RED: &str = "\x1b[0;31m";
const GREEN: &str = "\x1b[0;32m";
const YELLOW: &str = "\x1b[1;33m";
const BLUE: &str = "\x1b[0;34m";
const CYAN: &str = "\x1b[0;36m";
const WHITE: &str = "\x1b[1;37m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const NC: &str = "\x1b[0m";

pub fn banner() {
    println!(
        r#"
                    {WHITE}HELIX{NC}
         Trustless Distributed ML Training

    Train on untrusted GPUs worldwide
    Model weights stay private via MPC
    Every computation verified with ZK proofs
"#,
        WHITE = WHITE,
        NC = NC,
    );
}

pub fn phase(num: u32, title: &str, subtitle: &str) {
    println!();
    println!(
        "{BOLD}{BLUE}--- Phase {num}: {title} ---{NC}",
        BOLD = BOLD,
        BLUE = BLUE,
        NC = NC,
    );
    println!("    {DIM}{subtitle}{NC}", DIM = DIM, NC = NC);
    println!();
}

pub fn success(msg: &str) {
    println!("  {GREEN}[ok]{NC} {msg}", GREEN = GREEN, NC = NC);
}

pub fn info(msg: &str) {
    println!("  {CYAN}[..]{NC} {msg}", CYAN = CYAN, NC = NC);
}

pub fn metric(msg: &str) {
    println!("  {YELLOW}[**]{NC} {msg}", YELLOW = YELLOW, NC = NC);
}

pub fn warn(msg: &str) {
    println!("  {YELLOW}[!!]{NC} {msg}", YELLOW = YELLOW, NC = NC);
}

pub fn alert(msg: &str) {
    println!("  {RED}[XX]{NC} {msg}", RED = RED, NC = NC);
}

/// Truncates an address to 0xABCD...EF12 format.
pub fn short_addr(addr: &ethers::types::Address) -> String {
    let s = format!("{:?}", addr);
    if s.len() > 14 {
        format!("{}...{}", &s[..8], &s[s.len() - 6..])
    } else {
        s
    }
}

/// Draws an ASCII loss curve from multiple workers' loss histories.
pub fn loss_curve(all_losses: &[Vec<f64>]) {
    if all_losses.is_empty() || all_losses.iter().all(|l| l.is_empty()) {
        println!("  (no loss data)");
        return;
    }

    // Compute average loss per step across workers
    let max_steps = all_losses.iter().map(|l| l.len()).max().unwrap_or(0);
    if max_steps == 0 {
        return;
    }

    let mut avg_losses = Vec::with_capacity(max_steps);
    for step in 0..max_steps {
        let mut sum = 0.0;
        let mut count = 0;
        for worker_losses in all_losses {
            if step < worker_losses.len() {
                sum += worker_losses[step];
                count += 1;
            }
        }
        if count > 0 {
            avg_losses.push(sum / count as f64);
        }
    }

    let max_loss = avg_losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_loss = avg_losses.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = (max_loss - min_loss).max(0.001);

    let height = 12;
    let width = 60.min(avg_losses.len());

    // Downsample if needed
    let step_size = (avg_losses.len() as f64 / width as f64).ceil() as usize;

    println!();
    println!("  {BOLD}Loss Curve (avg across workers){NC}", BOLD = BOLD, NC = NC);
    println!("  {:.4} |", max_loss);

    for row in 0..height {
        let threshold = max_loss - (row as f64 / (height - 1) as f64) * range;
        let mut line = String::new();
        for col in 0..width {
            let idx = col * step_size;
            if idx < avg_losses.len() {
                let val = avg_losses[idx];
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
                "  {:.4} |{line}|",
                min_loss + range / 2.0,
            );
        } else {
            println!("         |{line}|");
        }
    }

    println!("  {:.4} |{}|", min_loss, "-".repeat(width));
    if width > 6 {
        println!(
            "         {DIM}step 0{:>width$}{NC}",
            format!("step {}", avg_losses.len()),
            width = width - 6,
            DIM = DIM,
            NC = NC,
        );
    } else {
        println!(
            "         {DIM}step 0 .. step {}{NC}",
            avg_losses.len(),
            DIM = DIM,
            NC = NC,
        );
    }
    println!();
}

pub fn summary(
    total_time: Duration,
    total_proofs: usize,
    num_workers: usize,
    loss_start: f64,
    loss_end: f64,
    on_chain: bool,
) {
    println!();
    println!(
        "  {GREEN}==============================================={NC}",
        GREEN = GREEN,
        NC = NC,
    );
    println!(
        "  {GREEN}          HELIX Demo Complete{NC}",
        GREEN = GREEN,
        NC = NC,
    );
    println!(
        "  {GREEN}==============================================={NC}",
        GREEN = GREEN,
        NC = NC,
    );
    println!();
    println!("  {BOLD}What was demonstrated:{NC}", BOLD = BOLD, NC = NC);
    println!("    {GREEN}[ok]{NC} Real ZK proof generation (Halo2 KZG)", GREEN = GREEN, NC = NC);
    println!("    {GREEN}[ok]{NC} Multi-worker distributed training", GREEN = GREEN, NC = NC);
    println!(
        "    {GREEN}[ok]{NC} Loss convergence: {:.4} -> {:.4}",
        loss_start,
        loss_end,
        GREEN = GREEN,
        NC = NC,
    );

    if on_chain {
        println!("    {GREEN}[ok]{NC} On-chain proof submission via HelixCoordinatorV2", GREEN = GREEN, NC = NC);
        println!("    {GREEN}[ok]{NC} Adversarial detection and slashing", GREEN = GREEN, NC = NC);
    }

    println!();
    println!("  {BOLD}Key metrics:{NC}", BOLD = BOLD, NC = NC);
    println!(
        "    Total proofs:  {CYAN}{}{NC} across {} workers",
        total_proofs,
        num_workers,
        CYAN = CYAN,
        NC = NC,
    );
    println!(
        "    Total time:    {CYAN}{:.1}s{NC}",
        total_time.as_secs_f64(),
        CYAN = CYAN,
        NC = NC,
    );
    let reduction = if loss_start > 1e-10 {
        (1.0 - loss_end / loss_start) * 100.0
    } else {
        0.0
    };
    println!(
        "    Loss reduction: {CYAN}{:.1}%{NC}",
        reduction,
        CYAN = CYAN,
        NC = NC,
    );
    println!();
}
