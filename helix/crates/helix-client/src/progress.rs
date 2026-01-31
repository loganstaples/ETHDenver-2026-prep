//! HELIX Progress Display Module
//!
//! Provides terminal progress indicators and spinners for long-running operations.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use colored::*;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};

/// Progress display manager for CLI operations
pub struct ProgressDisplay {
    /// Current spinner if active
    spinner: Option<ProgressBar>,
    /// Multi-progress for concurrent operations
    multi: MultiProgress,
    /// Whether currently running
    running: Arc<AtomicBool>,
}

impl ProgressDisplay {
    /// Create a new progress display
    pub fn new() -> Self {
        Self {
            spinner: None,
            multi: MultiProgress::new(),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start a spinner with the given message
    pub fn start_spinner(&mut self, message: &str) {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                .template("{spinner:.cyan} {msg}")
                .unwrap(),
        );
        spinner.set_message(message.to_string());
        spinner.enable_steady_tick(Duration::from_millis(80));
        self.running.store(true, Ordering::SeqCst);
        self.spinner = Some(spinner);
    }

    /// Finish the current spinner with a success message
    pub fn finish_spinner(&mut self, message: &str) {
        if let Some(spinner) = self.spinner.take() {
            spinner.finish_with_message(format!("{} {}", "✓".green(), message));
        }
        self.running.store(false, Ordering::SeqCst);
    }

    /// Finish the spinner with an error message
    pub fn finish_spinner_error(&mut self, message: &str) {
        if let Some(spinner) = self.spinner.take() {
            spinner.finish_with_message(format!("{} {}", "✗".red(), message));
        }
        self.running.store(false, Ordering::SeqCst);
    }

    /// Abandon the spinner without a message
    pub fn abandon_spinner(&mut self) {
        if let Some(spinner) = self.spinner.take() {
            spinner.abandon();
        }
        self.running.store(false, Ordering::SeqCst);
    }

    /// Create a progress bar with the given length
    pub fn create_progress_bar(&self, len: u64, template: Option<&str>) -> ProgressBar {
        let pb = self.multi.add(ProgressBar::new(len));
        let style = ProgressStyle::default_bar()
            .template(template.unwrap_or(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            ))
            .unwrap()
            .progress_chars("#>-");
        pb.set_style(style);
        pb
    }

    /// Create a multi-step progress display
    pub fn create_steps(&self, steps: &[&str]) -> StepProgress {
        StepProgress::new(steps)
    }

    /// Check if currently running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl Default for ProgressDisplay {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-step progress tracker
pub struct StepProgress {
    steps: Vec<String>,
    current: usize,
    spinner: Option<ProgressBar>,
}

impl StepProgress {
    /// Create a new step progress tracker
    pub fn new(steps: &[&str]) -> Self {
        Self {
            steps: steps.iter().map(|s| s.to_string()).collect(),
            current: 0,
            spinner: None,
        }
    }

    /// Start the next step
    pub fn next_step(&mut self) {
        if self.current > 0 {
            if let Some(spinner) = self.spinner.take() {
                let prev_step = &self.steps[self.current - 1];
                spinner.finish_with_message(format!("{} {}", "✓".green(), prev_step));
            }
        }

        if self.current < self.steps.len() {
            let step_msg = format!(
                "[{}/{}] {}",
                self.current + 1,
                self.steps.len(),
                self.steps[self.current]
            );

            let spinner = ProgressBar::new_spinner();
            spinner.set_style(
                ProgressStyle::default_spinner()
                    .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                    .template("{spinner:.cyan} {msg}")
                    .unwrap(),
            );
            spinner.set_message(step_msg);
            spinner.enable_steady_tick(Duration::from_millis(80));
            self.spinner = Some(spinner);
            self.current += 1;
        }
    }

    /// Complete all steps
    pub fn complete(&mut self) {
        if let Some(spinner) = self.spinner.take() {
            if self.current > 0 && self.current <= self.steps.len() {
                let step = &self.steps[self.current - 1];
                spinner.finish_with_message(format!("{} {}", "✓".green(), step));
            }
        }
    }

    /// Fail the current step
    pub fn fail(&mut self, error: &str) {
        if let Some(spinner) = self.spinner.take() {
            spinner.finish_with_message(format!("{} {}: {}", "✗".red(), self.steps[self.current - 1], error));
        }
    }
}

/// Phase tracker for demo operations
pub struct PhaseDisplay {
    phases: Vec<String>,
    current_phase: usize,
}

impl PhaseDisplay {
    /// Create a new phase display
    pub fn new(phases: &[&str]) -> Self {
        Self {
            phases: phases.iter().map(|s| s.to_string()).collect(),
            current_phase: 0,
        }
    }

    /// Start the next phase
    pub fn start_phase(&mut self) {
        if self.current_phase < self.phases.len() {
            println!(
                "\n{} Phase {}: {}",
                "▶".cyan(),
                self.current_phase + 1,
                self.phases[self.current_phase].bold()
            );
            self.current_phase += 1;
        }
    }

    /// Complete the current phase
    pub fn complete_phase(&self, message: &str) {
        println!("  {} {}", "✓".green(), message);
    }

    /// Fail the current phase
    pub fn fail_phase(&self, error: &str) {
        println!("  {} {}", "✗".red(), error);
    }
}
