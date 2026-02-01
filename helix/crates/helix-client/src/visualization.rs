//! HELIX Interactive Training Visualization
//!
//! Terminal-based UI for monitoring distributed ML training in real-time.
//! Uses ratatui for rich terminal graphics including:
//! - Training progress and loss curves
//! - Worker status dashboard
//! - Proof generation timeline
//! - Network topology visualization
//! - Event log streaming

use std::collections::VecDeque;
use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, List, ListItem,
        Paragraph, Row, Table, Tabs,
    },
    Frame, Terminal,
};
use tokio::sync::broadcast;

use crate::rpc::{
    ProofPhase, ProofStatus, RealTimeProofTracker, RealTimeTrainingTracker,
    TrainingPhase, TrainingStatus,
};

// ============================================================================
// Visualization State
// ============================================================================

/// State for the visualization UI
pub struct VisualizationState {
    /// Selected tab index
    pub selected_tab: usize,
    /// Event log (most recent events)
    pub event_log: VecDeque<VisualizationEvent>,
    /// Maximum events to keep in log
    pub max_events: usize,
    /// Loss history for charting
    pub loss_data: Vec<(f64, f64)>,
    /// Error bound history
    pub error_bound_data: Vec<(f64, f64)>,
    /// Scroll offset for event log
    pub log_scroll: usize,
    /// Whether visualization is paused
    pub paused: bool,
    /// Start time
    pub start_time: Instant,
    /// Selected worker index (for detail view)
    pub selected_worker: usize,
    /// Current round
    pub current_round: u64,
    /// Total rounds
    pub total_rounds: u64,
    /// Current loss
    pub current_loss: f64,
    /// Current error bound
    pub current_error_bound: f64,
    /// Max error bound
    pub max_error_bound: f64,
    /// Active workers
    pub active_workers: u32,
    /// Total workers
    pub total_workers: u32,
    /// Proofs generated
    pub proofs_generated: u64,
    /// Proofs verified
    pub proofs_verified: u64,
    /// Current proof generation phase
    pub proof_phase: ProofPhase,
    /// Proof generation progress (0-100)
    pub proof_progress: u8,
    /// Whether proof is being generated
    pub proof_generating: bool,
    /// Current training phase
    pub training_phase: TrainingPhase,
    /// Whether connected to real node
    pub connected_to_node: bool,
    /// GPU acceleration enabled
    pub gpu_accelerated: bool,
    /// Proof constraints satisfied
    pub constraints_satisfied: u64,
    /// Total proof constraints
    pub total_constraints: u64,
    /// Memory usage (bytes)
    pub memory_usage: u64,
    // Wallet/staking fields
    /// Wallet address (truncated for display)
    pub wallet_address: Option<String>,
    /// Staked amount in HLX tokens
    pub staked_amount: f64,
    /// Pending rewards in HLX tokens
    pub pending_rewards: f64,
    /// Total rewards claimed
    pub total_rewards_claimed: f64,
    /// Staking APY percentage
    pub staking_apy: f64,
    /// Hardware wallet connected
    pub hardware_wallet_connected: bool,
    /// Hardware wallet model name
    pub hardware_wallet_model: Option<String>,
}

/// Visualization event
#[derive(Clone)]
pub struct VisualizationEvent {
    /// Event timestamp (ms since start)
    pub timestamp_ms: u64,
    /// Event type string
    pub event_type: String,
    /// Description
    pub description: String,
    /// Color for display
    pub color: Color,
}

impl VisualizationState {
    /// Create new visualization state
    pub fn new() -> Self {
        Self {
            selected_tab: 0,
            event_log: VecDeque::with_capacity(100),
            max_events: 100,
            loss_data: Vec::new(),
            error_bound_data: Vec::new(),
            log_scroll: 0,
            paused: false,
            start_time: Instant::now(),
            selected_worker: 0,
            current_round: 0,
            total_rounds: 10,
            current_loss: 2.5,
            current_error_bound: 0.0,
            max_error_bound: 1000.0,
            active_workers: 5,
            total_workers: 5,
            proofs_generated: 0,
            proofs_verified: 0,
            proof_phase: ProofPhase::Idle,
            proof_progress: 0,
            proof_generating: false,
            training_phase: TrainingPhase::Idle,
            connected_to_node: false,
            gpu_accelerated: false,
            constraints_satisfied: 0,
            total_constraints: 0,
            memory_usage: 0,
            // Wallet/staking defaults
            wallet_address: None,
            staked_amount: 0.0,
            pending_rewards: 0.0,
            total_rewards_claimed: 0.0,
            staking_apy: 12.5, // Default APY
            hardware_wallet_connected: false,
            hardware_wallet_model: None,
        }
    }

    /// Update from real-time proof tracker
    pub fn update_from_proof_status(&mut self, status: &ProofStatus) {
        self.proof_phase = status.phase;
        self.proof_progress = status.progress_percent;
        self.proof_generating = status.generating;
        self.constraints_satisfied = status.constraints_satisfied;
        self.total_constraints = status.total_constraints;
        self.gpu_accelerated = status.gpu_accelerated;
        self.memory_usage = status.memory_usage_bytes;
    }

    /// Update from real-time training tracker
    pub fn update_from_training_status(&mut self, status: &TrainingStatus) {
        self.current_round = status.current_round;
        self.total_rounds = status.total_rounds;
        self.current_loss = status.current_loss;
        self.current_error_bound = status.accumulated_error;
        self.max_error_bound = status.max_error_bound;
        self.training_phase = status.phase;
    }

    /// Set connection status
    pub fn set_connected(&mut self, connected: bool) {
        self.connected_to_node = connected;
    }

    /// Update wallet information
    pub fn update_wallet_info(
        &mut self,
        address: Option<String>,
        staked: f64,
        rewards: f64,
        apy: f64,
    ) {
        self.wallet_address = address;
        self.staked_amount = staked;
        self.pending_rewards = rewards;
        self.staking_apy = apy;
    }

    /// Set hardware wallet connection status
    pub fn set_hardware_wallet(&mut self, connected: bool, model: Option<String>) {
        self.hardware_wallet_connected = connected;
        self.hardware_wallet_model = model;
    }

    /// Add rewards (called when rewards are earned)
    pub fn add_rewards(&mut self, amount: f64) {
        self.pending_rewards += amount;
    }

    /// Claim rewards (move from pending to claimed)
    pub fn claim_rewards(&mut self) {
        self.total_rewards_claimed += self.pending_rewards;
        self.pending_rewards = 0.0;
    }

    /// Add an event to the log
    pub fn add_event(&mut self, event: VisualizationEvent) {
        if self.event_log.len() >= self.max_events {
            self.event_log.pop_front();
        }
        self.event_log.push_back(event);
    }

    /// Update training metrics
    pub fn update_metrics(&mut self, round: u64, loss: f64, error_bound: f64) {
        self.current_round = round;
        self.current_loss = loss;
        self.current_error_bound = error_bound;
        self.loss_data.push((round as f64, loss));
        self.error_bound_data.push((round as f64, error_bound));
    }

    /// Toggle pause state
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Select next tab
    pub fn next_tab(&mut self) {
        self.selected_tab = (self.selected_tab + 1) % 4;
    }

    /// Select previous tab
    pub fn prev_tab(&mut self) {
        self.selected_tab = if self.selected_tab == 0 { 3 } else { self.selected_tab - 1 };
    }

    /// Select next worker
    pub fn next_worker(&mut self) {
        self.selected_worker = (self.selected_worker + 1) % self.total_workers as usize;
    }

    /// Select previous worker
    pub fn prev_worker(&mut self) {
        let total = self.total_workers as usize;
        self.selected_worker = if self.selected_worker == 0 { total - 1 } else { self.selected_worker - 1 };
    }

    /// Scroll event log down
    pub fn scroll_down(&mut self) {
        if self.log_scroll < self.event_log.len().saturating_sub(10) {
            self.log_scroll += 1;
        }
    }

    /// Scroll event log up
    pub fn scroll_up(&mut self) {
        if self.log_scroll > 0 {
            self.log_scroll -= 1;
        }
    }
}

impl Default for VisualizationState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Terminal Visualization Runner
// ============================================================================

/// Main visualization runner
pub struct VisualizationRunner {
    /// Terminal backend
    terminal: Terminal<CrosstermBackend<Stdout>>,
    /// Visualization state
    state: VisualizationState,
    /// Tick rate for UI updates
    tick_rate: Duration,
    /// Real-time proof tracker
    proof_tracker: Option<Arc<RealTimeProofTracker>>,
    /// Real-time training tracker
    training_tracker: Option<Arc<RealTimeTrainingTracker>>,
}

impl VisualizationRunner {
    /// Create a new visualization runner
    pub fn new() -> Result<Self> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        Ok(Self {
            terminal,
            state: VisualizationState::new(),
            tick_rate: Duration::from_millis(100),
            proof_tracker: None,
            training_tracker: None,
        })
    }

    /// Create visualization with real-time trackers
    pub fn with_trackers(
        proof_tracker: Arc<RealTimeProofTracker>,
        training_tracker: Arc<RealTimeTrainingTracker>,
    ) -> Result<Self> {
        let mut runner = Self::new()?;
        runner.proof_tracker = Some(proof_tracker);
        runner.training_tracker = Some(training_tracker);
        Ok(runner)
    }

    /// Set proof tracker
    pub fn set_proof_tracker(&mut self, tracker: Arc<RealTimeProofTracker>) {
        self.proof_tracker = Some(tracker);
    }

    /// Set training tracker
    pub fn set_training_tracker(&mut self, tracker: Arc<RealTimeTrainingTracker>) {
        self.training_tracker = Some(tracker);
    }

    /// Get mutable state reference
    pub fn state_mut(&mut self) -> &mut VisualizationState {
        &mut self.state
    }

    /// Update state from trackers
    async fn update_from_trackers(&mut self) {
        if let Some(ref tracker) = self.proof_tracker {
            let status = tracker.get_status().await;
            self.state.update_from_proof_status(&status);
        }

        if let Some(ref tracker) = self.training_tracker {
            let status = tracker.get_status().await;
            self.state.update_from_training_status(&status);
        }
    }

    /// Run the visualization loop
    pub async fn run(&mut self, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
        let tick_rate = self.tick_rate;
        let mut last_tick = Instant::now();
        let mut simulation_tick = 0u64;

        loop {
            // Draw UI
            let state = &self.state;
            self.terminal.draw(|f| {
                draw_ui(f, state);
            })?;

            // Handle events
            let timeout = tick_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));

            if crossterm::event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => {
                                return Ok(());
                            }
                            KeyCode::Tab => self.state.next_tab(),
                            KeyCode::BackTab => self.state.prev_tab(),
                            KeyCode::Char('p') => self.state.toggle_pause(),
                            KeyCode::Down | KeyCode::Char('j') => self.state.scroll_down(),
                            KeyCode::Up | KeyCode::Char('k') => self.state.scroll_up(),
                            KeyCode::Right | KeyCode::Char('l') => self.state.next_worker(),
                            KeyCode::Left | KeyCode::Char('h') => self.state.prev_worker(),
                            KeyCode::Char('1') => self.state.selected_tab = 0,
                            KeyCode::Char('2') => self.state.selected_tab = 1,
                            KeyCode::Char('3') => self.state.selected_tab = 2,
                            KeyCode::Char('4') => self.state.selected_tab = 3,
                            _ => {}
                        }
                    }
                }
            }

            // Check for shutdown
            if shutdown.try_recv().is_ok() {
                return Ok(());
            }

            // Update from real-time trackers or simulate progress
            if last_tick.elapsed() >= tick_rate && !self.state.paused {
                simulation_tick += 1;

                // If we have trackers, use them for updates
                let has_trackers = self.proof_tracker.is_some() || self.training_tracker.is_some();

                if has_trackers {
                    // Update from real-time trackers
                    self.update_from_trackers().await;
                } else {
                    // Simulate demo progress (fallback when no trackers)
                    // Every 50 ticks, advance a round
                    if simulation_tick % 50 == 0 && self.state.current_round < self.state.total_rounds {
                        self.state.current_round += 1;
                        let loss = self.state.current_loss * 0.92;
                        let error = self.state.current_error_bound + 5.0 + (rand::random::<f64>() * 3.0);
                        self.state.update_metrics(self.state.current_round, loss, error);

                        self.state.proofs_generated += self.state.active_workers as u64;
                        self.state.proofs_verified += self.state.active_workers as u64;

                        self.state.add_event(VisualizationEvent {
                            timestamp_ms: self.state.start_time.elapsed().as_millis() as u64,
                            event_type: "RoundCompleted".to_string(),
                            description: format!("Round {} completed", self.state.current_round),
                            color: Color::Green,
                        });
                    }

                    // Simulate proof generation phases
                    if simulation_tick % 10 == 0 {
                        let phase_idx = (simulation_tick / 10) % 5;
                        self.state.proof_phase = match phase_idx {
                            0 => ProofPhase::WitnessGeneration,
                            1 => ProofPhase::CommitmentGeneration,
                            2 => ProofPhase::ProofComputation,
                            3 => ProofPhase::Complete,
                            _ => ProofPhase::Idle,
                        };
                        self.state.proof_generating = phase_idx < 3;
                        self.state.proof_progress = ((phase_idx + 1) * 25).min(100) as u8;
                    }
                }

                last_tick = Instant::now();
            }
        }
    }

    /// Cleanup terminal on exit
    pub fn cleanup(&mut self) -> Result<()> {
        disable_raw_mode()?;
        execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        self.terminal.show_cursor()?;
        Ok(())
    }
}

impl Drop for VisualizationRunner {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

// ============================================================================
// UI Drawing Functions
// ============================================================================

fn draw_ui(frame: &mut Frame, state: &VisualizationState) {
    let size = frame.size();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Length(3),  // Tabs
            Constraint::Min(10),    // Main content
            Constraint::Length(3),  // Footer
        ])
        .split(size);

    // Draw header
    draw_header(frame, chunks[0], state);

    // Draw tabs
    draw_tabs(frame, chunks[1], state);

    // Draw main content based on selected tab
    match state.selected_tab {
        0 => draw_overview_tab(frame, chunks[2], state),
        1 => draw_training_tab(frame, chunks[2], state),
        2 => draw_workers_tab(frame, chunks[2], state),
        3 => draw_events_tab(frame, chunks[2], state),
        _ => {}
    }

    // Draw footer
    draw_footer(frame, chunks[3]);
}

fn draw_header(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let elapsed = state.start_time.elapsed();
    let elapsed_str = format!(
        "{:02}:{:02}:{:02}",
        elapsed.as_secs() / 3600,
        (elapsed.as_secs() % 3600) / 60,
        elapsed.as_secs() % 60
    );

    let pause_indicator = if state.paused { " [PAUSED]" } else { "" };

    let header = Paragraph::new(format!(
        " HELIX Training Monitor | Elapsed: {} | Status: RUNNING{}",
        elapsed_str,
        pause_indicator
    ))
    .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)));

    frame.render_widget(header, area);
}

fn draw_tabs(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let tabs = Tabs::new(vec![
        Line::from(" Overview "),
        Line::from(" Training "),
        Line::from(" Workers "),
        Line::from(" Events "),
    ])
    .select(state.selected_tab)
    .style(Style::default().fg(Color::White))
    .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
    .divider(symbols::DOT)
    .block(Block::default().borders(Borders::ALL).title(" Tabs [1-4] "));

    frame.render_widget(tabs, area);
}

fn draw_overview_tab(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(35), Constraint::Percentage(30)])
        .split(chunks[0]);

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // Status panel
    draw_status_panel(frame, left_chunks[0], state);

    // Metrics panel
    draw_metrics_panel(frame, left_chunks[1], state);

    // Wallet/staking panel
    draw_wallet_panel(frame, left_chunks[2], state);

    // Loss chart
    draw_loss_chart(frame, right_chunks[0], state);

    // Network status
    draw_network_panel(frame, right_chunks[1], state);
}

fn draw_status_panel(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let phase = if state.current_round >= state.total_rounds {
        "Complete"
    } else if state.proof_generating {
        "Proving"
    } else {
        state.training_phase.name()
    };

    let phase_color = if state.current_round >= state.total_rounds {
        Color::Green
    } else if state.proof_generating {
        Color::Cyan
    } else {
        Color::Yellow
    };

    let mode = if state.connected_to_node {
        ("Real", Color::Green)
    } else {
        ("Demo", Color::Yellow)
    };

    let status_text = vec![
        Line::from(vec![
            Span::styled(" Mode:     ", Style::default().fg(Color::Gray)),
            Span::styled(mode.0, Style::default().fg(mode.1).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(" Phase:    ", Style::default().fg(Color::Gray)),
            Span::styled(phase, Style::default().fg(phase_color).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(" Model:    ", Style::default().fg(Color::Gray)),
            Span::raw("Demo Model (ID: 0)"),
        ]),
        Line::from(vec![
            Span::styled(" Round:    ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}/{}", state.current_round, state.total_rounds), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled(" Progress: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}%", state.current_round * 100 / state.total_rounds.max(1)), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled(" GPU:      ", Style::default().fg(Color::Gray)),
            if state.gpu_accelerated {
                Span::styled("Enabled", Style::default().fg(Color::Green))
            } else {
                Span::styled("Disabled", Style::default().fg(Color::Red))
            },
        ]),
    ];

    let status = Paragraph::new(status_text)
        .block(Block::default().borders(Borders::ALL).title(" Status "));

    frame.render_widget(status, area);
}

fn draw_metrics_panel(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let progress = state.current_round as f64 / state.total_rounds.max(1) as f64;
    let error_ratio = state.current_error_bound / state.max_error_bound;

    let inner_chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Min(0),
        ])
        .split(area);

    // Training progress gauge
    let training_gauge = Gauge::default()
        .block(Block::default().title("Training Progress"))
        .gauge_style(Style::default().fg(Color::Green))
        .ratio(progress.min(1.0))
        .label(format!("{}%", (progress * 100.0) as u32));

    // Proof generation gauge
    let proof_label = if state.proof_generating {
        format!("{} {}%", state.proof_phase.name(), state.proof_progress)
    } else {
        "Idle".to_string()
    };
    let proof_gauge = Gauge::default()
        .block(Block::default().title("Proof Generation"))
        .gauge_style(Style::default().fg(if state.proof_generating { Color::Cyan } else { Color::DarkGray }))
        .ratio(state.proof_progress as f64 / 100.0)
        .label(proof_label);

    // Proofs verified gauge
    let proofs_ratio = if state.proofs_generated > 0 {
        state.proofs_verified as f64 / state.proofs_generated as f64
    } else {
        0.0
    };
    let proofs_gauge = Gauge::default()
        .block(Block::default().title("Proofs Verified"))
        .gauge_style(Style::default().fg(Color::Blue))
        .ratio(proofs_ratio.min(1.0))
        .label(format!("{}/{}", state.proofs_verified, state.proofs_generated));

    // Error bound gauge
    let error_gauge = Gauge::default()
        .block(Block::default().title("Error Bound"))
        .gauge_style(Style::default().fg(Color::Yellow))
        .ratio(error_ratio.min(1.0))
        .label(format!("{:.0}/{:.0}", state.current_error_bound, state.max_error_bound));

    // Workers gauge
    let workers_ratio = state.active_workers as f64 / state.total_workers.max(1) as f64;
    let workers_gauge = Gauge::default()
        .block(Block::default().title("Active Workers"))
        .gauge_style(Style::default().fg(Color::Magenta))
        .ratio(workers_ratio.min(1.0))
        .label(format!("{}/{}", state.active_workers, state.total_workers));

    let outer_block = Block::default().borders(Borders::ALL).title(" Metrics ");
    frame.render_widget(outer_block, area);

    frame.render_widget(training_gauge, inner_chunks[0]);
    frame.render_widget(proof_gauge, inner_chunks[1]);
    frame.render_widget(proofs_gauge, inner_chunks[2]);
    frame.render_widget(error_gauge, inner_chunks[3]);
    frame.render_widget(workers_gauge, inner_chunks[4]);
}

fn draw_wallet_panel(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let wallet_display = state.wallet_address.as_ref()
        .map(|addr| {
            if addr.len() > 12 {
                format!("{}...{}", &addr[..6], &addr[addr.len()-4..])
            } else {
                addr.clone()
            }
        })
        .unwrap_or_else(|| "Not Connected".to_string());

    let wallet_color = if state.wallet_address.is_some() {
        Color::Green
    } else {
        Color::Red
    };

    let hw_status = if state.hardware_wallet_connected {
        let model = state.hardware_wallet_model.as_deref().unwrap_or("Unknown");
        (format!("{}", model), Color::Green)
    } else {
        ("Not Connected".to_string(), Color::DarkGray)
    };

    let wallet_text = vec![
        Line::from(vec![
            Span::styled(" Wallet:  ", Style::default().fg(Color::Gray)),
            Span::styled(wallet_display, Style::default().fg(wallet_color)),
        ]),
        Line::from(vec![
            Span::styled(" HW:      ", Style::default().fg(Color::Gray)),
            Span::styled(hw_status.0, Style::default().fg(hw_status.1)),
        ]),
        Line::from(vec![
            Span::styled(" Staked:  ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.2} HLX", state.staked_amount),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled(" Rewards: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.4} HLX", state.pending_rewards),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled(" APY:     ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1}%", state.staking_apy),
                Style::default().fg(Color::Green),
            ),
        ]),
    ];

    let wallet = Paragraph::new(wallet_text)
        .block(Block::default().borders(Borders::ALL).title(" Wallet & Staking "));

    frame.render_widget(wallet, area);
}

fn draw_loss_chart(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let loss_data: Vec<(f64, f64)> = if state.loss_data.is_empty() {
        (0..10)
            .map(|i| (i as f64, 2.5 - (i as f64 * 0.15)))
            .collect()
    } else {
        state.loss_data.clone()
    };

    let datasets = vec![
        Dataset::default()
            .name("Loss")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Cyan))
            .graph_type(GraphType::Line)
            .data(&loss_data),
    ];

    let max_round = state.total_rounds as f64;
    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title(" Training Loss "))
        .x_axis(
            Axis::default()
                .title("Round")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, max_round])
                .labels(vec!["0".into(), format!("{}", max_round as u64 / 2).into(), format!("{}", max_round as u64).into()]),
        )
        .y_axis(
            Axis::default()
                .title("Loss")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, 3.0])
                .labels(vec!["0".into(), "1.5".into(), "3.0".into()]),
        );

    frame.render_widget(chart, area);
}

fn draw_network_panel(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let worker_symbols: String = (0..state.total_workers)
        .map(|i| if i < state.active_workers { "●" } else { "○" })
        .collect::<Vec<_>>()
        .join("");

    let network_text = vec![
        Line::from(vec![
            Span::styled(" Workers:     ", Style::default().fg(Color::Gray)),
            Span::styled(&worker_symbols, Style::default().fg(Color::Green)),
            Span::raw(format!(" {}/{} active", state.active_workers, state.total_workers)),
        ]),
        Line::from(vec![
            Span::styled(" Aggregators: ", Style::default().fg(Color::Gray)),
            Span::styled("●●", Style::default().fg(Color::Green)),
            Span::raw(" 2/2 active"),
        ]),
        Line::from(vec![
            Span::styled(" Peers:       ", Style::default().fg(Color::Gray)),
            Span::raw(format!("{} connected", state.total_workers + 2)),
        ]),
        Line::from(vec![
            Span::styled(" Block:       ", Style::default().fg(Color::Gray)),
            Span::raw("#12,345,678"),
        ]),
        Line::from(vec![
            Span::styled(" Chain:       ", Style::default().fg(Color::Gray)),
            Span::raw("31337 (local)"),
        ]),
    ];

    let network = Paragraph::new(network_text)
        .block(Block::default().borders(Borders::ALL).title(" Network "));

    frame.render_widget(network, area);
}

fn draw_training_tab(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // Loss and error bound charts
    let chart_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[0]);

    draw_loss_chart(frame, chart_chunks[0], state);
    draw_error_bound_chart(frame, chart_chunks[1], state);

    // Round details
    draw_round_details(frame, chunks[1], state);
}

fn draw_error_bound_chart(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let error_data: Vec<(f64, f64)> = if state.error_bound_data.is_empty() {
        (0..10)
            .map(|i| (i as f64, i as f64 * 8.0))
            .collect()
    } else {
        state.error_bound_data.clone()
    };

    let datasets = vec![
        Dataset::default()
            .name("Error Bound")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Yellow))
            .graph_type(GraphType::Line)
            .data(&error_data),
    ];

    let max_round = state.total_rounds as f64;
    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title(" Error Bound Accumulation "))
        .x_axis(
            Axis::default()
                .title("Round")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, max_round])
                .labels(vec!["0".into(), format!("{}", max_round as u64 / 2).into(), format!("{}", max_round as u64).into()]),
        )
        .y_axis(
            Axis::default()
                .title("Error")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, state.max_error_bound])
                .labels(vec!["0".into(), format!("{}", state.max_error_bound as u64 / 2).into(), format!("{}", state.max_error_bound as u64).into()]),
        );

    frame.render_widget(chart, area);
}

fn draw_round_details(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let header_cells = ["Round", "Status", "Loss", "Error Δ", "Proofs", "Duration"];
    let header = Row::new(header_cells.iter().map(|h| {
        ratatui::widgets::Cell::from(*h).style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
    })).height(1);

    let rows: Vec<Row> = (1..=state.current_round.min(10)).map(|i| {
        let loss = 2.5 - (i as f64 * 0.15);
        let error = 5.0 + (i as f64 * 0.5);
        Row::new(vec![
            format!("{}", i),
            if i < state.current_round { "Complete".to_string() } else { "In Progress".to_string() },
            format!("{:.4}", loss),
            format!("+{:.1}", error),
            format!("{}", state.active_workers),
            format!("{}s", 8 + (i % 5)),
        ]).height(1)
    }).collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(10),
            Constraint::Percentage(20),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" Round History "));

    frame.render_widget(table, area);
}

fn draw_workers_tab(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // Worker table
    draw_worker_table(frame, chunks[0], state);

    // Worker detail
    draw_worker_detail(frame, chunks[1], state);
}

fn draw_worker_table(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let header_cells = ["ID", "Status", "Stake", "Proofs", "Verified", "Reputation"];
    let header = Row::new(header_cells.iter().map(|h| {
        ratatui::widgets::Cell::from(*h).style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
    })).height(1);

    let rows: Vec<Row> = (0..state.total_workers).map(|i| {
        let is_selected = i as usize == state.selected_worker;
        let is_active = i < state.active_workers;
        let style = if is_selected {
            Style::default().bg(Color::DarkGray)
        } else {
            Style::default()
        };

        let status = if is_active { "Training" } else { "Offline" };
        let status_style = if is_active {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::Red)
        };

        Row::new(vec![
            ratatui::widgets::Cell::from(format!("worker-{}", i + 1)),
            ratatui::widgets::Cell::from(status).style(status_style),
            ratatui::widgets::Cell::from("1.0 ETH"),
            ratatui::widgets::Cell::from(format!("{}", state.proofs_generated / state.total_workers as u64)),
            ratatui::widgets::Cell::from(format!("{}", state.proofs_verified / state.total_workers as u64)),
            ratatui::widgets::Cell::from("100%"),
        ]).style(style).height(1)
    }).collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(15),
            Constraint::Percentage(18),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(" Workers [arrow keys to select] "));

    frame.render_widget(table, area);
}

fn draw_worker_detail(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let worker_id = format!("worker-{}", state.selected_worker + 1);
    let is_active = (state.selected_worker as u32) < state.active_workers;

    let detail_text = vec![
        Line::from(vec![
            Span::styled(" Worker ID:    ", Style::default().fg(Color::Gray)),
            Span::styled(&worker_id, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(" Address:      ", Style::default().fg(Color::Gray)),
            Span::raw(format!("0x742d35Cc6634C{}...", state.selected_worker)),
        ]),
        Line::from(vec![
            Span::styled(" Status:       ", Style::default().fg(Color::Gray)),
            if is_active {
                Span::styled("Training", Style::default().fg(Color::Green))
            } else {
                Span::styled("Offline", Style::default().fg(Color::Red))
            },
        ]),
        Line::from(vec![
            Span::styled(" Stake:        ", Style::default().fg(Color::Gray)),
            Span::raw("1.0 ETH"),
        ]),
        Line::from(vec![
            Span::styled(" Lock Period:  ", Style::default().fg(Color::Gray)),
            Span::raw("5d 12h remaining"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(" === Performance ===", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled(" Proofs Submitted: ", Style::default().fg(Color::Gray)),
            Span::raw(format!("{}", state.proofs_generated / state.total_workers.max(1) as u64)),
        ]),
        Line::from(vec![
            Span::styled(" Proofs Verified:  ", Style::default().fg(Color::Gray)),
            Span::raw(format!("{}", state.proofs_verified / state.total_workers.max(1) as u64)),
        ]),
        Line::from(vec![
            Span::styled(" Reputation:       ", Style::default().fg(Color::Gray)),
            Span::styled("100%", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled(" Slashed:          ", Style::default().fg(Color::Gray)),
            Span::styled("No", Style::default().fg(Color::Green)),
        ]),
    ];

    let detail = Paragraph::new(detail_text)
        .block(Block::default().borders(Borders::ALL).title(" Worker Detail "));

    frame.render_widget(detail, area);
}

fn draw_events_tab(frame: &mut Frame, area: Rect, state: &VisualizationState) {
    let events: Vec<ListItem> = state.event_log.iter().rev().take(20).map(|event| {
        ListItem::new(Line::from(vec![
            Span::styled(format!(" {:>8}ms ", event.timestamp_ms), Style::default().fg(Color::DarkGray)),
            Span::styled(format!("[{:15}] ", event.event_type), Style::default().fg(event.color)),
            Span::raw(event.description.clone()),
        ]))
    }).collect();

    // Add some sample events if empty
    let events = if events.is_empty() {
        vec![
            create_event_item("14:30:01", "RoundCompleted", "Round 5 committed on-chain", Color::Green),
            create_event_item("14:30:00", "ProofVerified", "Worker 3 proof verified", Color::Cyan),
            create_event_item("14:29:58", "ProofGenerated", "Worker 3 submitted proof", Color::Blue),
            create_event_item("14:29:55", "ProofVerified", "Worker 2 proof verified", Color::Cyan),
            create_event_item("14:29:53", "ProofGenerated", "Worker 2 submitted proof", Color::Blue),
            create_event_item("14:29:50", "ProofVerified", "Worker 1 proof verified", Color::Cyan),
            create_event_item("14:29:48", "ProofGenerated", "Worker 1 submitted proof", Color::Blue),
            create_event_item("14:29:45", "RoundStarted", "Round 5 initiated", Color::Yellow),
        ]
    } else {
        events
    };

    let events_list = List::new(events)
        .block(Block::default().borders(Borders::ALL).title(" Event Log [arrow keys to scroll] "));

    frame.render_widget(events_list, area);
}

fn draw_footer(frame: &mut Frame, area: Rect) {
    let help_text = " [Tab] Switch tabs | [1-4] Jump to tab | [arrows] Scroll/Select | [P] Pause | [Q] Quit ";

    let footer = Paragraph::new(help_text)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));

    frame.render_widget(footer, area);
}

fn create_event_item(time: &str, event_type: &str, description: &str, color: Color) -> ListItem<'static> {
    ListItem::new(Line::from(vec![
        Span::styled(format!(" {} ", time), Style::default().fg(Color::DarkGray)),
        Span::styled(format!("[{:15}] ", event_type), Style::default().fg(color)),
        Span::raw(description.to_string()),
    ]))
}

// ============================================================================
// Standalone Visualization Functions
// ============================================================================

/// Display a compact training status in the terminal (non-interactive)
pub fn print_training_status(
    round: u64,
    total_rounds: u64,
    loss: f64,
    error_bound: f64,
    max_error: f64,
    workers_active: u32,
    workers_total: u32,
) {
    let progress = (round as f64 / total_rounds as f64 * 100.0) as u32;
    let error_ratio = error_bound / max_error;

    // Progress bar
    let bar_width = 30;
    let filled = (progress as usize * bar_width / 100).min(bar_width);
    let bar: String = format!(
        "[{}{}]",
        "#".repeat(filled),
        "-".repeat(bar_width - filled)
    );

    println!(
        "Round {}/{} {} {}%",
        round,
        total_rounds,
        bar,
        progress
    );
    println!(
        "  Loss: {:.6}  Error: {:.1}/{:.1} ({:.1}%)  Workers: {}/{}",
        loss,
        error_bound,
        max_error,
        error_ratio * 100.0,
        workers_active,
        workers_total
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualization_state_creation() {
        let vis_state = VisualizationState::new();
        assert_eq!(vis_state.selected_tab, 0);
        assert!(!vis_state.paused);
        assert_eq!(vis_state.event_log.len(), 0);
    }

    #[test]
    fn test_tab_navigation() {
        let mut vis_state = VisualizationState::new();

        assert_eq!(vis_state.selected_tab, 0);
        vis_state.next_tab();
        assert_eq!(vis_state.selected_tab, 1);
        vis_state.next_tab();
        assert_eq!(vis_state.selected_tab, 2);
        vis_state.prev_tab();
        assert_eq!(vis_state.selected_tab, 1);
    }

    #[test]
    fn test_event_log() {
        let mut vis_state = VisualizationState::new();

        let event = VisualizationEvent {
            timestamp_ms: 1000,
            event_type: "RoundStarted".to_string(),
            description: "Test event".to_string(),
            color: Color::Green,
        };

        vis_state.add_event(event.clone());
        assert_eq!(vis_state.event_log.len(), 1);
    }
}
