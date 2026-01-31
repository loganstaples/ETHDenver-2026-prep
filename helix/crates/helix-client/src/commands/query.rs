//! Query command implementation
//!
//! Provides querying functionality for model state, round information,
//! stake info, proofs, and error bounds.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use colored::*;
use serde::{Deserialize, Serialize};

use crate::progress::ProgressDisplay;

/// Query types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Query model information
    Model,
    /// Query round information
    Round,
    /// Query stake information
    Stake,
    /// Query proof information
    Proof,
    /// Query error bound information
    Error,
    /// Query worker information
    Worker,
    /// Query aggregator information
    Aggregator,
    /// Query metrics
    Metrics,
}

impl std::fmt::Display for QueryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryType::Model => write!(f, "Model"),
            QueryType::Round => write!(f, "Round"),
            QueryType::Stake => write!(f, "Stake"),
            QueryType::Proof => write!(f, "Proof"),
            QueryType::Error => write!(f, "Error"),
            QueryType::Worker => write!(f, "Worker"),
            QueryType::Aggregator => write!(f, "Aggregator"),
            QueryType::Metrics => write!(f, "Metrics"),
        }
    }
}

/// Query command options
#[derive(Debug, Clone)]
pub struct QueryOptions {
    /// Query type
    pub query_type: QueryType,
    /// Model ID
    pub model_id: u64,
    /// Round ID (for round queries)
    pub round_id: Option<u64>,
    /// Worker address (for worker queries)
    pub worker: Option<String>,
    /// Output format
    pub format: QueryFormat,
    /// Include historical data
    pub history: bool,
    /// Number of historical entries
    pub history_count: usize,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            query_type: QueryType::Model,
            model_id: 0,
            round_id: None,
            worker: None,
            format: QueryFormat::Text,
            history: false,
            history_count: 10,
        }
    }
}

/// Query output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryFormat {
    Text,
    Json,
    Table,
}

/// Query result types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum QueryResult {
    Model(ModelQueryResult),
    Round(RoundQueryResult),
    Stake(StakeQueryResult),
    Proof(ProofQueryResult),
    Error(ErrorQueryResult),
    Worker(WorkerQueryResult),
    Aggregator(AggregatorQueryResult),
    Metrics(MetricsQueryResult),
}

/// Model query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelQueryResult {
    /// Model ID
    pub id: u64,
    /// Model name
    pub name: String,
    /// IPFS hash
    pub ipfs_hash: String,
    /// Current state commitment
    pub state_commitment: String,
    /// Current round
    pub current_round: u64,
    /// Total rounds
    pub total_rounds: u64,
    /// Minimum stake required
    pub min_stake: f64,
    /// Is active
    pub is_active: bool,
    /// Owner address
    pub owner: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Number of active workers
    pub active_workers: u32,
    /// Total stake locked
    pub total_stake: f64,
    /// Architecture info
    pub architecture: Option<ModelArchitecture>,
}

/// Model architecture information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// Model type
    pub model_type: String,
    /// Number of parameters
    pub parameters: u64,
    /// Number of layers
    pub layers: u32,
    /// Hidden dimension
    pub hidden_dim: u32,
    /// Precision level
    pub precision: String,
}

/// Round query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundQueryResult {
    /// Model ID
    pub model_id: u64,
    /// Round ID
    pub round_id: u64,
    /// Previous state commitment
    pub prev_commitment: String,
    /// New state commitment
    pub new_commitment: String,
    /// Is completed
    pub completed: bool,
    /// Deadline timestamp
    pub deadline: chrono::DateTime<chrono::Utc>,
    /// Prover address
    pub prover: Option<String>,
    /// Proof hash
    pub proof_hash: Option<String>,
    /// Workers participated
    pub workers: Vec<String>,
    /// Aggregator address
    pub aggregator: String,
    /// Loss value
    pub loss: Option<f64>,
    /// Error bound for this round
    pub error_bound: f64,
    /// Round duration in ms
    pub duration_ms: u64,
}

/// Stake query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakeQueryResult {
    /// Model ID
    pub model_id: u64,
    /// Worker address
    pub worker: String,
    /// Staked amount
    pub amount: f64,
    /// Lock timestamp
    pub locked_at: chrono::DateTime<chrono::Utc>,
    /// Unlock timestamp
    pub unlocks_at: chrono::DateTime<chrono::Utc>,
    /// Is currently locked
    pub is_locked: bool,
    /// Has been slashed
    pub is_slashed: bool,
    /// Slashed amount
    pub slashed_amount: f64,
    /// Reputation score
    pub reputation: f64,
    /// Rounds participated
    pub rounds_participated: u64,
    /// Successful proofs
    pub successful_proofs: u64,
    /// Pending rewards
    pub pending_rewards: f64,
}

/// Proof query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofQueryResult {
    /// Model ID
    pub model_id: u64,
    /// Round ID
    pub round_id: u64,
    /// Proof hash
    pub proof_hash: String,
    /// Prover address
    pub prover: String,
    /// Is verified
    pub verified: bool,
    /// Verification timestamp
    pub verified_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Submitted timestamp
    pub submitted_at: chrono::DateTime<chrono::Utc>,
    /// Proof size in bytes
    pub size_bytes: u64,
    /// Verification gas used
    pub gas_used: Option<u64>,
    /// Error bounds proven
    pub error_bounds: ErrorBoundsProof,
}

/// Error bounds from proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBoundsProof {
    /// Forward pass error
    pub forward: f64,
    /// Backward pass error
    pub backward: f64,
    /// Gradient error
    pub gradient: f64,
    /// Total accumulated error
    pub total: f64,
    /// Maximum allowed
    pub max_allowed: f64,
}

/// Error query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorQueryResult {
    /// Model ID
    pub model_id: u64,
    /// Current round
    pub current_round: u64,
    /// Accumulated error
    pub accumulated_error: f64,
    /// Maximum allowed error
    pub max_allowed: f64,
    /// Error status
    pub status: ErrorStatus,
    /// Error history by round
    pub history: Vec<RoundErrorInfo>,
    /// Error budget remaining
    pub budget_remaining: f64,
    /// Projected error at completion
    pub projected_final: f64,
}

/// Error status
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ErrorStatus {
    /// Error is acceptable
    Acceptable,
    /// Error is approaching limit
    Warning,
    /// Error is critical
    Critical,
    /// Error exceeded limit
    Exceeded,
}

impl std::fmt::Display for ErrorStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorStatus::Acceptable => write!(f, "Acceptable"),
            ErrorStatus::Warning => write!(f, "Warning"),
            ErrorStatus::Critical => write!(f, "Critical"),
            ErrorStatus::Exceeded => write!(f, "Exceeded"),
        }
    }
}

/// Error info for a round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundErrorInfo {
    /// Round ID
    pub round_id: u64,
    /// Error for this round
    pub error: f64,
    /// Cumulative error after this round
    pub cumulative: f64,
}

/// Worker query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerQueryResult {
    /// Worker address
    pub address: String,
    /// Worker ID
    pub id: String,
    /// Models participating in
    pub models: Vec<u64>,
    /// Total stake
    pub total_stake: f64,
    /// Overall reputation
    pub reputation: f64,
    /// Rounds completed
    pub rounds_completed: u64,
    /// Proofs submitted
    pub proofs_submitted: u64,
    /// Proofs verified
    pub proofs_verified: u64,
    /// Proofs failed
    pub proofs_failed: u64,
    /// Slashing events
    pub slashing_events: u32,
    /// Total rewards earned
    pub total_rewards: f64,
    /// Status
    pub status: String,
    /// Last active
    pub last_active: chrono::DateTime<chrono::Utc>,
}

/// Aggregator query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorQueryResult {
    /// Aggregator address
    pub address: String,
    /// Aggregator ID
    pub id: String,
    /// Models aggregating for
    pub models: Vec<u64>,
    /// Rounds aggregated
    pub rounds_aggregated: u64,
    /// Successful aggregations
    pub successful: u64,
    /// Failed aggregations
    pub failed: u64,
    /// Average aggregation time
    pub avg_time_ms: u64,
    /// Total fees collected
    pub fees_collected: f64,
    /// Status
    pub status: String,
}

/// Metrics query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsQueryResult {
    /// Model ID
    pub model_id: u64,
    /// Training metrics
    pub training: TrainingMetrics,
    /// Proof metrics
    pub proofs: ProofMetrics,
    /// Network metrics
    pub network: NetworkMetrics,
    /// Economic metrics
    pub economics: EconomicMetrics,
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Rounds completed
    pub rounds_completed: u64,
    /// Total rounds
    pub total_rounds: u64,
    /// Average loss
    pub avg_loss: f64,
    /// Loss trend (improving/worsening)
    pub loss_trend: String,
    /// Average round time
    pub avg_round_time_ms: u64,
    /// Total training time
    pub total_time_seconds: u64,
}

/// Proof metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetrics {
    /// Total proofs generated
    pub total_generated: u64,
    /// Total proofs verified
    pub total_verified: u64,
    /// Verification rate
    pub verification_rate: f64,
    /// Average proof time
    pub avg_proof_time_ms: u64,
    /// Average verification time
    pub avg_verify_time_ms: u64,
    /// Total gas used
    pub total_gas_used: u64,
}

/// Network metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Active workers
    pub active_workers: u32,
    /// Active aggregators
    pub active_aggregators: u32,
    /// Total messages
    pub total_messages: u64,
    /// Average latency
    pub avg_latency_ms: u32,
    /// Bandwidth used (bytes)
    pub bandwidth_bytes: u64,
}

/// Economic metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicMetrics {
    /// Total stake locked
    pub total_stake: f64,
    /// Total rewards distributed
    pub total_rewards: f64,
    /// Total fees paid
    pub total_fees: f64,
    /// Slashing events
    pub slashing_events: u32,
    /// Total slashed
    pub total_slashed: f64,
}

/// Query command handler
pub struct QueryCommand {
    /// Progress display
    progress: ProgressDisplay,
}

impl QueryCommand {
    /// Create a new query command
    pub fn new() -> Self {
        Self {
            progress: ProgressDisplay::new(),
        }
    }

    /// Execute the query command
    pub async fn execute(&self, options: QueryOptions) -> Result<QueryResult> {
        match options.query_type {
            QueryType::Model => self.query_model(&options).await,
            QueryType::Round => self.query_round(&options).await,
            QueryType::Stake => self.query_stake(&options).await,
            QueryType::Proof => self.query_proof(&options).await,
            QueryType::Error => self.query_error(&options).await,
            QueryType::Worker => self.query_worker(&options).await,
            QueryType::Aggregator => self.query_aggregator(&options).await,
            QueryType::Metrics => self.query_metrics(&options).await,
        }
    }

    /// Query model information
    async fn query_model(&self, options: &QueryOptions) -> Result<QueryResult> {
        Ok(QueryResult::Model(ModelQueryResult {
            id: options.model_id,
            name: format!("helix-model-{}", options.model_id),
            ipfs_hash: "QmXoYPm8rPnxV3YHNqpGd8tVwL5c7sW9eZyMbTxNxNwXYZ".to_string(),
            state_commitment: format!("0x{}", hex::encode([0xAB; 32])),
            current_round: 42,
            total_rounds: 100,
            min_stake: 0.1,
            is_active: true,
            owner: "0x742d35Cc6634C0532925a3b844Bc454e4438f44e".to_string(),
            created_at: chrono::Utc::now() - chrono::Duration::days(7),
            updated_at: chrono::Utc::now(),
            active_workers: 4,
            total_stake: 6.0,
            architecture: Some(ModelArchitecture {
                model_type: "Transformer".to_string(),
                parameters: 1_500_000,
                layers: 6,
                hidden_dim: 256,
                precision: "FP16".to_string(),
            }),
        }))
    }

    /// Query round information
    async fn query_round(&self, options: &QueryOptions) -> Result<QueryResult> {
        let round_id = options.round_id.unwrap_or(42);

        Ok(QueryResult::Round(RoundQueryResult {
            model_id: options.model_id,
            round_id,
            prev_commitment: format!("0x{}", hex::encode([0x11; 32])),
            new_commitment: format!("0x{}", hex::encode([0x22; 32])),
            completed: true,
            deadline: chrono::Utc::now() + chrono::Duration::minutes(5),
            prover: Some("0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199".to_string()),
            proof_hash: Some(format!("0x{}", hex::encode([0xDE; 32]))),
            workers: vec![
                "0x742d35Cc6634C0532925a3b844Bc454e4438f44e".to_string(),
                "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199".to_string(),
                "0xdD2FD4581271e230360230F9337D5c0430Bf44C0".to_string(),
            ],
            aggregator: "0x5FbDB2315678afecb367f032d93F642f64180aa3".to_string(),
            loss: Some(0.234),
            error_bound: 1.2,
            duration_ms: 2500,
        }))
    }

    /// Query stake information
    async fn query_stake(&self, options: &QueryOptions) -> Result<QueryResult> {
        let worker = options.worker.clone()
            .unwrap_or_else(|| "0x742d35Cc6634C0532925a3b844Bc454e4438f44e".to_string());

        Ok(QueryResult::Stake(StakeQueryResult {
            model_id: options.model_id,
            worker,
            amount: 1.5,
            locked_at: chrono::Utc::now() - chrono::Duration::days(2),
            unlocks_at: chrono::Utc::now() + chrono::Duration::days(5),
            is_locked: true,
            is_slashed: false,
            slashed_amount: 0.0,
            reputation: 0.98,
            rounds_participated: 42,
            successful_proofs: 40,
            pending_rewards: 0.023,
        }))
    }

    /// Query proof information
    async fn query_proof(&self, options: &QueryOptions) -> Result<QueryResult> {
        let round_id = options.round_id.unwrap_or(42);

        Ok(QueryResult::Proof(ProofQueryResult {
            model_id: options.model_id,
            round_id,
            proof_hash: format!("0x{}", hex::encode([0xDE; 32])),
            prover: "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199".to_string(),
            verified: true,
            verified_at: Some(chrono::Utc::now()),
            submitted_at: chrono::Utc::now() - chrono::Duration::seconds(30),
            size_bytes: 4096,
            gas_used: Some(500_000),
            error_bounds: ErrorBoundsProof {
                forward: 0.5,
                backward: 0.4,
                gradient: 0.3,
                total: 1.2,
                max_allowed: 10.0,
            },
        }))
    }

    /// Query error information
    async fn query_error(&self, options: &QueryOptions) -> Result<QueryResult> {
        let history = if options.history {
            (1..=options.history_count.min(10))
                .map(|i| RoundErrorInfo {
                    round_id: 42 - i as u64,
                    error: 1.0 + i as f64 * 0.1,
                    cumulative: 42.0 + i as f64,
                })
                .collect()
        } else {
            vec![]
        };

        Ok(QueryResult::Error(ErrorQueryResult {
            model_id: options.model_id,
            current_round: 42,
            accumulated_error: 45.2,
            max_allowed: 1000.0,
            status: ErrorStatus::Acceptable,
            history,
            budget_remaining: 954.8,
            projected_final: 108.0,
        }))
    }

    /// Query worker information
    async fn query_worker(&self, options: &QueryOptions) -> Result<QueryResult> {
        let address = options.worker.clone()
            .unwrap_or_else(|| "0x742d35Cc6634C0532925a3b844Bc454e4438f44e".to_string());

        Ok(QueryResult::Worker(WorkerQueryResult {
            address: address.clone(),
            id: "helix-node-a1b2c3d4".to_string(),
            models: vec![0, 1],
            total_stake: 2.5,
            reputation: 0.98,
            rounds_completed: 127,
            proofs_submitted: 127,
            proofs_verified: 125,
            proofs_failed: 2,
            slashing_events: 0,
            total_rewards: 0.15,
            status: "active".to_string(),
            last_active: chrono::Utc::now(),
        }))
    }

    /// Query aggregator information
    async fn query_aggregator(&self, options: &QueryOptions) -> Result<QueryResult> {
        Ok(QueryResult::Aggregator(AggregatorQueryResult {
            address: "0x5FbDB2315678afecb367f032d93F642f64180aa3".to_string(),
            id: "helix-agg-01".to_string(),
            models: vec![0],
            rounds_aggregated: 42,
            successful: 42,
            failed: 0,
            avg_time_ms: 150,
            fees_collected: 0.042,
            status: "active".to_string(),
        }))
    }

    /// Query metrics
    async fn query_metrics(&self, options: &QueryOptions) -> Result<QueryResult> {
        Ok(QueryResult::Metrics(MetricsQueryResult {
            model_id: options.model_id,
            training: TrainingMetrics {
                rounds_completed: 42,
                total_rounds: 100,
                avg_loss: 0.312,
                loss_trend: "improving".to_string(),
                avg_round_time_ms: 2500,
                total_time_seconds: 105000,
            },
            proofs: ProofMetrics {
                total_generated: 127,
                total_verified: 125,
                verification_rate: 98.4,
                avg_proof_time_ms: 1200,
                avg_verify_time_ms: 50,
                total_gas_used: 62_500_000,
            },
            network: NetworkMetrics {
                active_workers: 4,
                active_aggregators: 1,
                total_messages: 30210,
                avg_latency_ms: 15,
                bandwidth_bytes: 156_789_012,
            },
            economics: EconomicMetrics {
                total_stake: 6.0,
                total_rewards: 0.15,
                total_fees: 0.042,
                slashing_events: 0,
                total_slashed: 0.0,
            },
        }))
    }

    /// Display query result
    pub fn display(&self, result: &QueryResult, format: QueryFormat) {
        match format {
            QueryFormat::Json => {
                println!("{}", serde_json::to_string_pretty(result).unwrap_or_default());
            }
            QueryFormat::Table | QueryFormat::Text => {
                self.display_text(result);
            }
        }
    }

    /// Display result as text
    fn display_text(&self, result: &QueryResult) {
        match result {
            QueryResult::Model(m) => self.display_model(m),
            QueryResult::Round(r) => self.display_round(r),
            QueryResult::Stake(s) => self.display_stake(s),
            QueryResult::Proof(p) => self.display_proof(p),
            QueryResult::Error(e) => self.display_error(e),
            QueryResult::Worker(w) => self.display_worker(w),
            QueryResult::Aggregator(a) => self.display_aggregator(a),
            QueryResult::Metrics(m) => self.display_metrics(m),
        }
    }

    fn display_model(&self, m: &ModelQueryResult) {
        println!("{}", "Model Information".cyan().bold());
        println!("  Model ID:       {}", m.id);
        println!("  Name:           {}", m.name);
        println!("  IPFS Hash:      {}", m.ipfs_hash);
        println!("  State:          {}...", &m.state_commitment[..20]);
        println!("  Round:          {}/{}", m.current_round, m.total_rounds);
        println!("  Min Stake:      {} ETH", m.min_stake);
        println!("  Active:         {} {}", if m.is_active { "●".green() } else { "●".red() }, if m.is_active { "Yes" } else { "No" });
        println!("  Owner:          {}", m.owner);
        println!("  Workers:        {}", m.active_workers);
        println!("  Total Stake:    {} ETH", m.total_stake);

        if let Some(arch) = &m.architecture {
            println!("\n{}", "Architecture:".yellow());
            println!("  Type:           {}", arch.model_type);
            println!("  Parameters:     {}", arch.parameters);
            println!("  Layers:         {}", arch.layers);
            println!("  Hidden Dim:     {}", arch.hidden_dim);
            println!("  Precision:      {}", arch.precision);
        }
    }

    fn display_round(&self, r: &RoundQueryResult) {
        println!("{}", "Round Information".cyan().bold());
        println!("  Model ID:       {}", r.model_id);
        println!("  Round ID:       {}", r.round_id);
        println!("  Prev Commit:    {}...", &r.prev_commitment[..20]);
        println!("  New Commit:     {}...", &r.new_commitment[..20]);
        println!("  Completed:      {} {}", if r.completed { "●".green() } else { "●".yellow() }, if r.completed { "Yes" } else { "No" });
        println!("  Aggregator:     {}", r.aggregator);
        if let Some(loss) = r.loss {
            println!("  Loss:           {:.6}", loss);
        }
        println!("  Error Bound:    {:.2}", r.error_bound);
        println!("  Duration:       {} ms", r.duration_ms);
        println!("  Workers:        {}", r.workers.len());
    }

    fn display_stake(&self, s: &StakeQueryResult) {
        println!("{}", "Stake Information".cyan().bold());
        println!("  Model ID:       {}", s.model_id);
        println!("  Worker:         {}", s.worker);
        println!("  Amount:         {} ETH", s.amount);
        println!("  Locked:         {} {}", if s.is_locked { "●".yellow() } else { "●".green() }, if s.is_locked { "Yes" } else { "No" });
        println!("  Slashed:        {} {}", if s.is_slashed { "●".red() } else { "●".green() }, if s.is_slashed { "Yes" } else { "No" });
        println!("  Reputation:     {:.0}%", s.reputation * 100.0);
        println!("  Rounds:         {}", s.rounds_participated);
        println!("  Proofs:         {}", s.successful_proofs);
        println!("  Rewards:        {} ETH (pending)", s.pending_rewards);
    }

    fn display_proof(&self, p: &ProofQueryResult) {
        println!("{}", "Proof Information".cyan().bold());
        println!("  Model ID:       {}", p.model_id);
        println!("  Round ID:       {}", p.round_id);
        println!("  Proof Hash:     {}...", &p.proof_hash[..20]);
        println!("  Prover:         {}", p.prover);
        println!("  Verified:       {} {}", if p.verified { "●".green() } else { "●".yellow() }, if p.verified { "Yes" } else { "Pending" });
        println!("  Size:           {} bytes", p.size_bytes);
        if let Some(gas) = p.gas_used {
            println!("  Gas Used:       {}", gas);
        }
        println!("\n{}", "Error Bounds:".yellow());
        println!("  Forward:        {:.2}", p.error_bounds.forward);
        println!("  Backward:       {:.2}", p.error_bounds.backward);
        println!("  Gradient:       {:.2}", p.error_bounds.gradient);
        println!("  Total:          {:.2} / {:.0} max", p.error_bounds.total, p.error_bounds.max_allowed);
    }

    fn display_error(&self, e: &ErrorQueryResult) {
        let status_color = match e.status {
            ErrorStatus::Acceptable => "●".green(),
            ErrorStatus::Warning => "●".yellow(),
            ErrorStatus::Critical => "●".red(),
            ErrorStatus::Exceeded => "●".red().bold(),
        };

        println!("{}", "Error Bound Information".cyan().bold());
        println!("  Model ID:       {}", e.model_id);
        println!("  Current Round:  {}", e.current_round);
        println!("  Accumulated:    {:.1}", e.accumulated_error);
        println!("  Max Allowed:    {:.0}", e.max_allowed);
        println!("  Status:         {} {}", status_color, e.status);
        println!("  Budget Left:    {:.1}", e.budget_remaining);
        println!("  Projected:      {:.1}", e.projected_final);

        if !e.history.is_empty() {
            println!("\n{}", "Error History:".yellow());
            for entry in &e.history {
                println!("  Round {}: {:.2} (cum: {:.1})", entry.round_id, entry.error, entry.cumulative);
            }
        }
    }

    fn display_worker(&self, w: &WorkerQueryResult) {
        println!("{}", "Worker Information".cyan().bold());
        println!("  Address:        {}", w.address);
        println!("  ID:             {}", w.id);
        println!("  Status:         {} {}", if w.status == "active" { "●".green() } else { "●".yellow() }, w.status);
        println!("  Total Stake:    {} ETH", w.total_stake);
        println!("  Reputation:     {:.0}%", w.reputation * 100.0);
        println!("  Models:         {:?}", w.models);
        println!("  Rounds:         {}", w.rounds_completed);
        println!("  Proofs OK:      {}", w.proofs_verified);
        println!("  Proofs Failed:  {}", w.proofs_failed);
        println!("  Rewards:        {} ETH", w.total_rewards);
    }

    fn display_aggregator(&self, a: &AggregatorQueryResult) {
        println!("{}", "Aggregator Information".cyan().bold());
        println!("  Address:        {}", a.address);
        println!("  ID:             {}", a.id);
        println!("  Status:         {} {}", if a.status == "active" { "●".green() } else { "●".yellow() }, a.status);
        println!("  Models:         {:?}", a.models);
        println!("  Rounds:         {}", a.rounds_aggregated);
        println!("  Successful:     {}", a.successful);
        println!("  Failed:         {}", a.failed);
        println!("  Avg Time:       {} ms", a.avg_time_ms);
        println!("  Fees:           {} ETH", a.fees_collected);
    }

    fn display_metrics(&self, m: &MetricsQueryResult) {
        println!("{}", "Metrics for Model".cyan().bold());
        println!("  Model ID:       {}", m.model_id);

        println!("\n{}", "Training:".yellow());
        println!("  Progress:       {}/{} ({:.1}%)",
            m.training.rounds_completed,
            m.training.total_rounds,
            m.training.rounds_completed as f64 / m.training.total_rounds as f64 * 100.0
        );
        println!("  Avg Loss:       {:.4} ({})", m.training.avg_loss, m.training.loss_trend);
        println!("  Avg Round:      {} ms", m.training.avg_round_time_ms);

        println!("\n{}", "Proofs:".yellow());
        println!("  Generated:      {}", m.proofs.total_generated);
        println!("  Verified:       {} ({:.1}%)", m.proofs.total_verified, m.proofs.verification_rate);
        println!("  Avg Time:       {} ms", m.proofs.avg_proof_time_ms);
        println!("  Gas Used:       {}", m.proofs.total_gas_used);

        println!("\n{}", "Network:".yellow());
        println!("  Workers:        {}", m.network.active_workers);
        println!("  Aggregators:    {}", m.network.active_aggregators);
        println!("  Latency:        {} ms", m.network.avg_latency_ms);

        println!("\n{}", "Economics:".yellow());
        println!("  Total Stake:    {} ETH", m.economics.total_stake);
        println!("  Rewards:        {} ETH", m.economics.total_rewards);
        println!("  Slashing:       {} events ({} ETH)", m.economics.slashing_events, m.economics.total_slashed);
    }
}

impl Default for QueryCommand {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_model() {
        let cmd = QueryCommand::new();
        let options = QueryOptions {
            query_type: QueryType::Model,
            model_id: 0,
            ..Default::default()
        };

        let result = cmd.execute(options).await.unwrap();
        if let QueryResult::Model(m) = result {
            assert_eq!(m.id, 0);
            assert!(m.is_active);
        } else {
            panic!("Expected Model result");
        }
    }

    #[tokio::test]
    async fn test_query_error() {
        let cmd = QueryCommand::new();
        let options = QueryOptions {
            query_type: QueryType::Error,
            model_id: 0,
            history: true,
            history_count: 5,
            ..Default::default()
        };

        let result = cmd.execute(options).await.unwrap();
        if let QueryResult::Error(e) = result {
            assert!(!e.history.is_empty());
        } else {
            panic!("Expected Error result");
        }
    }
}
