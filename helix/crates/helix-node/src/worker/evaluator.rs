//! Round Evaluation Logic.
//!
//! Decides whether a worker should join an announced training round based on:
//! - Model architecture compatibility (can the worker handle these dimensions?)
//! - Available capacity (is the worker already in a round?)
//! - Error budget (is the error budget reasonable?)
//! - Deadline feasibility (can the worker finish before the deadline?)
//! - Minimum reward threshold (is it worth participating?)

use crate::network::messages::{ModelDims, NodeCapabilities};

/// Configuration for round evaluation decisions.
#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    /// Maximum model parameters the worker can handle (based on memory).
    pub max_model_params: usize,
    /// Maximum error budget the worker is willing to accept.
    pub max_error_budget: f64,
    /// Minimum seconds of slack before deadline to consider joining.
    pub min_deadline_slack_secs: u64,
    /// Maximum number of steps per worker the worker is willing to do.
    pub max_steps_per_round: u32,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            max_model_params: 1_000_000,
            max_error_budget: 1.0,
            min_deadline_slack_secs: 30,
            max_steps_per_round: 1000,
        }
    }
}

/// Reason a round was rejected during evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// Worker is already participating in a round.
    AlreadyInRound(u64),
    /// Model is too large for this worker's capabilities.
    ModelTooLarge {
        model_params: usize,
        max_params: usize,
    },
    /// Error budget exceeds the worker's tolerance.
    ErrorBudgetTooHigh {
        budget: String,
        max: String,
    },
    /// Deadline is too tight to complete the required steps.
    DeadlineTooTight {
        remaining_secs: u64,
        min_slack: u64,
    },
    /// Too many steps required.
    TooManySteps {
        required: u32,
        max: u32,
    },
    /// Worker capabilities insufficient (can't train or can't prove).
    InsufficientCapabilities,
    /// Auto-join is disabled and this was not a manual join.
    AutoJoinDisabled,
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyInRound(id) => write!(f, "already in round {}", id),
            Self::ModelTooLarge { model_params, max_params } => {
                write!(f, "model too large: {} params (max {})", model_params, max_params)
            }
            Self::ErrorBudgetTooHigh { budget, max } => {
                write!(f, "error budget too high: {} (max {})", budget, max)
            }
            Self::DeadlineTooTight { remaining_secs, min_slack } => {
                write!(f, "deadline too tight: {}s remaining (need {}s)", remaining_secs, min_slack)
            }
            Self::TooManySteps { required, max } => {
                write!(f, "too many steps: {} (max {})", required, max)
            }
            Self::InsufficientCapabilities => write!(f, "insufficient capabilities"),
            Self::AutoJoinDisabled => write!(f, "auto-join disabled"),
        }
    }
}

/// Evaluates training round opportunities for a worker.
pub struct RoundEvaluator {
    config: EvaluatorConfig,
    capabilities: NodeCapabilities,
}

impl RoundEvaluator {
    /// Creates a new evaluator with the given config and capabilities.
    pub fn new(config: EvaluatorConfig, capabilities: NodeCapabilities) -> Self {
        Self { config, capabilities }
    }

    /// Evaluates whether the worker should join a round.
    ///
    /// Returns `Ok(())` if the round is acceptable, or `Err(RejectReason)` explaining
    /// why the worker should not join.
    pub fn evaluate(
        &self,
        model_dims: &ModelDims,
        steps_per_worker: u32,
        error_budget: f64,
        deadline: u64,
        current_round: Option<u64>,
        auto_join: bool,
    ) -> Result<(), RejectReason> {
        // 1. Check auto-join mode
        if !auto_join {
            return Err(RejectReason::AutoJoinDisabled);
        }

        // 2. Check if already in a round
        if let Some(active_round_id) = current_round {
            return Err(RejectReason::AlreadyInRound(active_round_id));
        }

        // 3. Check capabilities
        if !self.capabilities.can_train || !self.capabilities.can_prove {
            return Err(RejectReason::InsufficientCapabilities);
        }

        // 4. Check model size
        let model_params = estimate_model_params(model_dims);
        if model_params > self.config.max_model_params {
            return Err(RejectReason::ModelTooLarge {
                model_params,
                max_params: self.config.max_model_params,
            });
        }

        // 5. Check error budget
        if error_budget > self.config.max_error_budget {
            return Err(RejectReason::ErrorBudgetTooHigh {
                budget: format!("{:.6}", error_budget),
                max: format!("{:.6}", self.config.max_error_budget),
            });
        }

        // 6. Check deadline
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if deadline > 0 && deadline > now {
            let remaining = deadline - now;
            if remaining < self.config.min_deadline_slack_secs {
                return Err(RejectReason::DeadlineTooTight {
                    remaining_secs: remaining,
                    min_slack: self.config.min_deadline_slack_secs,
                });
            }
        }

        // 7. Check step count
        if steps_per_worker > self.config.max_steps_per_round {
            return Err(RejectReason::TooManySteps {
                required: steps_per_worker,
                max: self.config.max_steps_per_round,
            });
        }

        Ok(())
    }

    /// Returns a reference to the evaluator's configuration.
    pub fn config(&self) -> &EvaluatorConfig {
        &self.config
    }
}

/// Estimates the number of trainable parameters from model dimensions.
///
/// For a 2-layer MLP: params = d_in*d_hid + d_hid + d_hid*d_out + d_out
fn estimate_model_params(dims: &ModelDims) -> usize {
    let layers = dims.num_layers.max(1) as usize;
    if layers == 1 {
        // Single hidden layer: W1[d_hid×d_in] + b1[d_hid] + W2[d_out×d_hid] + b2[d_out]
        dims.d_in * dims.d_hid + dims.d_hid + dims.d_hid * dims.d_out + dims.d_out
    } else {
        // Multi-layer: first layer + (layers-2) hidden-to-hidden + final layer
        let first = dims.d_in * dims.d_hid + dims.d_hid;
        let middle = (layers.saturating_sub(2)) * (dims.d_hid * dims.d_hid + dims.d_hid);
        let last = dims.d_hid * dims.d_out + dims.d_out;
        first + middle + last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_capabilities() -> NodeCapabilities {
        NodeCapabilities {
            can_train: true,
            can_aggregate: false,
            can_prove: true,
            gpu_memory_mb: 4096,
            cpu_cores: 8,
            storage_gb: 100,
        }
    }

    fn test_dims() -> ModelDims {
        ModelDims {
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            num_layers: 2,
            num_heads: 0,
            activation_type: 0,
        }
    }

    #[test]
    fn test_evaluate_accepts_valid_round() {
        let evaluator = RoundEvaluator::new(
            EvaluatorConfig::default(),
            test_capabilities(),
        );
        let dims = test_dims();
        let deadline = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() + 600;

        let result = evaluator.evaluate(&dims, 10, 0.1, deadline, None, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_evaluate_rejects_auto_join_disabled() {
        let evaluator = RoundEvaluator::new(
            EvaluatorConfig::default(),
            test_capabilities(),
        );
        let dims = test_dims();
        let result = evaluator.evaluate(&dims, 10, 0.1, 0, None, false);
        assert_eq!(result.unwrap_err(), RejectReason::AutoJoinDisabled);
    }

    #[test]
    fn test_evaluate_rejects_already_in_round() {
        let evaluator = RoundEvaluator::new(
            EvaluatorConfig::default(),
            test_capabilities(),
        );
        let dims = test_dims();
        let result = evaluator.evaluate(&dims, 10, 0.1, 0, Some(5), true);
        assert_eq!(result.unwrap_err(), RejectReason::AlreadyInRound(5));
    }

    #[test]
    fn test_evaluate_rejects_model_too_large() {
        let evaluator = RoundEvaluator::new(
            EvaluatorConfig {
                max_model_params: 10,
                ..Default::default()
            },
            test_capabilities(),
        );
        let dims = test_dims();
        let result = evaluator.evaluate(&dims, 10, 0.1, 0, None, true);
        assert!(matches!(result.unwrap_err(), RejectReason::ModelTooLarge { .. }));
    }

    #[test]
    fn test_evaluate_rejects_error_budget_too_high() {
        let evaluator = RoundEvaluator::new(
            EvaluatorConfig {
                max_error_budget: 0.01,
                ..Default::default()
            },
            test_capabilities(),
        );
        let dims = test_dims();
        let result = evaluator.evaluate(&dims, 10, 0.5, 0, None, true);
        assert!(matches!(result.unwrap_err(), RejectReason::ErrorBudgetTooHigh { .. }));
    }

    #[test]
    fn test_evaluate_rejects_insufficient_capabilities() {
        let caps = NodeCapabilities {
            can_train: false,
            can_prove: true,
            ..Default::default()
        };
        let evaluator = RoundEvaluator::new(EvaluatorConfig::default(), caps);
        let dims = test_dims();
        let result = evaluator.evaluate(&dims, 10, 0.1, 0, None, true);
        assert_eq!(result.unwrap_err(), RejectReason::InsufficientCapabilities);
    }

    #[test]
    fn test_evaluate_rejects_too_many_steps() {
        let evaluator = RoundEvaluator::new(
            EvaluatorConfig {
                max_steps_per_round: 5,
                ..Default::default()
            },
            test_capabilities(),
        );
        let dims = test_dims();
        let result = evaluator.evaluate(&dims, 100, 0.1, 0, None, true);
        assert!(matches!(result.unwrap_err(), RejectReason::TooManySteps { .. }));
    }

    #[test]
    fn test_evaluate_rejects_tight_deadline() {
        let evaluator = RoundEvaluator::new(
            EvaluatorConfig {
                min_deadline_slack_secs: 600,
                ..Default::default()
            },
            test_capabilities(),
        );
        let dims = test_dims();
        let deadline = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() + 10; // only 10 seconds

        let result = evaluator.evaluate(&dims, 10, 0.1, deadline, None, true);
        assert!(matches!(result.unwrap_err(), RejectReason::DeadlineTooTight { .. }));
    }

    #[test]
    fn test_estimate_model_params_single_layer() {
        let dims = ModelDims {
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            num_layers: 1,
            num_heads: 0,
            activation_type: 0,
        };
        // W1: 4*8=32, b1: 8, W2: 8*2=16, b2: 2 = 58
        assert_eq!(estimate_model_params(&dims), 58);
    }

    #[test]
    fn test_estimate_model_params_multi_layer() {
        let dims = ModelDims {
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            num_layers: 3,
            num_heads: 0,
            activation_type: 0,
        };
        // first: 4*8+8=40, middle(1): 8*8+8=72, last: 8*2+2=18 = 130
        assert_eq!(estimate_model_params(&dims), 130);
    }

    #[test]
    fn test_reject_reason_display() {
        let reason = RejectReason::AlreadyInRound(5);
        assert_eq!(reason.to_string(), "already in round 5");

        let reason = RejectReason::AutoJoinDisabled;
        assert_eq!(reason.to_string(), "auto-join disabled");
    }
}
