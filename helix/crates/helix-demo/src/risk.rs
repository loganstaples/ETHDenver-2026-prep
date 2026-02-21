//! Risk assessment for automatic ZK proof activation.
//!
//! Evaluates conditions from an MPC training result to determine whether
//! ZK proofs should be generated. Once risk is triggered, it stays active
//! (matching on-chain `zkActivatedByRisk` which is irreversible).

use helix_mpc::e2e_integration::MPCIntegrationResult;

/// Evaluates risk conditions and determines which checkpoints need ZK proofs.
pub struct RiskAssessor {
    min_workers: usize,
    /// Step at which risk was first detected (None = no risk).
    risk_triggered_at: Option<usize>,
}

impl RiskAssessor {
    pub fn new(min_workers_for_mpc: usize) -> Self {
        Self {
            min_workers: min_workers_for_mpc,
            risk_triggered_at: None,
        }
    }

    /// Evaluate the training result and determine the risk trigger point.
    ///
    /// Returns the step number at which risk was first detected, or None if
    /// training completed without risk conditions.
    pub fn evaluate(&mut self, result: &MPCIntegrationResult) -> Option<usize> {
        // Condition 1: Cheater detected during training
        if let Some(ref cheater) = result.cheater_detected {
            let step = cheater.detected_at_step as usize;
            self.risk_triggered_at = Some(step);
            return Some(step);
        }

        // Condition 2: Recovery occurred (implies worker loss)
        if result.recovery_completed {
            // Risk from step 0 — recovery means the network was compromised
            self.risk_triggered_at = Some(0);
            return Some(0);
        }

        None
    }

    /// Whether a given checkpoint step requires a ZK proof.
    pub fn needs_proof(&self, checkpoint_step: usize) -> bool {
        match self.risk_triggered_at {
            Some(trigger_step) => checkpoint_step >= trigger_step,
            None => false,
        }
    }

    /// Whether risk was triggered at all.
    pub fn is_triggered(&self) -> bool {
        self.risk_triggered_at.is_some()
    }

    /// The step at which risk was triggered.
    pub fn trigger_step(&self) -> Option<usize> {
        self.risk_triggered_at
    }
}
