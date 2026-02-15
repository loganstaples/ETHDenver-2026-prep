//! Complete cheater recovery orchestration.
//!
//! This module ties together all the components needed to recover from a cheater
//! detection during MPC training:
//!
//! 1. **MAC failure detection** → `mac_verification::full_mac_check()`
//! 2. **Blame report signing** → `blame_report::create_signed_blame_report()`
//! 3. **On-chain slashing** → submit `reportMACFailure()` via ethers-rs
//! 4. **Share redistribution** → `share_redistribution::redistribute_shares_after_removal()`
//! 5. **Training resumption** → restore trainer state from checkpoint + new shares
//!
//! # Usage
//!
//! The `CheaterRecoveryOrchestrator` is created once per training session and
//! called when a MAC check fails. It returns a `RecoveryOutcome` that the
//! training loop uses to decide whether to continue or abort.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::blame_report::{
    self, SignedBlameReport, WorkerEthIdentity,
    build_mac_failure_message, serialize_evidence, to_eth_signed_message_hash,
};
use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::mac_verification::{
    MACCheckResult, MACFailureReport, MACState, TrainingCheckpoint,
};
use crate::session::transport::MPCTransport;
use crate::share_redistribution::{self, RedistributionResult};
use crate::types::PartyId;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the cheater recovery system.
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// On-chain job ID for this training session.
    pub job_id: u64,
    /// Session identifier.
    pub session_id: String,
    /// This worker's Ethereum signing identity.
    pub eth_identity: WorkerEthIdentity,
    /// Mapping from MPC party index to Ethereum address.
    /// Used to look up the cheater's on-chain address.
    pub party_addresses: HashMap<usize, [u8; 20]>,
    /// Minimum number of parties to continue training after removal.
    pub min_parties: usize,
    /// Random seed for share redistribution.
    pub redistribution_seed: u64,
}

// ============================================================================
// Recovery outcome
// ============================================================================

/// Outcome of the cheater recovery process.
pub enum RecoveryOutcome {
    /// Recovery succeeded: training can continue from the checkpoint.
    Recovered {
        /// The identified cheater's party index.
        cheater_index: usize,
        /// The signed blame report (may have been submitted on-chain).
        blame_report: SignedBlameReport,
        /// New weight shares for this party.
        new_shares: RedistributionResult,
        /// Step to resume training from.
        resume_from_step: u64,
    },
    /// Recovery failed: not enough parties to continue.
    InsufficientParties {
        remaining: usize,
        required: usize,
    },
    /// Recovery failed: could not identify the cheater.
    CheaterUnidentified {
        /// The MAC failure report without a confirmed cheater.
        failure_report: MACFailureReport,
    },
    /// Recovery failed: cheater's Ethereum address not known.
    UnknownCheaterAddress {
        cheater_index: usize,
    },
}

// ============================================================================
// Orchestrator
// ============================================================================

/// Orchestrates the complete cheater recovery flow.
///
/// Holds the configuration and state needed to perform recovery when a MAC
/// check fails during training.
pub struct CheaterRecoveryOrchestrator {
    config: RecoveryConfig,
}

impl CheaterRecoveryOrchestrator {
    /// Creates a new recovery orchestrator.
    pub fn new(config: RecoveryConfig) -> Self {
        Self { config }
    }

    /// Returns the on-chain job ID.
    pub fn job_id(&self) -> u64 {
        self.config.job_id
    }

    /// Runs the complete recovery flow after a MAC failure is detected.
    ///
    /// # Steps
    ///
    /// 1. Validate the failure report has an identified cheater.
    /// 2. Look up the cheater's Ethereum address.
    /// 3. Create and sign a blame report with all honest workers.
    /// 4. (On-chain submission is handled by the caller, since it requires
    ///    an ethers provider that we don't want to mandate at this layer.)
    /// 5. Redistribute shares among remaining honest parties.
    /// 6. Return the recovery outcome.
    ///
    /// # Arguments
    ///
    /// * `failure_report` - The MAC failure report from the sigma protocol.
    /// * `checkpoint` - The last verified checkpoint to roll back to.
    /// * `transport` - MPC transport for communication with honest peers.
    /// * `num_parties` - Total number of parties (including cheater).
    pub async fn recover<T: MPCTransport>(
        &self,
        failure_report: &MACFailureReport,
        checkpoint: &TrainingCheckpoint,
        transport: &T,
        num_parties: usize,
    ) -> MPCResult<RecoveryOutcome> {
        let party_index = self.config.eth_identity.party_index;

        // Step 1: Check that cheater was identified.
        let cheater_index = match failure_report.identified_cheater {
            Some(idx) => idx,
            None => {
                warn!("MAC failure detected but cheater could not be identified");
                return Ok(RecoveryOutcome::CheaterUnidentified {
                    failure_report: failure_report.clone(),
                });
            }
        };

        info!(
            party = party_index,
            cheater = cheater_index,
            step = failure_report.step_number,
            "Starting cheater recovery"
        );

        // Step 2: Look up cheater's Ethereum address.
        let cheater_address = match self.config.party_addresses.get(&cheater_index) {
            Some(addr) => *addr,
            None => {
                error!(
                    cheater = cheater_index,
                    "Cheater's Ethereum address not found in party_addresses"
                );
                return Ok(RecoveryOutcome::UnknownCheaterAddress { cheater_index });
            }
        };

        // Step 3: Check if enough parties remain.
        let remaining = num_parties - 1;
        if remaining < self.config.min_parties {
            return Ok(RecoveryOutcome::InsufficientParties {
                remaining,
                required: self.config.min_parties,
            });
        }

        // Step 4: Create signed blame report.
        info!(party = party_index, "Creating signed blame report");
        let blame_report = blame_report::create_signed_blame_report(
            failure_report,
            self.config.job_id,
            &cheater_address,
            &self.config.eth_identity,
            transport,
            num_parties,
            cheater_index,
        )
        .await?;

        info!(
            party = party_index,
            signatures = blame_report.signatures.len(),
            "Blame report signed by {} honest workers",
            blame_report.signatures.len()
        );

        // Step 5: Redistribute shares (this is the heavy lifting).
        info!(party = party_index, "Starting share redistribution");
        let new_shares = share_redistribution::redistribute_shares_after_removal(
            checkpoint,
            transport,
            party_index,
            num_parties,
            cheater_index,
            &self.config.session_id,
            self.config.redistribution_seed,
        )
        .await?;

        info!(
            party = party_index,
            resume_step = checkpoint.step,
            "Recovery complete — ready to resume training"
        );

        Ok(RecoveryOutcome::Recovered {
            cheater_index,
            blame_report,
            new_shares,
            resume_from_step: checkpoint.step,
        })
    }
}

// ============================================================================
// On-chain submission helpers (requires ethers feature)
// ============================================================================

/// Encodes the blame report data for on-chain `reportMACFailure()` call.
///
/// Returns the ABI-encoded call data that can be submitted to the
/// `HelixCoordinatorV4` contract.
///
/// This function works without the `ethers` feature — it produces raw call data
/// that any Ethereum client library can submit.
pub fn encode_report_mac_failure_calldata(report: &SignedBlameReport) -> Vec<u8> {
    // Function selector: keccak256("reportMACFailure(uint256,uint256,address,bytes,bytes[])")
    let selector = blame_report::keccak256(
        b"reportMACFailure(uint256,uint256,address,bytes,bytes[])"
    );

    // For a complete ABI encoding we need:
    // - selector (4 bytes)
    // - jobId (uint256, 32 bytes)
    // - stepNumber (uint256, 32 bytes)
    // - cheater (address, 32 bytes, left-padded)
    // - offset to evidence (uint256, 32 bytes)
    // - offset to signatures (uint256, 32 bytes)
    // - evidence length + data
    // - signatures array length + offsets + data

    let mut data = Vec::new();

    // Selector (first 4 bytes of the hash)
    data.extend_from_slice(&selector[..4]);

    // jobId (uint256)
    let mut job_id_bytes = [0u8; 32];
    job_id_bytes[24..].copy_from_slice(&report.job_id.to_be_bytes());
    data.extend_from_slice(&job_id_bytes);

    // stepNumber (uint256)
    let mut step_bytes = [0u8; 32];
    step_bytes[24..].copy_from_slice(&report.step_number.to_be_bytes());
    data.extend_from_slice(&step_bytes);

    // cheater (address, left-padded to 32 bytes)
    let mut cheater_bytes = [0u8; 32];
    cheater_bytes[12..].copy_from_slice(&report.cheater_address);
    data.extend_from_slice(&cheater_bytes);

    // Offset to evidence bytes (dynamic type)
    // Head part is 5 * 32 = 160 bytes (selector not counted in ABI offset)
    let evidence_offset: u64 = 5 * 32; // after 5 head slots
    let mut offset_bytes = [0u8; 32];
    offset_bytes[24..].copy_from_slice(&evidence_offset.to_be_bytes());
    data.extend_from_slice(&offset_bytes);

    // Offset to signatures array (dynamic type)
    // evidence_offset + 32 (length) + ceil(evidence.len() / 32) * 32
    let evidence_padded_len = ((report.evidence.len() + 31) / 32) * 32;
    let sigs_offset = evidence_offset + 32 + evidence_padded_len as u64;
    let mut sigs_offset_bytes = [0u8; 32];
    sigs_offset_bytes[24..].copy_from_slice(&sigs_offset.to_be_bytes());
    data.extend_from_slice(&sigs_offset_bytes);

    // Evidence bytes (length + data, padded to 32-byte boundary)
    let mut evidence_len_bytes = [0u8; 32];
    evidence_len_bytes[24..].copy_from_slice(&(report.evidence.len() as u64).to_be_bytes());
    data.extend_from_slice(&evidence_len_bytes);
    data.extend_from_slice(&report.evidence);
    // Pad to 32-byte boundary
    let padding = evidence_padded_len - report.evidence.len();
    data.extend(std::iter::repeat(0u8).take(padding));

    // Signatures array (length + offsets + data)
    let num_sigs = report.signatures.len();
    let mut num_sigs_bytes = [0u8; 32];
    num_sigs_bytes[24..].copy_from_slice(&(num_sigs as u64).to_be_bytes());
    data.extend_from_slice(&num_sigs_bytes);

    // Each signature is 65 bytes (bytes type in Solidity = dynamic)
    // Offsets for each signature element
    let sig_data_start = num_sigs as u64 * 32; // past all offset slots
    for i in 0..num_sigs {
        // Each previous sig: 32 (length) + 96 (padded 65 bytes to 96) = 128 bytes
        let sig_offset = sig_data_start + i as u64 * (32 + 96);
        let mut sig_offset_bytes = [0u8; 32];
        sig_offset_bytes[24..].copy_from_slice(&sig_offset.to_be_bytes());
        data.extend_from_slice(&sig_offset_bytes);
    }

    // Signature data
    for sig in &report.signatures {
        // Length (65 bytes)
        let mut sig_len_bytes = [0u8; 32];
        sig_len_bytes[24..].copy_from_slice(&65u64.to_be_bytes());
        data.extend_from_slice(&sig_len_bytes);
        // Data (65 bytes, padded to 96)
        let mut padded_sig = [0u8; 96];
        padded_sig[..65].copy_from_slice(sig);
        data.extend_from_slice(&padded_sig);
    }

    data
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mac_verification::{CheaterEvidence, PairwiseCheckResult};
    use crate::session::transport::LocalTransport;

    fn make_test_config(party_index: usize, party_addresses: HashMap<usize, [u8; 20]>) -> RecoveryConfig {
        let mut pk = [0u8; 32];
        pk[31] = (party_index + 1) as u8;
        let identity = WorkerEthIdentity::from_private_key(party_index, pk).unwrap();

        RecoveryConfig {
            job_id: 1,
            session_id: "test-recovery".to_string(),
            eth_identity: identity,
            party_addresses,
            min_parties: 2,
            redistribution_seed: 42,
        }
    }

    fn make_test_checkpoint(step: u64) -> TrainingCheckpoint {
        TrainingCheckpoint {
            step,
            w1: vec![Fr::from_f64(0.1), Fr::from_f64(0.2)],
            b1: vec![Fr::from_f64(0.01)],
            w2: vec![Fr::from_f64(0.05)],
            b2: vec![Fr::from_f64(0.001)],
            w1_macs: Vec::new(),
            b1_macs: Vec::new(),
            w2_macs: Vec::new(),
            b2_macs: Vec::new(),
            beaver_cursor: 0,
            auth_beaver_cursor: 0,
        }
    }

    fn make_test_report(cheater: usize) -> MACFailureReport {
        MACFailureReport {
            session_id: "test-recovery".to_string(),
            step_number: 30,
            identified_cheater: Some(cheater),
            sigma_values: vec![vec![1]; 3],
            commitments: vec![[0u8; 32]; 3],
            evidence: CheaterEvidence {
                pairwise_results: vec![
                    PairwiseCheckResult { party_a: 0, party_b: 2, consistent: false },
                    PairwiseCheckResult { party_a: 1, party_b: 2, consistent: false },
                ],
                round1_sigmas: vec![vec![1]; 3],
                round2_sigmas: vec![vec![1]; 3],
            },
        }
    }

    #[tokio::test]
    async fn test_recovery_orchestrator_happy_path() {
        let num_parties = 3;
        let cheater_index = 2;

        // Create party addresses.
        let mut party_addresses = HashMap::new();
        for i in 0..num_parties {
            let mut pk = [0u8; 32];
            pk[31] = (i + 1) as u8;
            let addr = blame_report::address_from_private_key(&pk).unwrap();
            party_addresses.insert(i, addr);
        }

        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        let report = make_test_report(cheater_index);
        let checkpoint = make_test_checkpoint(20);

        // Run recovery for all honest parties concurrently.
        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            if i == cheater_index {
                continue; // Cheater doesn't participate
            }

            let config = make_test_config(i, party_addresses.clone());
            let report_clone = report.clone();
            let cp_clone = checkpoint.clone();

            handles.push(tokio::spawn(async move {
                let orchestrator = CheaterRecoveryOrchestrator::new(config);
                orchestrator.recover(
                    &report_clone,
                    &cp_clone,
                    &transport,
                    num_parties,
                ).await
            }));
        }

        // Collect results.
        for handle in handles {
            let outcome = handle.await.unwrap().expect("recovery should succeed");
            match outcome {
                RecoveryOutcome::Recovered {
                    cheater_index: ci,
                    blame_report,
                    new_shares,
                    resume_from_step,
                } => {
                    assert_eq!(ci, cheater_index);
                    assert_eq!(resume_from_step, 20);
                    assert!(blame_report.signatures.len() >= 2, "should have at least 2 signatures");
                    assert!(new_shares.w1.len() > 0, "should have new weight shares");
                }
                _ => panic!("expected Recovered variant"),
            }
        }
    }

    #[tokio::test]
    async fn test_recovery_unidentified_cheater() {
        let num_parties = 3;
        let mut party_addresses = HashMap::new();
        for i in 0..num_parties {
            let mut pk = [0u8; 32];
            pk[31] = (i + 1) as u8;
            let addr = blame_report::address_from_private_key(&pk).unwrap();
            party_addresses.insert(i, addr);
        }

        let config = make_test_config(0, party_addresses);
        let orchestrator = CheaterRecoveryOrchestrator::new(config);

        // Report without identified cheater.
        let mut report = make_test_report(2);
        report.identified_cheater = None;

        let checkpoint = make_test_checkpoint(20);
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        let outcome = orchestrator
            .recover(&report, &checkpoint, &transports[0], num_parties)
            .await
            .unwrap();

        match outcome {
            RecoveryOutcome::CheaterUnidentified { .. } => {}
            _ => panic!("expected CheaterUnidentified variant"),
        }
    }

    #[tokio::test]
    async fn test_recovery_unknown_address() {
        let config = make_test_config(0, HashMap::new()); // Empty addresses
        let orchestrator = CheaterRecoveryOrchestrator::new(config);

        let report = make_test_report(2);
        let checkpoint = make_test_checkpoint(20);
        let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        let outcome = orchestrator
            .recover(&report, &checkpoint, &transports[0], 3)
            .await
            .unwrap();

        match outcome {
            RecoveryOutcome::UnknownCheaterAddress { cheater_index } => {
                assert_eq!(cheater_index, 2);
            }
            _ => panic!("expected UnknownCheaterAddress variant"),
        }
    }

    #[test]
    fn test_encode_calldata() {
        let report = SignedBlameReport {
            job_id: 1,
            step_number: 30,
            cheater_address: [0xAB; 20],
            evidence: vec![1, 2, 3, 4],
            message_hash: [0u8; 32],
            eth_signed_hash: [0u8; 32],
            signatures: vec![[0xCC; 65], [0xDD; 65]],
            signer_addresses: vec![[0x11; 20], [0x22; 20]],
        };

        let calldata = encode_report_mac_failure_calldata(&report);
        // Should start with the function selector (4 bytes).
        assert!(calldata.len() > 4);
        // Selector should be first 4 bytes of keccak256 of the function signature.
        let expected_selector = blame_report::keccak256(
            b"reportMACFailure(uint256,uint256,address,bytes,bytes[])"
        );
        assert_eq!(&calldata[..4], &expected_selector[..4]);
    }

    #[tokio::test]
    async fn test_recovery_insufficient_parties() {
        let num_parties = 2; // Only 2 parties, removing one leaves 1
        let mut party_addresses = HashMap::new();
        for i in 0..num_parties {
            let mut pk = [0u8; 32];
            pk[31] = (i + 1) as u8;
            let addr = blame_report::address_from_private_key(&pk).unwrap();
            party_addresses.insert(i, addr);
        }

        let config = RecoveryConfig {
            job_id: 1,
            session_id: "test".to_string(),
            eth_identity: WorkerEthIdentity::from_private_key(0, {
                let mut pk = [0u8; 32]; pk[31] = 1; pk
            }).unwrap(),
            party_addresses,
            min_parties: 2,
            redistribution_seed: 42,
        };

        let orchestrator = CheaterRecoveryOrchestrator::new(config);
        let report = MACFailureReport {
            session_id: "test".to_string(),
            step_number: 10,
            identified_cheater: Some(1),
            sigma_values: vec![],
            commitments: vec![],
            evidence: CheaterEvidence {
                pairwise_results: vec![],
                round1_sigmas: vec![],
                round2_sigmas: vec![],
            },
        };

        let checkpoint = make_test_checkpoint(5);
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        let outcome = orchestrator
            .recover(&report, &checkpoint, &transports[0], num_parties)
            .await
            .unwrap();

        match outcome {
            RecoveryOutcome::InsufficientParties { remaining, required } => {
                assert_eq!(remaining, 1);
                assert_eq!(required, 2);
            }
            _ => panic!("expected InsufficientParties variant"),
        }
    }
}
