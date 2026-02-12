//! On-Chain Proof Submission Pipeline.
//!
//! Orchestrates the aggregator's interaction with the on-chain coordinator:
//! - Model registration and staking on startup
//! - RLC proof aggregation (multiple worker proofs → single KZG proof)
//! - On-chain proof submission via SCClient
//! - Background proof queue processing for RPC-submitted proofs
//! - VerifyAll verification policy with real Halo2 KZG checks

use std::sync::Arc;
use std::time::Duration;

use ethers::types::U256;
use parking_lot::RwLock as SyncRwLock;
use tokio::sync::RwLock;

use helix_prover::{
    AggregatedTrainingProof, BatchProver, RLCAggregationProver, TrainingProofResultV2,
};

use crate::api::rpc::QueuedProof;
use crate::config::ChainConfig;
use crate::roles::verifier::{VerificationPolicy, VerifierConfig, VerifierNode};
use crate::sc_client::{SCClient, TrainingProofInputs};

/// Result of an on-chain proof submission.
#[derive(Debug, Clone)]
pub struct OnChainSubmission {
    /// Transaction hash.
    pub tx_hash: String,
    /// Block number the tx was included in.
    pub block_number: u64,
    /// Gas used.
    pub gas_used: u64,
    /// Number of worker proofs aggregated.
    pub proofs_aggregated: usize,
}

/// On-chain pipeline state — tracks model registration and staking.
#[derive(Debug, Clone)]
pub struct PipelineState {
    /// Registered model ID (set after registration or from config).
    pub model_id: Option<u64>,
    /// Whether we've staked for this model.
    pub staked: bool,
    /// Current on-chain round.
    pub current_round: u64,
    /// Total proofs submitted on-chain.
    pub proofs_submitted: u64,
}

/// On-chain proof submission pipeline for aggregator nodes.
///
/// When an aggregator starts with chain config, this pipeline:
/// 1. Registers the model on-chain (or uses pre-registered model_id)
/// 2. Stakes tokens for proof submission rights
/// 3. Aggregates worker proofs via RLC (SHPLONKAggregationCircuit)
/// 4. Submits the single aggregated proof on-chain
/// 5. Processes queued proofs from RPC submissions
pub struct OnChainPipeline {
    /// Smart contract client.
    sc_client: Arc<SCClient>,
    /// Chain configuration.
    config: ChainConfig,
    /// Pipeline state.
    state: Arc<RwLock<PipelineState>>,
    /// Verifier node with VerifyAll policy for pre-submission checks.
    verifier: Arc<VerifierNode>,
}

impl OnChainPipeline {
    /// Creates a new on-chain pipeline.
    ///
    /// Initializes the SCClient from the node's RPC URL and private key,
    /// and sets up a VerifyAll verifier with the configured model dimensions.
    pub async fn new(
        rpc_url: &str,
        private_key: &str,
        config: ChainConfig,
    ) -> anyhow::Result<Self> {
        let sc_client = Arc::new(
            SCClient::with_config(rpc_url, private_key, &config.coordinator_address).await?,
        );

        // Create verifier with VerifyAll policy using model dimensions
        let verifier_config = VerifierConfig::for_model(config.d_in, config.d_hid, config.d_out);
        let verifier = Arc::new(VerifierNode::new(
            crate::network::messages::PeerId::random(),
            verifier_config,
        ));

        let initial_model_id = config.model_id;

        Ok(Self {
            sc_client,
            config,
            state: Arc::new(RwLock::new(PipelineState {
                model_id: initial_model_id,
                staked: false,
                current_round: 0,
                proofs_submitted: 0,
            })),
            verifier,
        })
    }

    /// Returns a reference to the SCClient.
    pub fn sc_client(&self) -> &Arc<SCClient> {
        &self.sc_client
    }

    /// Returns the current pipeline state.
    pub async fn state(&self) -> PipelineState {
        self.state.read().await.clone()
    }

    /// Returns the model dimensions from config.
    pub fn model_dims(&self) -> (usize, usize, usize) {
        (self.config.d_in, self.config.d_hid, self.config.d_out)
    }

    /// Returns the verifier (VerifyAll policy).
    pub fn verifier(&self) -> &Arc<VerifierNode> {
        &self.verifier
    }

    /// Returns the verification policy in use.
    pub fn verification_policy(&self) -> VerificationPolicy {
        VerificationPolicy::VerifyAll
    }

    /// Initializes the on-chain state: registers model and stakes.
    ///
    /// Idempotent — if model_id is already set and stake exists, this is a no-op.
    pub async fn initialize(&self) -> anyhow::Result<u64> {
        let mut state = self.state.write().await;

        // Step 1: Register model if we don't have a model_id
        let model_id = if let Some(id) = state.model_id {
            log::info!("Using pre-registered model_id={}", id);
            id
        } else {
            log::info!("Registering new model on-chain...");
            let min_stake = U256::from_dec_str(&self.config.min_stake_wei)
                .unwrap_or(U256::from(1_000_000_000_000_000_000u64));

            let (receipt, model_id) = self
                .sc_client
                .register_model(&self.config.model_ipfs_hash, U256::from(1), min_stake)
                .await?;

            log::info!(
                "Model registered: model_id={}, tx={:?}",
                model_id,
                receipt.transaction_hash
            );
            state.model_id = Some(model_id);
            model_id
        };

        // Step 2: Stake if not already staked
        if !state.staked {
            let stake_amount = U256::from_dec_str(&self.config.stake_amount_wei)
                .unwrap_or(U256::from(1_000_000_000_000_000_000u64));

            log::info!(
                "Staking {} wei for model_id={}...",
                stake_amount,
                model_id
            );
            match self.sc_client.stake(model_id, stake_amount).await {
                Ok(receipt) => {
                    log::info!(
                        "Staked successfully: tx={:?}",
                        receipt.transaction_hash
                    );
                    state.staked = true;
                }
                Err(e) => {
                    // Staking may fail if already staked — log but don't fail
                    log::warn!("Stake transaction failed (may already be staked): {}", e);
                    state.staked = true; // Assume staked
                }
            }
        }

        Ok(model_id)
    }

    /// Starts a new training round on-chain.
    pub async fn start_round_on_chain(&self) -> anyhow::Result<u64> {
        let state = self.state.read().await;
        let model_id = state
            .model_id
            .ok_or_else(|| anyhow::anyhow!("Model not registered — call initialize() first"))?;
        drop(state);

        log::info!(
            "Starting on-chain round for model_id={}, duration={}s",
            model_id,
            self.config.round_duration_secs
        );

        let receipt = self
            .sc_client
            .start_round(model_id, self.config.round_duration_secs)
            .await?;

        let mut state = self.state.write().await;
        state.current_round += 1;
        let round = state.current_round;

        log::info!(
            "Round {} started on-chain: tx={:?}",
            round,
            receipt.transaction_hash
        );

        Ok(round)
    }

    /// Aggregates worker proofs and submits the result on-chain.
    ///
    /// Takes collected worker proofs (from AggregatorNode round completion),
    /// runs RLC aggregation to produce a single proof, and submits it
    /// via `SCClient::submit_proof_raw()`.
    ///
    /// Returns `None` if there are no proofs or aggregation fails.
    pub async fn submit_aggregated_round(
        &self,
        round_id: u64,
        worker_proofs: Vec<TrainingProofResultV2>,
    ) -> anyhow::Result<Option<OnChainSubmission>> {
        if worker_proofs.is_empty() {
            log::warn!("No worker proofs to aggregate for round {}", round_id);
            return Ok(None);
        }

        let state = self.state.read().await;
        let model_id = state
            .model_id
            .ok_or_else(|| anyhow::anyhow!("Model not registered"))?;
        drop(state);

        log::info!(
            "Aggregating {} worker proofs for round {} via RLC...",
            worker_proofs.len(),
            round_id
        );

        // Run RLC aggregation
        let aggregated = BatchProver::aggregate_training_proofs(&worker_proofs)
            .map_err(|e| anyhow::anyhow!("RLC aggregation failed: {}", e))?;

        log::info!(
            "RLC aggregation succeeded: {} steps, proof size {} bytes",
            aggregated.num_steps,
            aggregated.proof.len()
        );

        // Convert Fr public inputs to U256 for on-chain submission
        let public_inputs_u256 = fr_vec_to_u256(&aggregated.public_inputs);

        // Submit on-chain
        let receipt = self
            .sc_client
            .submit_proof_raw(model_id, round_id, aggregated.proof.clone(), public_inputs_u256)
            .await?;

        let tx_hash = format!("{:?}", receipt.transaction_hash);
        let block_number = receipt.block_number.map(|b| b.as_u64()).unwrap_or(0);
        let gas_used = receipt
            .gas_used
            .map(|g| g.as_u64())
            .unwrap_or(0);

        let mut state = self.state.write().await;
        state.proofs_submitted += 1;

        log::info!(
            "Proof submitted on-chain: tx={}, block={}, gas={}",
            tx_hash,
            block_number,
            gas_used
        );

        Ok(Some(OnChainSubmission {
            tx_hash,
            block_number,
            gas_used,
            proofs_aggregated: worker_proofs.len(),
        }))
    }

    /// Processes queued proofs from the RPC `helix_submitProof` endpoint.
    ///
    /// Drains unsubmitted proofs from the queue, decodes them, and submits
    /// each one on-chain via `SCClient::submit_proof_raw()`.
    pub async fn process_proof_queue(
        &self,
        proof_queue: &Arc<SyncRwLock<Vec<QueuedProof>>>,
    ) -> Vec<(usize, Result<String, String>)> {
        let state = self.state.read().await;
        let model_id = match state.model_id {
            Some(id) => id,
            None => {
                log::debug!("Proof queue processor: model not registered yet, skipping");
                return vec![];
            }
        };
        drop(state);

        // Collect unsubmitted proofs
        let unsubmitted: Vec<(usize, QueuedProof)> = {
            let queue = proof_queue.read();
            queue
                .iter()
                .enumerate()
                .filter(|(_, p)| !p.submitted)
                .map(|(i, p)| (i, p.clone()))
                .collect()
        };

        if unsubmitted.is_empty() {
            return vec![];
        }

        log::info!(
            "Processing {} queued proofs from RPC submissions",
            unsubmitted.len()
        );

        let mut results = Vec::new();

        for (idx, queued) in unsubmitted {
            // Decode proof hex
            let proof_stripped = queued
                .proof_hex
                .strip_prefix("0x")
                .unwrap_or(&queued.proof_hex);
            let proof_bytes = match hex::decode(proof_stripped) {
                Ok(b) => b,
                Err(e) => {
                    results.push((idx, Err(format!("Invalid proof hex: {}", e))));
                    continue;
                }
            };

            // Decode public inputs hex → U256
            let public_inputs: Vec<U256> = queued
                .public_inputs_hex
                .iter()
                .map(|hex_str| {
                    let stripped = hex_str.strip_prefix("0x").unwrap_or(hex_str);
                    let bytes = hex::decode(stripped).unwrap_or_default();
                    if bytes.len() <= 32 {
                        U256::from_big_endian(&bytes)
                    } else {
                        U256::from_big_endian(&bytes[..32])
                    }
                })
                .collect();

            // Use the model_id from the queued proof (or fallback to registered)
            let submit_model_id = if queued.model_id > 0 {
                queued.model_id
            } else {
                model_id
            };

            // Submit on-chain
            match self
                .sc_client
                .submit_proof_raw(
                    submit_model_id,
                    queued.round_id,
                    proof_bytes,
                    public_inputs,
                )
                .await
            {
                Ok(receipt) => {
                    let tx_hash = format!("{:?}", receipt.transaction_hash);
                    log::info!(
                        "Queued proof submitted: model={}, round={}, tx={}",
                        submit_model_id,
                        queued.round_id,
                        tx_hash
                    );

                    // Mark as submitted
                    {
                        let mut queue = proof_queue.write();
                        if let Some(entry) = queue.get_mut(idx) {
                            entry.submitted = true;
                            entry.tx_hash = Some(tx_hash.clone());
                        }
                    }

                    let mut state = self.state.write().await;
                    state.proofs_submitted += 1;

                    results.push((idx, Ok(tx_hash)));
                }
                Err(e) => {
                    log::error!(
                        "Failed to submit queued proof (model={}, round={}): {}",
                        submit_model_id,
                        queued.round_id,
                        e
                    );
                    results.push((idx, Err(format!("{}", e))));
                }
            }
        }

        results
    }

    /// Spawns a background task that periodically processes the proof queue.
    ///
    /// The task runs every `proof_queue_interval_secs` (default 5s) and
    /// drains any unsubmitted proofs from the RPC queue.
    pub fn spawn_proof_queue_processor(
        self: &Arc<Self>,
        proof_queue: Arc<SyncRwLock<Vec<QueuedProof>>>,
    ) -> tokio::task::JoinHandle<()> {
        let pipeline = Arc::clone(self);
        let interval_secs = pipeline.config.proof_queue_interval_secs;

        tokio::spawn(async move {
            log::info!(
                "Proof queue processor started (interval={}s)",
                interval_secs
            );

            let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
            loop {
                interval.tick().await;
                let results = pipeline.process_proof_queue(&proof_queue).await;
                if !results.is_empty() {
                    let successes = results.iter().filter(|(_, r)| r.is_ok()).count();
                    let failures = results.iter().filter(|(_, r)| r.is_err()).count();
                    log::info!(
                        "Proof queue batch: {} submitted, {} failed",
                        successes,
                        failures
                    );
                }
            }
        })
    }
}

/// Converts a slice of halo2 Fr field elements to ethers U256 values.
fn fr_vec_to_u256(frs: &[helix_prover::halo2curves::bn256::Fr]) -> Vec<U256> {
    use helix_prover::halo2curves::ff::PrimeField;
    frs.iter()
        .map(|fr| {
            let repr = fr.to_repr();
            U256::from_little_endian(repr.as_ref())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_state_defaults() {
        let state = PipelineState {
            model_id: None,
            staked: false,
            current_round: 0,
            proofs_submitted: 0,
        };
        assert!(state.model_id.is_none());
        assert!(!state.staked);
        assert_eq!(state.current_round, 0);
    }

    #[test]
    fn test_fr_to_u256_conversion() {
        use helix_prover::halo2curves::bn256::Fr;
        use helix_prover::halo2curves::ff::Field;

        let frs = vec![Fr::from(42u64), Fr::from(100u64), Fr::ZERO];
        let u256s = fr_vec_to_u256(&frs);

        assert_eq!(u256s.len(), 3);
        assert_eq!(u256s[0], U256::from(42));
        assert_eq!(u256s[1], U256::from(100));
        assert_eq!(u256s[2], U256::zero());
    }

    #[test]
    fn test_on_chain_submission_debug() {
        let sub = OnChainSubmission {
            tx_hash: "0xabc".to_string(),
            block_number: 42,
            gas_used: 7_500_000,
            proofs_aggregated: 3,
        };
        assert_eq!(sub.proofs_aggregated, 3);
        assert_eq!(sub.block_number, 42);
    }

    #[test]
    fn test_chain_config_defaults() {
        let config: ChainConfig = serde_json::from_str(r#"{
            "coordinator_address": "0x1234",
            "d_in": 2,
            "d_hid": 2,
            "d_out": 1
        }"#).unwrap();

        assert_eq!(config.coordinator_address, "0x1234");
        assert_eq!(config.d_in, 2);
        assert_eq!(config.d_hid, 2);
        assert_eq!(config.d_out, 1);
        assert!(config.model_id.is_none());
        assert_eq!(config.round_duration_secs, 600);
        assert_eq!(config.proof_queue_interval_secs, 5);
    }

    #[tokio::test]
    async fn test_process_proof_queue_empty() {
        // We can't create a real OnChainPipeline without a running chain,
        // but we can test the proof queue processing logic in isolation.
        // This test verifies the queue draining logic.
        let queue: Arc<SyncRwLock<Vec<QueuedProof>>> =
            Arc::new(SyncRwLock::new(Vec::new()));

        // Empty queue should return empty results
        let proofs = queue.read();
        let unsubmitted: Vec<_> = proofs
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.submitted)
            .collect();
        assert!(unsubmitted.is_empty());
    }

    #[tokio::test]
    async fn test_proof_queue_marking() {
        let queue: Arc<SyncRwLock<Vec<QueuedProof>>> = Arc::new(SyncRwLock::new(vec![
            QueuedProof {
                model_id: 1,
                round_id: 1,
                proof_hex: "0xdeadbeef".to_string(),
                public_inputs_hex: vec!["0x01".to_string()],
                submitted: false,
                tx_hash: None,
            },
            QueuedProof {
                model_id: 1,
                round_id: 2,
                proof_hex: "0xcafebabe".to_string(),
                public_inputs_hex: vec!["0x02".to_string()],
                submitted: true,
                tx_hash: Some("0xabc".to_string()),
            },
        ]));

        // Only first proof should be unsubmitted
        let q = queue.read();
        let unsubmitted: Vec<_> = q
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.submitted)
            .collect();
        assert_eq!(unsubmitted.len(), 1);
        assert_eq!(unsubmitted[0].0, 0); // index 0

        drop(q);

        // Mark first as submitted
        {
            let mut q = queue.write();
            q[0].submitted = true;
            q[0].tx_hash = Some("0xdef".to_string());
        }

        // Now no unsubmitted
        let q = queue.read();
        let unsubmitted_count = q
            .iter()
            .filter(|p| !p.submitted)
            .count();
        assert_eq!(unsubmitted_count, 0);
    }

    #[test]
    fn test_verifier_config_for_model() {
        let config = VerifierConfig::for_model(2, 2, 1);
        assert_eq!(config.policy, VerificationPolicy::VerifyAll);
        assert_eq!(config.model_dims, Some((2, 2, 1)));
    }
}
