//! Worker-side entry point for MPC training participation.
//!
//! This module implements the complete worker lifecycle for joining and
//! participating in a HELIX MPC training session:
//!
//! 1. **Startup** -- Generate x25519 key pair, bind TCP listener
//! 2. **Chain registration** -- (Optional) Stake ETH on the V4 contract
//! 3. **Share reception** -- Receive encrypted weight shares from the model owner
//! 4. **Training participation** -- The owner's orchestrator drives the MPC
//!    training loop; the worker participates via the share distribution protocol
//! 5. **Checkpoint signing** -- Sign Pedersen commitment attestations when the
//!    owner requests them over a control channel
//! 6. **MAC failure reporting** -- If a cheater is detected, sign the failure
//!    report for on-chain slashing
//! 7. **Share return** -- Send final trained weight shares back to the owner
//!
//! # Architecture
//!
//! The worker runs as a standalone async process. It exposes two TCP endpoints:
//!
//! - **Data channel** (primary listen address): Handles the share distribution
//!   and reconstruction protocol defined in `helix_mpc::network_distribution`.
//! - **Control channel** (data port + 1): Handles signing requests for
//!   checkpoints and MAC failure reports.
//!
//! The owner connects to both channels after the worker announces readiness.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[cfg(feature = "chain")]
use ethers::signers::LocalWallet;
#[cfg(feature = "chain")]
use ethers::types::{Address, U256};

use helix_mpc::network_distribution::{
    recv_message, send_message, worker_receive_distribution, worker_send_final_share,
    ProtocolMessage,
};
use helix_mpc::share_distribution::{generate_x25519_keypair, ShareReceiver, X25519PublicKey};
use helix_mpc::types::PartyId;

#[cfg(feature = "chain")]
use crate::rpc::chain_v4::{sign_checkpoint, sign_completion, sign_inference, sign_mac_failure, ChainClientV4};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a worker node participating in MPC training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// TCP address to listen on for data channel connections (e.g. "0.0.0.0:9001").
    pub listen_addr: String,
    /// This worker's party index (0-based).
    pub party_index: usize,
    /// Seed for deterministic x25519 key generation (for reproducible testing).
    pub seed: u64,
    /// Ethereum RPC URL for chain interaction (None = off-chain mode).
    #[cfg(feature = "chain")]
    pub eth_rpc_url: Option<String>,
    /// Hex-encoded Ethereum private key for signing and staking.
    #[cfg(feature = "chain")]
    pub private_key: String,
    /// The on-chain job ID to join (None = skip chain registration).
    #[cfg(feature = "chain")]
    pub job_id: Option<u64>,
    /// The V4 coordinator contract address.
    #[cfg(feature = "chain")]
    pub coordinator_address: Option<String>,
    /// Amount of ETH to stake when joining (in whole ETH, converted to wei).
    #[cfg(feature = "chain")]
    pub stake_amount_eth: f64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:9001".to_string(),
            party_index: 0,
            seed: 42,
            #[cfg(feature = "chain")]
            eth_rpc_url: None,
            #[cfg(feature = "chain")]
            private_key: String::new(),
            #[cfg(feature = "chain")]
            job_id: None,
            #[cfg(feature = "chain")]
            coordinator_address: None,
            #[cfg(feature = "chain")]
            stake_amount_eth: 0.1,
        }
    }
}

// ============================================================================
// Result types
// ============================================================================

/// Summary of a completed worker session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResult {
    /// Number of training steps completed during MPC participation.
    pub steps_completed: usize,
    /// Number of checkpoint attestation signatures produced.
    pub checkpoint_signatures: usize,
    /// Whether the final trained share was successfully sent back to the owner.
    pub final_share_sent: bool,
    /// Number of MAC failure (slashing) reports signed.
    pub slashing_reports_signed: usize,
}

// ============================================================================
// Control channel messages
// ============================================================================

/// Messages exchanged over the control channel between owner and worker.
///
/// The control channel runs on `data_port + 1` and handles signing requests
/// that occur during training (checkpoint attestations, MAC failure reports,
/// completion signatures).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Owner -> Worker: request signature on a checkpoint attestation.
    SignCheckpointRequest {
        job_id: u64,
        step_number: u64,
        weight_commitment: [u8; 32],
        loss_scaled: u64,
    },
    /// Worker -> Owner: checkpoint attestation signature.
    SignCheckpointResponse {
        signature: Vec<u8>,
    },
    /// Owner -> Worker: request signature on a MAC failure report.
    SignMacFailureRequest {
        job_id: u64,
        step_number: u64,
        cheater_address: [u8; 20],
        evidence: Vec<u8>,
    },
    /// Worker -> Owner: MAC failure report signature.
    SignMacFailureResponse {
        signature: Vec<u8>,
    },
    /// Owner -> Worker: request signature on the training completion message.
    SignCompletionRequest {
        job_id: u64,
        final_commitment: [u8; 32],
    },
    /// Worker -> Owner: completion signature.
    SignCompletionResponse {
        signature: Vec<u8>,
    },
    /// Owner -> Worker: request signature on an inference result attestation.
    SignInferenceRequest {
        job_id: u64,
        prediction: u64,
        input_hash: [u8; 32],
        output_hash: [u8; 32],
    },
    /// Worker -> Owner: inference attestation signature.
    SignInferenceResponse {
        signature: Vec<u8>,
    },
    /// Owner -> Worker: signal that training is complete and worker may shut down.
    TrainingComplete {
        steps_completed: usize,
    },
    /// Worker -> Owner: acknowledgement of completion.
    TrainingCompleteAck,
}

// ============================================================================
// Control channel framing (reuses the same length-prefix scheme)
// ============================================================================

/// Maximum control message size (1 MB -- control messages are small).
const MAX_CONTROL_MSG_SIZE: usize = 1024 * 1024;

/// Sends a control message over a TCP stream with 4-byte big-endian length prefix.
pub async fn send_control_message(
    stream: &mut tokio::net::TcpStream,
    msg: &ControlMessage,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let data = serde_json::to_vec(msg)
        .map_err(|e| anyhow!("control message serialize failed: {}", e))?;
    if data.len() > MAX_CONTROL_MSG_SIZE {
        return Err(anyhow!(
            "control message too large: {} bytes (max {})",
            data.len(),
            MAX_CONTROL_MSG_SIZE
        ));
    }
    let len = (data.len() as u32).to_be_bytes();
    stream
        .write_all(&len)
        .await
        .context("control: write length")?;
    stream
        .write_all(&data)
        .await
        .context("control: write body")?;
    stream.flush().await.context("control: flush")?;
    Ok(())
}

/// Receives a control message from a TCP stream with 4-byte big-endian length prefix.
pub async fn recv_control_message(
    stream: &mut tokio::net::TcpStream,
) -> Result<ControlMessage> {
    use tokio::io::AsyncReadExt;
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .context("control: read length")?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_CONTROL_MSG_SIZE {
        return Err(anyhow!(
            "control message too large: {} bytes (max {})",
            len,
            MAX_CONTROL_MSG_SIZE
        ));
    }
    let mut buf = vec![0u8; len];
    stream
        .read_exact(&mut buf)
        .await
        .context("control: read body")?;
    serde_json::from_slice(&buf)
        .map_err(|e| anyhow!("control message deserialize failed: {}", e))
}

// ============================================================================
// Worker signing service
// ============================================================================

/// Handles checkpoint and MAC failure signing requests received over the
/// control channel.
///
/// Runs as a background task that listens for signing requests from the owner
/// and produces ECDSA signatures using the worker's Ethereum wallet.
pub struct WorkerSigningService {
    /// Accumulated count of checkpoint signatures produced.
    checkpoint_count: Arc<RwLock<usize>>,
    /// Accumulated count of MAC failure report signatures produced.
    slashing_count: Arc<RwLock<usize>>,
    /// Number of training steps reported as completed by the owner.
    steps_completed: Arc<RwLock<usize>>,
}

impl WorkerSigningService {
    /// Creates a new signing service.
    pub fn new() -> Self {
        Self {
            checkpoint_count: Arc::new(RwLock::new(0)),
            slashing_count: Arc::new(RwLock::new(0)),
            steps_completed: Arc::new(RwLock::new(0)),
        }
    }

    /// Returns the number of checkpoint signatures produced so far.
    pub async fn checkpoint_count(&self) -> usize {
        *self.checkpoint_count.read().await
    }

    /// Returns the number of slashing report signatures produced so far.
    pub async fn slashing_count(&self) -> usize {
        *self.slashing_count.read().await
    }

    /// Returns the number of training steps completed as reported by the owner.
    pub async fn steps_completed(&self) -> usize {
        *self.steps_completed.read().await
    }

    /// Returns cloned counters for external tracking.
    pub fn counters(
        &self,
    ) -> (
        Arc<RwLock<usize>>,
        Arc<RwLock<usize>>,
        Arc<RwLock<usize>>,
    ) {
        (
            Arc::clone(&self.checkpoint_count),
            Arc::clone(&self.slashing_count),
            Arc::clone(&self.steps_completed),
        )
    }

    /// Runs the signing service loop on an accepted control channel connection.
    ///
    /// Processes signing requests until the owner sends `TrainingComplete` or
    /// the connection drops.
    ///
    /// When compiled without the `chain` feature, signing requests are responded
    /// to with placeholder (zero) signatures. This allows off-chain testing of
    /// the protocol flow without requiring an Ethereum wallet.
    pub async fn run(
        &self,
        mut stream: tokio::net::TcpStream,
        #[cfg(feature = "chain")] wallet: Option<LocalWallet>,
    ) -> Result<()> {
        info!("Worker signing service started on control channel");

        loop {
            let msg = match recv_control_message(&mut stream).await {
                Ok(m) => m,
                Err(e) => {
                    // Connection closed or read error -- treat as end of session.
                    debug!("Control channel closed: {}", e);
                    break;
                }
            };

            match msg {
                ControlMessage::SignCheckpointRequest {
                    job_id,
                    step_number,
                    weight_commitment,
                    loss_scaled,
                } => {
                    debug!(
                        job_id = job_id,
                        step = step_number,
                        "Checkpoint signing request received"
                    );
                    let sig_bytes = self
                        .handle_checkpoint_sign(
                            job_id,
                            step_number,
                            weight_commitment,
                            loss_scaled,
                            #[cfg(feature = "chain")]
                            wallet.as_ref(),
                        )
                        .await?;
                    send_control_message(
                        &mut stream,
                        &ControlMessage::SignCheckpointResponse {
                            signature: sig_bytes,
                        },
                    )
                    .await?;
                    {
                        let mut count = self.checkpoint_count.write().await;
                        *count += 1;
                    }
                    info!(step = step_number, "Checkpoint signature sent");
                }

                ControlMessage::SignMacFailureRequest {
                    job_id,
                    step_number,
                    cheater_address,
                    evidence,
                } => {
                    warn!(
                        job_id = job_id,
                        step = step_number,
                        "MAC failure signing request received -- cheater detected"
                    );
                    let sig_bytes = self
                        .handle_mac_failure_sign(
                            job_id,
                            step_number,
                            cheater_address,
                            &evidence,
                            #[cfg(feature = "chain")]
                            wallet.as_ref(),
                        )
                        .await?;
                    send_control_message(
                        &mut stream,
                        &ControlMessage::SignMacFailureResponse {
                            signature: sig_bytes,
                        },
                    )
                    .await?;
                    {
                        let mut count = self.slashing_count.write().await;
                        *count += 1;
                    }
                    warn!(step = step_number, "MAC failure signature sent");
                }

                ControlMessage::SignCompletionRequest {
                    job_id,
                    final_commitment,
                } => {
                    info!(job_id = job_id, "Completion signing request received");
                    let sig_bytes = self
                        .handle_completion_sign(
                            job_id,
                            final_commitment,
                            #[cfg(feature = "chain")]
                            wallet.as_ref(),
                        )
                        .await?;
                    send_control_message(
                        &mut stream,
                        &ControlMessage::SignCompletionResponse {
                            signature: sig_bytes,
                        },
                    )
                    .await?;
                    info!(job_id = job_id, "Completion signature sent");
                }

                ControlMessage::SignInferenceRequest {
                    job_id,
                    prediction,
                    input_hash,
                    output_hash,
                } => {
                    info!(job_id = job_id, prediction = prediction, "Inference signing request received");
                    let sig_bytes = self
                        .handle_inference_sign(
                            job_id,
                            prediction,
                            input_hash,
                            output_hash,
                            #[cfg(feature = "chain")]
                            wallet.as_ref(),
                        )
                        .await?;
                    send_control_message(
                        &mut stream,
                        &ControlMessage::SignInferenceResponse {
                            signature: sig_bytes,
                        },
                    )
                    .await?;
                    info!(job_id = job_id, "Inference attestation signature sent");
                }

                ControlMessage::TrainingComplete { steps_completed } => {
                    info!(steps = steps_completed, "Training complete signal received");
                    {
                        let mut s = self.steps_completed.write().await;
                        *s = steps_completed;
                    }
                    send_control_message(
                        &mut stream,
                        &ControlMessage::TrainingCompleteAck,
                    )
                    .await?;
                    break;
                }

                // Ignore response messages (should not arrive on the worker side).
                other => {
                    warn!("Unexpected control message on worker side: {:?}", other);
                }
            }
        }

        let final_cp = *self.checkpoint_count.read().await;
        let final_sl = *self.slashing_count.read().await;
        info!(
            checkpoints = final_cp,
            slashings = final_sl,
            "Worker signing service finished"
        );
        Ok(())
    }

    /// Produces an ECDSA signature for a checkpoint attestation.
    async fn handle_checkpoint_sign(
        &self,
        job_id: u64,
        step_number: u64,
        weight_commitment: [u8; 32],
        loss_scaled: u64,
        #[cfg(feature = "chain")] wallet: Option<&LocalWallet>,
    ) -> Result<Vec<u8>> {
        #[cfg(feature = "chain")]
        {
            if let Some(w) = wallet {
                let sig = sign_checkpoint(
                    w,
                    U256::from(job_id),
                    U256::from(step_number),
                    weight_commitment,
                    U256::from(loss_scaled),
                )
                .await
                .context("sign_checkpoint ECDSA")?;
                return Ok(sig.to_vec());
            }
        }
        // Off-chain fallback: produce a deterministic placeholder signature.
        Ok(placeholder_signature(b"checkpoint", job_id, step_number))
    }

    /// Produces an ECDSA signature for a MAC failure report.
    async fn handle_mac_failure_sign(
        &self,
        job_id: u64,
        step_number: u64,
        cheater_address: [u8; 20],
        evidence: &[u8],
        #[cfg(feature = "chain")] wallet: Option<&LocalWallet>,
    ) -> Result<Vec<u8>> {
        #[cfg(feature = "chain")]
        {
            if let Some(w) = wallet {
                let cheater = Address::from(cheater_address);
                let sig = sign_mac_failure(
                    w,
                    U256::from(job_id),
                    U256::from(step_number),
                    cheater,
                    evidence,
                )
                .await
                .context("sign_mac_failure ECDSA")?;
                return Ok(sig.to_vec());
            }
        }
        // Off-chain fallback.
        Ok(placeholder_signature(b"mac_failure", job_id, step_number))
    }

    /// Produces an ECDSA signature for training completion.
    async fn handle_completion_sign(
        &self,
        job_id: u64,
        final_commitment: [u8; 32],
        #[cfg(feature = "chain")] wallet: Option<&LocalWallet>,
    ) -> Result<Vec<u8>> {
        #[cfg(feature = "chain")]
        {
            if let Some(w) = wallet {
                let sig = sign_completion(w, U256::from(job_id), final_commitment)
                    .await
                    .context("sign_completion ECDSA")?;
                return Ok(sig.to_vec());
            }
        }
        // Off-chain fallback.
        let mut data = Vec::with_capacity(40);
        data.extend_from_slice(b"completion");
        data.extend_from_slice(&job_id.to_le_bytes());
        data.extend_from_slice(&final_commitment);
        Ok(sha2_hash(&data).to_vec())
    }

    /// Produces an ECDSA signature for an inference attestation.
    async fn handle_inference_sign(
        &self,
        job_id: u64,
        prediction: u64,
        input_hash: [u8; 32],
        output_hash: [u8; 32],
        #[cfg(feature = "chain")] wallet: Option<&LocalWallet>,
    ) -> Result<Vec<u8>> {
        #[cfg(feature = "chain")]
        {
            if let Some(w) = wallet {
                let sig = sign_inference(
                    w,
                    U256::from(job_id),
                    U256::from(prediction),
                    input_hash,
                    output_hash,
                )
                .await
                .context("sign_inference ECDSA")?;
                return Ok(sig.to_vec());
            }
        }
        // Off-chain fallback.
        let mut data = Vec::with_capacity(80);
        data.extend_from_slice(b"inference");
        data.extend_from_slice(&job_id.to_le_bytes());
        data.extend_from_slice(&prediction.to_le_bytes());
        data.extend_from_slice(&input_hash);
        data.extend_from_slice(&output_hash);
        Ok(sha2_hash(&data).to_vec())
    }
}

/// Produces a deterministic placeholder signature for off-chain mode.
/// Uses SHA-256 of the domain tag, job ID, and step number.
fn placeholder_signature(domain: &[u8], job_id: u64, step_number: u64) -> Vec<u8> {
    let mut data = Vec::with_capacity(domain.len() + 16);
    data.extend_from_slice(domain);
    data.extend_from_slice(&job_id.to_le_bytes());
    data.extend_from_slice(&step_number.to_le_bytes());
    sha2_hash(&data).to_vec()
}

/// SHA-256 helper that returns a 32-byte hash.
fn sha2_hash(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

// ============================================================================
// WorkerRunner -- main orchestration
// ============================================================================

/// Runs the complete worker lifecycle for a single MPC training session.
///
/// The runner manages:
/// - x25519 key generation for encrypted share transport
/// - (Optional) on-chain staking via ChainClientV4
/// - TCP listener for incoming owner connections (data + control channels)
/// - Share reception, training participation, and share return
/// - Concurrent control channel for signing requests
pub struct WorkerRunner {
    config: WorkerConfig,
}

impl WorkerRunner {
    /// Creates a new worker runner with the given configuration.
    pub fn new(config: WorkerConfig) -> Self {
        Self { config }
    }

    /// Returns the party ID for this worker.
    pub fn party_id(&self) -> PartyId {
        PartyId::from_index(self.config.party_index)
    }

    /// Derives the control channel address from the data channel address.
    ///
    /// The control channel listens on `data_port + 1`.
    fn control_addr(data_addr: &SocketAddr) -> SocketAddr {
        SocketAddr::new(data_addr.ip(), data_addr.port() + 1)
    }

    /// Runs the full worker session from startup to completion.
    ///
    /// # Phases
    ///
    /// 1. Generate x25519 key pair for share encryption
    /// 2. Optionally stake on-chain via ChainClientV4
    /// 3. Start TCP listeners (data + control channels)
    /// 4. Accept owner connection on the data channel
    /// 5. Receive encrypted weight shares
    /// 6. Spawn the control channel signing service
    /// 7. Wait for training to complete (the owner drives the MPC loop)
    /// 8. Send final trained shares back to the owner
    /// 9. Log completion summary
    pub async fn run(&self) -> Result<WorkerResult> {
        let start = Instant::now();
        let party_id = self.party_id();

        info!(
            party = %party_id,
            listen = %self.config.listen_addr,
            index = self.config.party_index,
            "Worker starting up"
        );

        // ----------------------------------------------------------------
        // Phase 1: Generate x25519 key pair
        // ----------------------------------------------------------------
        let mut rng = seeded_rng(self.config.seed, self.config.party_index);
        let (secret_key, public_key) = generate_x25519_keypair(&mut rng);

        info!(
            party = %party_id,
            public_key = hex::encode(public_key.as_bytes()),
            "x25519 key pair generated"
        );

        // ----------------------------------------------------------------
        // Phase 2: On-chain staking (optional)
        // ----------------------------------------------------------------
        #[cfg(feature = "chain")]
        {
            if let Some(job_id) = self.config.job_id {
                self.stake_on_chain(job_id).await?;
            }
        }

        // ----------------------------------------------------------------
        // Phase 3: Start TCP listeners
        // ----------------------------------------------------------------
        let data_addr: SocketAddr = self
            .config
            .listen_addr
            .parse()
            .context("Invalid listen address")?;
        let control_addr = Self::control_addr(&data_addr);

        let data_listener = TcpListener::bind(data_addr)
            .await
            .with_context(|| format!("Failed to bind data channel on {}", data_addr))?;
        let actual_data_addr = data_listener.local_addr()?;

        info!(
            party = %party_id,
            data_addr = %actual_data_addr,
            control_addr = %control_addr,
            "Data listener ready -- waiting for owner connection"
        );

        // ----------------------------------------------------------------
        // Phase 4: Accept data connection and receive encrypted shares
        // ----------------------------------------------------------------
        let receiver = ShareReceiver::new(secret_key.clone(), party_id.clone());

        info!(party = %party_id, "Waiting for share distribution on data channel");

        let (share_state, mut data_stream) =
            worker_receive_distribution(&data_listener, &receiver, &secret_key)
                .await
                .map_err(|e| anyhow!("Share distribution failed: {}", e))?;

        info!(
            party = %party_id,
            share_len = share_state.weight_share.data.len(),
            "Weight shares received and commitment verified"
        );

        // ----------------------------------------------------------------
        // Phase 5: Bind control channel and accept owner connection
        // ----------------------------------------------------------------
        // The control port is bound AFTER shares are received (not at startup)
        // so that idle workers don't hit the 120s accept timeout while waiting
        // for the next training job. The orchestrator connects to control ports
        // after distribute_shares completes, so the listener will be ready.
        let control_listener = TcpListener::bind(control_addr)
            .await
            .with_context(|| format!("Failed to bind control channel on {}", control_addr))?;
        let actual_control_addr = control_listener.local_addr()?;
        info!(party = %party_id, control_addr = %actual_control_addr, "Control listener bound");

        let control_accept = tokio::spawn(async move {
            let accept_fut = control_listener.accept();
            let (stream, addr) = tokio::time::timeout(Duration::from_secs(30), accept_fut)
                .await
                .map_err(|_| anyhow!("Control channel accept timed out after 30s — owner never connected"))?
                .map_err(|e| anyhow!("Control channel accept failed: {}", e))?;
            info!(addr = %addr, "Control channel connection accepted");
            Ok::<_, anyhow::Error>(stream)
        });

        // ----------------------------------------------------------------
        // Phase 6: Start control channel signing service
        // ----------------------------------------------------------------
        let signing_service = WorkerSigningService::new();
        let (cp_counter, slash_counter, steps_counter) = signing_service.counters();

        #[cfg(feature = "chain")]
        let wallet_for_signing = self.build_wallet().ok();

        let control_stream = control_accept
            .await
            .context("Control channel accept task panicked")?
            .context("Control channel accept failed")?;

        let signing_handle = tokio::spawn({
            #[cfg(feature = "chain")]
            let wallet_clone = wallet_for_signing.clone();
            async move {
                signing_service
                    .run(
                        control_stream,
                        #[cfg(feature = "chain")]
                        wallet_clone,
                    )
                    .await
            }
        });

        info!(party = %party_id, "Control channel signing service spawned");

        // ----------------------------------------------------------------
        // Phase 7: Participate in MPC training
        // ----------------------------------------------------------------
        // The owner sends the next message on the data channel:
        //   - ReconstructionRequest → passive mode (owner runs training in-process)
        //   - StartDistributedTraining → active mode (worker runs training)
        //
        // In passive mode, the worker just waits and returns its shares.
        // In distributed mode, the worker creates a TcpTransport mesh with
        // peers and runs the full MPC training loop locally.
        info!(
            party = %party_id,
            "Worker waiting for training command on data channel"
        );

        let next_msg = recv_message(&mut data_stream).await
            .map_err(|e| anyhow!("Failed to receive training command: {}", e))?;

        let (final_share_sent, distributed_result) = match next_msg {
            ProtocolMessage::StartDistributedTraining {
                trainer_config,
                peer_addrs,
                training_data,
                weight_layout,
                owner_public_key,
                num_steps,
                checkpoint_interval,
                seed,
                mpc_bind_addr,
            } => {
                info!(
                    party = %party_id,
                    num_steps = num_steps,
                    peers = peer_addrs.len(),
                    mpc_addr = %mpc_bind_addr,
                    "Starting distributed MPC training"
                );

                // Split existing weight shares by layout.
                let (w1, b1, w2, b2) = weight_layout.split(&share_state.weight_share.data);

                // Run distributed training.
                let result = run_distributed_training(
                    self.config.party_index,
                    &mpc_bind_addr,
                    &peer_addrs,
                    trainer_config,
                    w1, b1, w2, b2,
                    training_data,
                    num_steps,
                    checkpoint_interval,
                    seed,
                    owner_public_key,
                    weight_layout,
                ).await?;

                // Send result back over data channel.
                let response = ProtocolMessage::DistributedTrainingResult {
                    steps_completed: result.steps_completed,
                    losses: result.losses.clone(),
                    mac_checks_passed: result.mac_checks_passed,
                    cheater_detected: result.cheater_detected.is_some(),
                    cheater_party: result.cheater_detected.as_ref().map(|c| c.party_index),
                    encrypted_final_share: result.encrypted_final_share.clone(),
                    checkpoints: result.checkpoints.clone(),
                };
                send_message(&mut data_stream, &response).await
                    .map_err(|e| anyhow!("Failed to send training result: {}", e))?;

                info!(
                    party = %party_id,
                    steps = result.steps_completed,
                    final_loss = result.losses.last().copied().unwrap_or(0.0),
                    "Distributed training complete, result sent to owner"
                );

                (true, Some(result))
            }

            ProtocolMessage::StartInference {
                input,
                peer_addrs,
                mpc_bind_addr,
                weight_layout,
                seed: _seed,
                num_parties,
            } => {
                info!(
                    party = %party_id,
                    input_len = input.len(),
                    num_parties = num_parties,
                    peers = peer_addrs.len(),
                    mpc_addr = %mpc_bind_addr,
                    "Starting distributed MPC inference"
                );

                // Split existing weight shares by layout.
                let (w1, b1, w2, b2) = weight_layout.split(&share_state.weight_share.data);
                let in_features = w1.len() / b1.len();
                let hidden_features = b1.len();
                let out_features = b2.len();

                // Run distributed inference.
                let result = run_distributed_inference(
                    self.config.party_index,
                    &mpc_bind_addr,
                    &peer_addrs,
                    w1, b1, w2, b2,
                    &input,
                    in_features,
                    hidden_features,
                    out_features,
                ).await?;

                // Send result back over data channel.
                let response = ProtocolMessage::InferenceResult {
                    prediction: result.prediction,
                    confidence_scaled: (result.confidence * 10000.0) as u64,
                    probabilities: result.probabilities.clone(),
                    output_hash: result.output_hash,
                    input_hash: result.input_hash,
                };
                send_message(&mut data_stream, &response).await
                    .map_err(|e| anyhow!("Failed to send inference result: {}", e))?;

                info!(
                    party = %party_id,
                    prediction = result.prediction,
                    confidence = format!("{:.2}%", result.confidence * 100.0),
                    "Distributed inference complete, result sent to owner"
                );

                (true, None)
            }

            ProtocolMessage::ReconstructionRequest { owner_public_key } => {
                // Passive mode: owner ran training in-process, just return shares.
                info!(party = %party_id, "Passive mode: sending final shares to owner");

                // Re-inject the ReconstructionRequest back through the protocol.
                // worker_send_final_share expects to read ReconstructionRequest,
                // but we already consumed it. Send the response directly instead.
                let owner_pk = X25519PublicKey::from(owner_public_key);
                let mut rng = seeded_rng(
                    self.config.seed.wrapping_add(1000),
                    self.config.party_index,
                );

                let encrypted_share = helix_mpc::share_distribution::encrypt_share_for_owner(
                    &share_state.weight_share,
                    &owner_pk,
                    &mut rng,
                ).map_err(|e| anyhow!("encrypt_share_for_owner failed: {}", e))?;

                let blindings: Vec<helix_mpc::field::Fr> = (0..share_state.weight_share.data.len())
                    .map(|_| helix_mpc::field::Fr::random(&mut rng))
                    .collect();

                let checkpoint_commit = helix_mpc::share_distribution::CheckpointCommitment::with_generators(
                    share_state.generators.clone(),
                );
                let commitment_share = checkpoint_commit.compute_share(
                    &share_state.weight_share,
                    &blindings,
                ).map_err(|e| anyhow!("compute_share failed: {}", e))?;

                let response = ProtocolMessage::FinalShareResponse {
                    encrypted_share,
                    commitment_share,
                    blindings: helix_mpc::network_distribution::FrVecPayload::from_fr_vec(&blindings),
                };
                send_message(&mut data_stream, &response).await
                    .map_err(|e| anyhow!("Failed to send final share: {}", e))?;

                info!(party = %party_id, "Final weight shares sent to owner (passive mode)");
                (true, None)
            }

            other => {
                error!(
                    party = %party_id,
                    msg_type = ?std::mem::discriminant(&other),
                    "Unexpected message after share distribution"
                );
                (false, None)
            }
        };

        // ----------------------------------------------------------------
        // Phase 9: Wait for signing service to finish
        // ----------------------------------------------------------------
        // The signing service will terminate when the owner sends
        // TrainingComplete or the control channel drops.
        let signing_timeout = Duration::from_secs(300);
        match tokio::time::timeout(signing_timeout, signing_handle).await {
            Ok(Ok(Ok(()))) => {
                debug!(party = %party_id, "Signing service completed cleanly");
            }
            Ok(Ok(Err(e))) => {
                warn!(party = %party_id, error = %e, "Signing service error");
            }
            Ok(Err(e)) => {
                warn!(party = %party_id, error = %e, "Signing service task panicked");
            }
            Err(_) => {
                warn!(
                    party = %party_id,
                    timeout_secs = signing_timeout.as_secs(),
                    "Signing service timed out -- proceeding with shutdown"
                );
            }
        }

        // ----------------------------------------------------------------
        // Phase 10: Completion summary
        // ----------------------------------------------------------------
        let checkpoint_signatures = *cp_counter.read().await;
        let slashing_reports_signed = *slash_counter.read().await;
        let steps_completed = if let Some(ref dr) = distributed_result {
            dr.steps_completed
        } else {
            *steps_counter.read().await
        };

        let elapsed = start.elapsed();
        let result = WorkerResult {
            steps_completed,
            checkpoint_signatures,
            final_share_sent,
            slashing_reports_signed,
        };

        info!(
            party = %party_id,
            steps = result.steps_completed,
            checkpoints_signed = result.checkpoint_signatures,
            slashing_signed = result.slashing_reports_signed,
            final_share_sent = result.final_share_sent,
            elapsed_ms = elapsed.as_millis(),
            "Worker session complete"
        );

        Ok(result)
    }

    /// Stakes ETH on the V4 contract for the specified job.
    #[cfg(feature = "chain")]
    async fn stake_on_chain(&self, job_id: u64) -> Result<()> {
        let rpc_url = self
            .config
            .eth_rpc_url
            .as_ref()
            .ok_or_else(|| anyhow!("eth_rpc_url is required for on-chain staking"))?;
        let coord_addr = self
            .config
            .coordinator_address
            .as_ref()
            .ok_or_else(|| anyhow!("coordinator_address is required for on-chain staking"))?;

        let chain_client = ChainClientV4::new(
            rpc_url,
            &self.config.private_key,
            coord_addr,
            None,
        )
        .await
        .context("Failed to create V4 chain client")?;

        let stake_wei = eth_to_wei(self.config.stake_amount_eth);
        info!(
            party = %self.party_id(),
            job_id = job_id,
            stake_eth = self.config.stake_amount_eth,
            stake_wei = %stake_wei,
            "Staking on-chain"
        );

        let receipt = chain_client
            .stake_and_join(job_id, stake_wei)
            .await
            .context("stake_and_join transaction failed")?;

        info!(
            party = %self.party_id(),
            job_id = job_id,
            tx_hash = ?receipt.transaction_hash,
            gas_used = ?receipt.gas_used,
            "Successfully staked and joined job"
        );

        // Verify we are now an active worker.
        let signer_addr = chain_client.signer_address();
        let is_active = chain_client
            .is_active_worker(job_id, signer_addr)
            .await
            .context("is_active_worker check failed")?;

        if !is_active {
            return Err(anyhow!(
                "Worker {} not listed as active after staking -- contract may have rejected",
                signer_addr
            ));
        }

        let summary = chain_client
            .get_job_summary(job_id)
            .await
            .context("get_job_summary failed")?;
        info!(
            job_id = job_id,
            active_workers = summary.active_worker_count,
            current_step = summary.current_step,
            "Job summary after staking"
        );

        Ok(())
    }

    /// Builds a LocalWallet from the configured private key.
    #[cfg(feature = "chain")]
    fn build_wallet(&self) -> Result<LocalWallet> {
        use std::str::FromStr;
        let pk = self
            .config
            .private_key
            .strip_prefix("0x")
            .unwrap_or(&self.config.private_key);
        LocalWallet::from_str(pk).map_err(|e| anyhow!("Invalid private key: {}", e))
    }
}

// ============================================================================
// Standalone worker launcher
// ============================================================================

/// Launches a standalone worker process with the given configuration.
///
/// This is the top-level function intended for use in a worker binary.
/// It creates a `WorkerRunner`, runs it to completion, and returns the result.
///
/// # Example
///
/// ```ignore
/// let config = WorkerConfig {
///     listen_addr: "0.0.0.0:9001".to_string(),
///     party_index: 0,
///     seed: 42,
///     ..Default::default()
/// };
/// let result = launch_worker(config).await?;
/// println!("Worker completed: {:?}", result);
/// ```
pub async fn launch_worker(config: WorkerConfig) -> Result<WorkerResult> {
    let runner = WorkerRunner::new(config);
    runner.run().await
}

/// Launches a worker that binds to an OS-assigned port and returns the actual
/// bound addresses alongside the result.
///
/// Useful for integration tests where port conflicts must be avoided.
/// The worker binds to `127.0.0.1:0` (data) and `127.0.0.1:1` offset for control.
///
/// Returns `(data_addr, control_addr, WorkerResult)`.
pub async fn launch_worker_ephemeral(
    party_index: usize,
    seed: u64,
) -> Result<(SocketAddr, SocketAddr, WorkerResult)> {
    // Bind the data listener on an ephemeral port first to discover the port.
    let data_listener = TcpListener::bind("127.0.0.1:0").await?;
    let data_addr = data_listener.local_addr()?;

    // Bind the control listener on data_port + 1.
    // If that port is taken, try the next few ports.
    let control_addr = find_control_port(data_addr).await?;
    let control_listener = TcpListener::bind(control_addr).await?;
    let actual_control_addr = control_listener.local_addr()?;

    info!(
        party_index = party_index,
        data_addr = %data_addr,
        control_addr = %actual_control_addr,
        "Ephemeral worker listeners bound"
    );

    let config = WorkerConfig {
        listen_addr: data_addr.to_string(),
        party_index,
        seed,
        ..Default::default()
    };

    // Run the worker with the pre-bound listeners.
    let result = run_worker_with_listeners(config, data_listener, control_listener).await?;
    Ok((data_addr, actual_control_addr, result))
}

/// Finds an available port for the control channel near the data port.
///
/// Tries `data_port + 1` first, then scans up to `data_port + 10`.
async fn find_control_port(data_addr: SocketAddr) -> Result<SocketAddr> {
    let base_port = data_addr.port();
    for offset in 1..=10 {
        let candidate = SocketAddr::new(data_addr.ip(), base_port + offset);
        match TcpListener::bind(candidate).await {
            Ok(listener) => {
                // Port is available. Drop the listener so we can re-bind later.
                let addr = listener.local_addr()?;
                drop(listener);
                return Ok(addr);
            }
            Err(_) => continue,
        }
    }
    Err(anyhow!(
        "Could not find an available control port near {}",
        data_addr
    ))
}

/// Runs the worker protocol using pre-bound TCP listeners.
///
/// This is the internal implementation used by both `launch_worker` and
/// `launch_worker_ephemeral`. It follows the same phase structure as
/// `WorkerRunner::run` but accepts pre-existing listeners.
async fn run_worker_with_listeners(
    config: WorkerConfig,
    data_listener: TcpListener,
    control_listener: TcpListener,
) -> Result<WorkerResult> {
    let start = Instant::now();
    let party_id = PartyId::from_index(config.party_index);

    // Generate x25519 keys.
    let mut rng = seeded_rng(config.seed, config.party_index);
    let (secret_key, public_key) = generate_x25519_keypair(&mut rng);

    info!(
        party = %party_id,
        public_key = hex::encode(public_key.as_bytes()),
        "Ephemeral worker x25519 key generated"
    );

    let receiver = ShareReceiver::new(secret_key.clone(), party_id.clone());

    // Accept connections concurrently with timeout.
    let control_accept = tokio::spawn(async move {
        let accept_fut = control_listener.accept();
        let (stream, addr) = tokio::time::timeout(Duration::from_secs(120), accept_fut)
            .await
            .map_err(|_| anyhow!("Ephemeral control accept timed out after 120s"))?
            .map_err(|e| anyhow!("Ephemeral control accept failed: {}", e))?;
        info!(addr = %addr, "Ephemeral control channel accepted");
        Ok::<_, anyhow::Error>(stream)
    });

    // Receive shares on data channel.
    let (share_state, mut data_stream) =
        worker_receive_distribution(&data_listener, &receiver, &secret_key)
            .await
            .map_err(|e| anyhow!("Share distribution failed: {}", e))?;

    info!(
        party = %party_id,
        share_len = share_state.weight_share.data.len(),
        "Ephemeral worker shares received"
    );

    // Start signing service.
    let signing_service = WorkerSigningService::new();
    let (cp_counter, slash_counter, steps_counter) = signing_service.counters();

    let control_stream = control_accept
        .await
        .context("Control accept panicked")?
        .context("Control accept failed")?;

    let signing_handle = tokio::spawn(async move {
        signing_service
            .run(
                control_stream,
                #[cfg(feature = "chain")]
                None,
            )
            .await
    });

    // Wait for reconstruction request and send final shares.
    let final_share_sent = match worker_send_final_share(
        &mut data_stream,
        &share_state.weight_share,
        &share_state.generators,
        Some(config.seed.wrapping_add(config.party_index as u64 + 1000)),
    )
    .await
    {
        Ok(()) => true,
        Err(e) => {
            error!(party = %party_id, error = %e, "Failed to send final shares");
            false
        }
    };

    // Wait for signing service.
    let _ = tokio::time::timeout(Duration::from_secs(30), signing_handle).await;

    let elapsed = start.elapsed();
    let result = WorkerResult {
        steps_completed: *steps_counter.read().await,
        checkpoint_signatures: *cp_counter.read().await,
        final_share_sent,
        slashing_reports_signed: *slash_counter.read().await,
    };

    info!(
        party = %party_id,
        elapsed_ms = elapsed.as_millis(),
        "Ephemeral worker session complete: {:?}",
        result
    );

    Ok(result)
}

// ============================================================================
// Utility helpers
// ============================================================================

/// Creates a deterministic RNG from a seed and party index.
///
/// Mixes the party index into the seed to ensure each worker in the same
/// session gets a distinct key stream.
fn seeded_rng(seed: u64, party_index: usize) -> rand::rngs::StdRng {
    use rand::SeedableRng;
    let mixed = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(party_index as u64);
    rand::rngs::StdRng::seed_from_u64(mixed)
}

/// Converts ETH (as f64) to wei (as U256).
#[cfg(feature = "chain")]
fn eth_to_wei(eth: f64) -> U256 {
    // 1 ETH = 1e18 wei. We multiply in two steps to avoid f64 precision issues
    // for values > 1e15 wei.
    let milliether = (eth * 1000.0) as u64;
    U256::from(milliether) * U256::from(10u64.pow(15))
}

/// Returns the x25519 public key bytes for a worker given its seed and index.
///
/// Convenience function for the owner to compute a worker's expected public
/// key when using deterministic seeds.
pub fn worker_public_key_from_seed(seed: u64, party_index: usize) -> [u8; 32] {
    let mut rng = seeded_rng(seed, party_index);
    let (_secret, public) = generate_x25519_keypair(&mut rng);
    *public.as_bytes()
}

// ============================================================================
// Distributed training helper
// ============================================================================

/// Runs distributed MPC training on a worker that already has decrypted shares.
///
/// Creates a TcpTransport mesh with peer workers and executes the full MPC
/// training loop (Beaver triple generation, MAC-verified training, checkpoints).
#[cfg(feature = "network-mpc")]
async fn run_distributed_training(
    party_index: usize,
    mpc_bind_addr: &str,
    peer_addrs: &[(String, String)],
    trainer_config: helix_mpc::mpc_trainer::MPCTrainerConfig,
    w1: Vec<helix_mpc::field::Fr>,
    b1: Vec<helix_mpc::field::Fr>,
    w2: Vec<helix_mpc::field::Fr>,
    b2: Vec<helix_mpc::field::Fr>,
    training_data: Vec<(Vec<f64>, Vec<f64>)>,
    num_steps: usize,
    checkpoint_interval: usize,
    seed: u64,
    owner_public_key: [u8; 32],
    weight_layout: helix_mpc::e2e_integration::WeightLayout,
) -> Result<helix_mpc::e2e_integration::PartyResult> {
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use helix_mpc::session::transport::TcpTransport;

    let bind_addr: SocketAddr = mpc_bind_addr.parse()
        .with_context(|| format!("Invalid MPC bind address: {}", mpc_bind_addr))?;

    let party_id = PartyId::from_index(party_index);

    // Build peer address map (excluding self).
    let mut peers: HashMap<PartyId, SocketAddr> = HashMap::new();
    for (pid_str, addr_str) in peer_addrs {
        let pid = PartyId::new(pid_str);
        if pid != party_id {
            let addr: SocketAddr = addr_str.parse()
                .with_context(|| format!("Invalid peer address: {}", addr_str))?;
            peers.insert(pid, addr);
        }
    }

    info!(
        party = %party_id,
        bind_addr = %bind_addr,
        num_peers = peers.len(),
        "Creating TcpTransport mesh for distributed training"
    );

    // Create TcpTransport — this does the deterministic connect/accept handshake.
    let transport = TcpTransport::bind(bind_addr, party_id, &peers).await
        .map_err(|e| anyhow!("TcpTransport::bind failed: {}", e))?;

    info!(
        party_index = party_index,
        "TcpTransport mesh established, starting distributed training"
    );

    let owner_pk = helix_mpc::share_distribution::X25519PublicKey::from(owner_public_key);

    // Run the training loop with pre-decrypted shares.
    helix_mpc::e2e_integration::run_distributed_party(
        trainer_config,
        transport,
        party_index,
        w1, b1, w2, b2,
        training_data,
        num_steps,
        checkpoint_interval,
        seed,
        Some(owner_pk),
        Some(weight_layout),
    ).await
}

/// Stub for when the `network-mpc` feature is not enabled.
#[cfg(not(feature = "network-mpc"))]
async fn run_distributed_training(
    _party_index: usize,
    _mpc_bind_addr: &str,
    _peer_addrs: &[(String, String)],
    _trainer_config: helix_mpc::mpc_trainer::MPCTrainerConfig,
    _w1: Vec<helix_mpc::field::Fr>,
    _b1: Vec<helix_mpc::field::Fr>,
    _w2: Vec<helix_mpc::field::Fr>,
    _b2: Vec<helix_mpc::field::Fr>,
    _training_data: Vec<(Vec<f64>, Vec<f64>)>,
    _num_steps: usize,
    _checkpoint_interval: usize,
    _seed: u64,
    _owner_public_key: [u8; 32],
    _weight_layout: helix_mpc::e2e_integration::WeightLayout,
) -> Result<helix_mpc::e2e_integration::PartyResult> {
    Err(anyhow!("Distributed training requires the 'network-mpc' feature"))
}

// ============================================================================
// Distributed inference
// ============================================================================

/// Runs the distributed MPC forward pass for inference.
///
/// Creates a TcpTransport mesh with peers, then executes the
/// `distributed_forward_pass` from `helix_mpc::distributed_inference`.
#[cfg(feature = "network-mpc")]
async fn run_distributed_inference(
    party_index: usize,
    mpc_bind_addr: &str,
    peer_addrs: &[(String, String)],
    w1: Vec<helix_mpc::field::Fr>,
    b1: Vec<helix_mpc::field::Fr>,
    w2: Vec<helix_mpc::field::Fr>,
    b2: Vec<helix_mpc::field::Fr>,
    input: &[f64],
    in_features: usize,
    hidden_features: usize,
    out_features: usize,
) -> Result<helix_mpc::distributed_inference::DistributedInferenceResult> {
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use helix_mpc::session::transport::TcpTransport;
    use helix_mpc::distributed_inference::{WorkerWeightShares, distributed_forward_pass};

    let bind_addr: SocketAddr = mpc_bind_addr.parse()
        .with_context(|| format!("Invalid MPC bind address: {}", mpc_bind_addr))?;

    let party_id = PartyId::from_index(party_index);

    // Build peer address map (excluding self).
    let mut peers: HashMap<PartyId, SocketAddr> = HashMap::new();
    for (pid_str, addr_str) in peer_addrs {
        let pid = PartyId::new(pid_str);
        if pid != party_id {
            let addr: SocketAddr = addr_str.parse()
                .with_context(|| format!("Invalid peer address: {}", addr_str))?;
            peers.insert(pid, addr);
        }
    }

    info!(
        party = %party_id,
        bind_addr = %bind_addr,
        num_peers = peers.len(),
        "Creating TcpTransport mesh for distributed inference"
    );

    // Create TcpTransport — deterministic connect/accept handshake.
    let transport = TcpTransport::bind(bind_addr, party_id, &peers).await
        .map_err(|e| anyhow!("TcpTransport::bind failed: {}", e))?;

    info!(
        party_index = party_index,
        "TcpTransport mesh established, starting distributed inference"
    );

    let shares = WorkerWeightShares { w1, b1, w2, b2 };

    distributed_forward_pass(
        &transport,
        &shares,
        input,
        in_features,
        hidden_features,
        out_features,
    ).await
    .map_err(|e| anyhow!("Distributed inference failed: {}", e))
}

/// Stub for when the `network-mpc` feature is not enabled.
#[cfg(not(feature = "network-mpc"))]
async fn run_distributed_inference(
    _party_index: usize,
    _mpc_bind_addr: &str,
    _peer_addrs: &[(String, String)],
    _w1: Vec<helix_mpc::field::Fr>,
    _b1: Vec<helix_mpc::field::Fr>,
    _w2: Vec<helix_mpc::field::Fr>,
    _b2: Vec<helix_mpc::field::Fr>,
    _input: &[f64],
    _in_features: usize,
    _hidden_features: usize,
    _out_features: usize,
) -> Result<helix_mpc::distributed_inference::DistributedInferenceResult> {
    Err(anyhow!("Distributed inference requires the 'network-mpc' feature"))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.listen_addr, "0.0.0.0:9001");
        assert_eq!(config.party_index, 0);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_worker_result_serialization() {
        let result = WorkerResult {
            steps_completed: 100,
            checkpoint_signatures: 10,
            final_share_sent: true,
            slashing_reports_signed: 0,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: WorkerResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.steps_completed, 100);
        assert_eq!(parsed.checkpoint_signatures, 10);
        assert!(parsed.final_share_sent);
        assert_eq!(parsed.slashing_reports_signed, 0);
    }

    #[test]
    fn test_control_message_serialization() {
        let msg = ControlMessage::SignCheckpointRequest {
            job_id: 42,
            step_number: 10,
            weight_commitment: [0xAB; 32],
            loss_scaled: 1_000_000,
        };
        let bytes = serde_json::to_vec(&msg).unwrap();
        let parsed: ControlMessage = serde_json::from_slice(&bytes).unwrap();
        match parsed {
            ControlMessage::SignCheckpointRequest {
                job_id,
                step_number,
                ..
            } => {
                assert_eq!(job_id, 42);
                assert_eq!(step_number, 10);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_control_message_mac_failure_roundtrip() {
        let msg = ControlMessage::SignMacFailureRequest {
            job_id: 1,
            step_number: 5,
            cheater_address: [0xDE; 20],
            evidence: vec![1, 2, 3, 4],
        };
        let bytes = serde_json::to_vec(&msg).unwrap();
        let parsed: ControlMessage = serde_json::from_slice(&bytes).unwrap();
        match parsed {
            ControlMessage::SignMacFailureRequest {
                cheater_address,
                evidence,
                ..
            } => {
                assert_eq!(cheater_address, [0xDE; 20]);
                assert_eq!(evidence, vec![1, 2, 3, 4]);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_control_message_completion_roundtrip() {
        let msg = ControlMessage::SignCompletionRequest {
            job_id: 99,
            final_commitment: [0xFF; 32],
        };
        let bytes = serde_json::to_vec(&msg).unwrap();
        let parsed: ControlMessage = serde_json::from_slice(&bytes).unwrap();
        match parsed {
            ControlMessage::SignCompletionRequest {
                job_id,
                final_commitment,
            } => {
                assert_eq!(job_id, 99);
                assert_eq!(final_commitment, [0xFF; 32]);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_placeholder_signature_deterministic() {
        let sig1 = placeholder_signature(b"test", 1, 2);
        let sig2 = placeholder_signature(b"test", 1, 2);
        assert_eq!(sig1, sig2);
        assert_eq!(sig1.len(), 32);
    }

    #[test]
    fn test_placeholder_signature_varies() {
        let sig1 = placeholder_signature(b"test", 1, 2);
        let sig2 = placeholder_signature(b"test", 1, 3);
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_seeded_rng_deterministic() {
        let mut rng1 = seeded_rng(42, 0);
        let mut rng2 = seeded_rng(42, 0);
        let (_, pk1) = generate_x25519_keypair(&mut rng1);
        let (_, pk2) = generate_x25519_keypair(&mut rng2);
        assert_eq!(pk1.as_bytes(), pk2.as_bytes());
    }

    #[test]
    fn test_seeded_rng_varies_by_index() {
        let mut rng0 = seeded_rng(42, 0);
        let mut rng1 = seeded_rng(42, 1);
        let (_, pk0) = generate_x25519_keypair(&mut rng0);
        let (_, pk1) = generate_x25519_keypair(&mut rng1);
        assert_ne!(pk0.as_bytes(), pk1.as_bytes());
    }

    #[test]
    fn test_worker_public_key_from_seed() {
        let pk = worker_public_key_from_seed(42, 0);
        assert_eq!(pk.len(), 32);
        // Deterministic: same seed + index -> same key.
        let pk2 = worker_public_key_from_seed(42, 0);
        assert_eq!(pk, pk2);
        // Different index -> different key.
        let pk3 = worker_public_key_from_seed(42, 1);
        assert_ne!(pk, pk3);
    }

    #[test]
    fn test_sha2_hash() {
        let h1 = sha2_hash(b"hello");
        let h2 = sha2_hash(b"hello");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 32);

        let h3 = sha2_hash(b"world");
        assert_ne!(h1, h3);
    }

    #[cfg(feature = "chain")]
    #[test]
    fn test_eth_to_wei() {
        let wei = eth_to_wei(1.0);
        assert_eq!(wei, U256::from(10u64.pow(18)));

        let wei_half = eth_to_wei(0.5);
        assert_eq!(wei_half, U256::from(5u64) * U256::from(10u64.pow(17)));

        let wei_tenth = eth_to_wei(0.1);
        assert_eq!(wei_tenth, U256::from(10u64.pow(17)));
    }

    #[test]
    fn test_control_addr_offset() {
        let data: SocketAddr = "127.0.0.1:9001".parse().unwrap();
        let ctrl = WorkerRunner::control_addr(&data);
        assert_eq!(ctrl.port(), 9002);
        assert_eq!(ctrl.ip(), data.ip());
    }

    #[tokio::test]
    async fn test_control_message_tcp_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let msg = ControlMessage::SignCheckpointRequest {
            job_id: 7,
            step_number: 42,
            weight_commitment: [0xBB; 32],
            loss_scaled: 999,
        };
        let msg_clone = msg.clone();

        let sender = tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
            send_control_message(&mut stream, &msg_clone).await.unwrap();
        });

        let (mut stream, _) = listener.accept().await.unwrap();
        let received = recv_control_message(&mut stream).await.unwrap();
        sender.await.unwrap();

        match received {
            ControlMessage::SignCheckpointRequest {
                job_id,
                step_number,
                loss_scaled,
                ..
            } => {
                assert_eq!(job_id, 7);
                assert_eq!(step_number, 42);
                assert_eq!(loss_scaled, 999);
            }
            _ => panic!("Wrong control message variant"),
        }
    }

    #[tokio::test]
    async fn test_signing_service_off_chain() {
        // Tests the signing service without a chain wallet (placeholder signatures).
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let service = WorkerSigningService::new();
        let (cp_count, slash_count, steps_count) = service.counters();

        // Spawn the signing service.
        let svc_handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            service
                .run(
                    stream,
                    #[cfg(feature = "chain")]
                    None,
                )
                .await
                .unwrap();
        });

        // Connect as "owner" and send signing requests.
        let mut owner_stream = tokio::net::TcpStream::connect(addr).await.unwrap();

        // Request a checkpoint signature.
        send_control_message(
            &mut owner_stream,
            &ControlMessage::SignCheckpointRequest {
                job_id: 1,
                step_number: 10,
                weight_commitment: [0xAA; 32],
                loss_scaled: 500,
            },
        )
        .await
        .unwrap();

        let resp = recv_control_message(&mut owner_stream).await.unwrap();
        match resp {
            ControlMessage::SignCheckpointResponse { signature } => {
                assert_eq!(signature.len(), 32); // SHA-256 placeholder
            }
            _ => panic!("Expected SignCheckpointResponse"),
        }

        // Request a MAC failure signature.
        send_control_message(
            &mut owner_stream,
            &ControlMessage::SignMacFailureRequest {
                job_id: 1,
                step_number: 15,
                cheater_address: [0xCC; 20],
                evidence: vec![1, 2, 3],
            },
        )
        .await
        .unwrap();

        let resp2 = recv_control_message(&mut owner_stream).await.unwrap();
        match resp2 {
            ControlMessage::SignMacFailureResponse { signature } => {
                assert_eq!(signature.len(), 32);
            }
            _ => panic!("Expected SignMacFailureResponse"),
        }

        // Signal completion.
        send_control_message(
            &mut owner_stream,
            &ControlMessage::TrainingComplete {
                steps_completed: 100,
            },
        )
        .await
        .unwrap();

        let resp3 = recv_control_message(&mut owner_stream).await.unwrap();
        match resp3 {
            ControlMessage::TrainingCompleteAck => {}
            _ => panic!("Expected TrainingCompleteAck"),
        }

        svc_handle.await.unwrap();

        // Verify counters.
        assert_eq!(*cp_count.read().await, 1);
        assert_eq!(*slash_count.read().await, 1);
        assert_eq!(*steps_count.read().await, 100);
    }
}
