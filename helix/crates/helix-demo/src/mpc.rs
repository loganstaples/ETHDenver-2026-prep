//! MPC (Multi-Party Computation) demo mode.
//!
//! Distributes model weights as additive shares across parties, trains
//! on shares using `MPCTrainer` from `helix-mpc`, then reconstructs
//! final weights and compares with non-MPC training.

use std::time::Instant;

use anyhow::{Context, Result};
use helix_mpc::mac_verification::MACVerificationConfig;
use helix_mpc::mpc_trainer::{MPCTrainer, MPCTrainerConfig, ModelWeights};
use helix_mpc::session::transport::LocalTransport;
use helix_mpc::types::PartyId;
use helix_mpc::Fr;

/// Results from an MPC training run.
#[derive(Debug)]
pub struct MpcResult {
    /// Per-step losses from party 0 (all parties compute the same loss).
    pub losses: Vec<f64>,
}

/// Reconstructed plaintext weights after MPC training.
#[derive(Debug)]
pub struct ReconstructedWeights {
    pub w1: Vec<f64>,
    #[allow(dead_code)]
    pub b1: Vec<f64>,
    #[allow(dead_code)]
    pub w2: Vec<f64>,
    #[allow(dead_code)]
    pub b2: Vec<f64>,
}

/// Runs the MPC training demo.
///
/// Creates `num_parties` MPC trainers connected via in-process `LocalTransport`,
/// distributes initial model weights as additive shares, trains on the dataset,
/// and reports results.
pub async fn run_mpc_training(
    num_parties: usize,
    d_hid: usize,
    lr: f64,
    seed: u64,
    dataset: &[(Vec<f64>, Vec<f64>)],
    steps: usize,
    generate_proofs: bool,
) -> Result<MpcResult> {
    let d_in = 2;
    let d_out = 1;

    crate::display::info(&format!(
        "Setting up {} MPC parties with LocalTransport...",
        num_parties,
    ));

    // Create party IDs and transport mesh
    let party_ids: Vec<PartyId> = (0..num_parties)
        .map(|i| PartyId::from_index(i))
        .collect();
    let transports = LocalTransport::create_mesh(&party_ids);

    // MPC trainer config
    let config = MPCTrainerConfig {
        d_in,
        d_hid,
        d_out,
        learning_rate: lr * 0.001, // Scale to match quantized domain
        num_parties,
        reshare_interval: 0, // Disable re-sharing for short demo
        beaver_batch_size: 512,
        generate_proofs,
        base_error: 1e-6,
        checkpoint_interval: 1,
        mac_config: Some(MACVerificationConfig {
            check_interval: 1,
            enable_cheater_identification: true,
            mac_seed: seed + 1000,
        }),
    };

    // Create initial weights (same as non-MPC for comparison)
    let initial_weights = create_mpc_initial_weights(d_in, d_hid, d_out, seed);

    // Truncate dataset to requested steps
    let train_data: Vec<(Vec<f64>, Vec<f64>)> = dataset
        .iter()
        .cycle()
        .take(steps)
        .cloned()
        .collect();

    let start = Instant::now();

    // Spawn one task per party — they communicate via LocalTransport channels
    let mut handles = Vec::with_capacity(num_parties);
    for (party_idx, transport) in transports.into_iter().enumerate() {
        let config = config.clone();
        let train_data = train_data.clone();
        let weights = if party_idx == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };

        let handle = tokio::spawn(async move {
            let mut trainer = MPCTrainer::new(config, transport, party_idx, seed);

            // Phase 1: Share weights
            trainer.share_weights(weights).await?;

            // Phase 2: Generate Beaver triples
            trainer.generate_beaver_triples(512).await?;

            // Phase 3: Train
            let results = trainer.train(&train_data).await?;

            // Collect losses and final weight shares
            let losses: Vec<f64> = results.iter().map(|r| r.loss).collect();
            let (w1, b1, w2, b2) = trainer.weight_shares();

            Ok::<_, helix_mpc::MPCError>((
                party_idx,
                losses,
                w1.to_vec(),
                b1.to_vec(),
                w2.to_vec(),
                b2.to_vec(),
            ))
        });
        handles.push(handle);
    }

    // Collect results from all parties
    let mut all_results = Vec::with_capacity(num_parties);
    for handle in handles {
        let result = handle
            .await
            .context("MPC party task panicked")?
            .context("MPC training failed")?;
        all_results.push(result);
    }

    let total_time = start.elapsed();

    // Sort by party index
    all_results.sort_by_key(|r| r.0);

    // Use party 0's losses as the canonical loss trajectory
    let losses = all_results[0].1.clone();

    // Reconstruct weights by summing shares across all parties
    let _final_weights = reconstruct_weights(&all_results);

    crate::display::success(&format!(
        "MPC training complete: {} steps across {} parties in {:.1}s",
        steps,
        num_parties,
        total_time.as_secs_f64(),
    ));

    if let Some(ref weights) = _final_weights {
        let w1_norm: f64 = weights.w1.iter().map(|w| w * w).sum::<f64>().sqrt();
        crate::display::metric(&format!(
            "Reconstructed weight norm: ||W1|| = {:.6}",
            w1_norm,
        ));
    }

    Ok(MpcResult {
        losses,
    })
}

/// Creates initial model weights in MPC field format from the same seed
/// used by the non-MPC demo, for comparison.
fn create_mpc_initial_weights(
    d_in: usize,
    d_hid: usize,
    d_out: usize,
    seed: u64,
) -> ModelWeights {
    let mut rng = seed;
    let mut next_small = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let raw = (rng >> 33) as f64 / (1u64 << 31) as f64;
        0.001 + (raw.abs() * 0.002)
    };

    let w1: Vec<f64> = (0..d_hid * d_in).map(|_| next_small()).collect();
    let b1 = vec![0.0; d_hid];
    let w2: Vec<f64> = (0..d_out * d_hid).map(|_| next_small()).collect();
    let b2 = vec![0.0; d_out];

    ModelWeights::from_f64(&w1, &b1, &w2, &b2)
}

/// Reconstructs plaintext weights from party shares by summing.
fn reconstruct_weights(
    all_results: &[(usize, Vec<f64>, Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>)],
) -> Option<ReconstructedWeights> {
    if all_results.is_empty() {
        return None;
    }

    let w1_len = all_results[0].2.len();
    let b1_len = all_results[0].3.len();
    let w2_len = all_results[0].4.len();
    let b2_len = all_results[0].5.len();

    // Sum shares across parties
    let mut w1_sum = vec![Fr::ZERO; w1_len];
    let mut b1_sum = vec![Fr::ZERO; b1_len];
    let mut w2_sum = vec![Fr::ZERO; w2_len];
    let mut b2_sum = vec![Fr::ZERO; b2_len];

    for (_, _, w1, b1, w2, b2) in all_results {
        for (i, s) in w1.iter().enumerate() {
            w1_sum[i] = Fr::add(&w1_sum[i], s);
        }
        for (i, s) in b1.iter().enumerate() {
            b1_sum[i] = Fr::add(&b1_sum[i], s);
        }
        for (i, s) in w2.iter().enumerate() {
            w2_sum[i] = Fr::add(&w2_sum[i], s);
        }
        for (i, s) in b2.iter().enumerate() {
            b2_sum[i] = Fr::add(&b2_sum[i], s);
        }
    }

    // Convert back to f64
    Some(ReconstructedWeights {
        w1: w1_sum.iter().map(|f| f.to_f64()).collect(),
        b1: b1_sum.iter().map(|f| f.to_f64()).collect(),
        w2: w2_sum.iter().map(|f| f.to_f64()).collect(),
        b2: b2_sum.iter().map(|f| f.to_f64()).collect(),
    })
}
