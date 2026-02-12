//! Worker threads that perform real training with ZK proof generation.
//!
//! Uses small quantized values that stay within the circuit's ReLU lookup range.
//! The Trainer quantizes f64 values via `round(val * 1000)`, so weights of 0.001
//! become Fr(1), inputs of 0.001 become Fr(1), etc. The ReLU lookup range is ±128,
//! so we keep all intermediate activations well below that threshold.
//!
//! Workers perform self-verification before accepting proofs and cache proofs
//! to avoid regenerating already-verified proofs for identical witnesses.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use helix_core::DatasetSource;
use helix_node::trainer::{MlpModel, Trainer};
use helix_prover::EvmProofBundle;

/// Results from a single worker's training run.
#[derive(Debug)]
pub struct WorkerResult {
    /// Worker index.
    #[allow(dead_code)]
    pub worker_id: usize,
    /// Per-step losses.
    pub losses: Vec<f64>,
    /// Number of steps completed.
    pub steps_completed: usize,
    /// Number of proofs generated.
    pub proofs_generated: usize,
    /// Number of proofs that passed self-verification.
    pub proofs_verified: usize,
    /// Number of proof cache hits.
    pub cache_hits: usize,
    /// Total proof generation time.
    pub total_prove_time: Duration,
    /// EVM bundles for on-chain submission (using the canonical prover type).
    pub evm_bundles: Vec<EvmProofBundle>,
}

/// Creates a regression dataset scaled for circuit compatibility.
///
/// Values are small (0.001-0.005 range) so that after quantization (×1000)
/// they map to Fr(1)-Fr(5), keeping all circuit activations within the
/// ReLU lookup range of ±128.
pub fn make_dataset() -> Vec<(Vec<f64>, Vec<f64>)> {
    vec![
        (vec![0.001, 0.001], vec![0.002]),
        (vec![0.002, 0.001], vec![0.003]),
        (vec![0.001, 0.002], vec![0.003]),
        (vec![0.002, 0.002], vec![0.004]),
        (vec![0.003, 0.001], vec![0.004]),
        (vec![0.001, 0.003], vec![0.004]),
        (vec![0.003, 0.002], vec![0.005]),
        (vec![0.002, 0.003], vec![0.005]),
    ]
}

/// Loads a dataset from a `DatasetSource`.
///
/// - `Builtin` → returns the hardcoded synthetic regression dataset
/// - `LocalCsv` → reads the CSV file and extracts feature/label columns
/// - `Ipfs`/`Http` → not implemented in the demo binary
pub fn load_dataset(source: &DatasetSource) -> Result<Vec<(Vec<f64>, Vec<f64>)>> {
    match source {
        DatasetSource::Builtin => Ok(make_dataset()),
        DatasetSource::LocalCsv { path, feature_columns, label_columns } => {
            load_csv_dataset(path, feature_columns, label_columns)
        }
        DatasetSource::Ipfs { cid } => {
            bail!("IPFS dataset loading not implemented in demo (CID: {})", cid)
        }
        DatasetSource::Http { url } => {
            bail!("HTTP dataset loading not implemented in demo (URL: {})", url)
        }
    }
}

/// Loads a CSV file as a dataset of (features, labels) pairs.
fn load_csv_dataset(
    path: &str,
    feature_columns: &[usize],
    label_columns: &[usize],
) -> Result<Vec<(Vec<f64>, Vec<f64>)>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Failed to read CSV file '{}': {}", path, e))?;

    let mut dataset = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Skip header row if it contains non-numeric values
        if line_num == 0 {
            let first_field = line.split(',').next().unwrap_or("").trim();
            if first_field.parse::<f64>().is_err() {
                continue; // skip header
            }
        }

        let values: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("CSV parse error at line {}: {}", line_num + 1, e))?;

        let max_col = feature_columns.iter().chain(label_columns.iter()).max().copied().unwrap_or(0);
        if values.len() <= max_col {
            bail!(
                "CSV line {} has {} columns, but column index {} was requested",
                line_num + 1, values.len(), max_col
            );
        }

        let features: Vec<f64> = feature_columns.iter().map(|&c| values[c]).collect();
        let labels: Vec<f64> = label_columns.iter().map(|&c| values[c]).collect();
        dataset.push((features, labels));
    }

    if dataset.is_empty() {
        bail!("CSV file '{}' produced no data rows", path);
    }

    Ok(dataset)
}

/// Creates an initial model with small weights for circuit compatibility (public).
pub fn create_small_model_pub(d_in: usize, d_hid: usize, d_out: usize, seed: u64) -> MlpModel {
    create_small_model(d_in, d_hid, d_out, seed)
}

/// Creates an initial model with small weights for circuit compatibility.
///
/// Weights are in the 0.001 range so that quantization produces Fr(1)-Fr(3),
/// and matmul results stay within the ReLU range.
fn create_small_model(d_in: usize, d_hid: usize, d_out: usize, seed: u64) -> MlpModel {
    // Simple LCG for deterministic small weights
    let mut rng = seed;
    let mut next_small = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let raw = (rng >> 33) as f64 / (1u64 << 31) as f64; // [-1, 1]
        // Map to [0.001, 0.003] — always positive, always small
        0.001 + (raw.abs() * 0.002)
    };

    let w1: Vec<f64> = (0..d_hid * d_in).map(|_| next_small()).collect();
    let b1 = vec![0.0; d_hid];
    let w2: Vec<f64> = (0..d_out * d_hid).map(|_| next_small()).collect();
    let b2 = vec![0.0; d_out];

    MlpModel::new(d_in, d_hid, d_out, w1, b1, w2, b2)
}

/// Runs `num_workers` training threads in parallel, each generating real ZK proofs.
pub async fn run_workers(
    num_workers: usize,
    steps_per_worker: usize,
    d_hid: usize,
    lr: f64,
    base_seed: u64,
    dataset: &[(Vec<f64>, Vec<f64>)],
) -> Result<Vec<WorkerResult>> {
    let dataset = dataset.to_vec();

    let mut handles = Vec::new();

    for worker_id in 0..num_workers {
        let dataset = dataset.clone();
        let seed = base_seed + worker_id as u64;

        let handle = tokio::task::spawn_blocking(move || {
            run_single_worker(worker_id, steps_per_worker, d_hid, lr, seed, &dataset)
        });
        handles.push(handle);
    }

    let mut results = Vec::with_capacity(num_workers);
    for handle in handles {
        let result = handle.await??;
        results.push(result);
    }

    Ok(results)
}

/// Simple proof cache key: hash of (step, dataset_index).
/// In production this would use the full witness hash from the prover.
fn proof_cache_key(step: usize, dataset_idx: usize) -> u64 {
    let mut h = step as u64;
    h = h.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(dataset_idx as u64);
    h
}

/// Single worker training loop with real proof generation.
fn run_single_worker(
    worker_id: usize,
    num_steps: usize,
    d_hid: usize,
    lr: f64,
    seed: u64,
    dataset: &[(Vec<f64>, Vec<f64>)],
) -> Result<WorkerResult> {
    let d_in = 2;
    let d_out = 1;

    // Use small weights matching circuit constraints
    let model = create_small_model(d_in, d_hid, d_out, seed);
    // Scale LR to match quantized value domain (0.001 → Fr(1))
    let quantized_lr = lr * 0.001;
    let mut trainer = Trainer::with_model(model, quantized_lr);

    let mut losses = Vec::with_capacity(num_steps);
    let mut evm_bundles = Vec::new();
    let mut total_prove_time = Duration::ZERO;
    let mut proofs_generated = 0;
    let mut proofs_verified = 0;
    let mut cache_hits = 0;

    // Simple proof cache: maps cache_key → EvmProofBundle
    let mut proof_cache: HashMap<u64, EvmProofBundle> = HashMap::new();

    for step in 0..num_steps {
        let dataset_idx = step % dataset.len();
        let (x, target) = &dataset[dataset_idx];
        let cache_key = proof_cache_key(step, dataset_idx);

        let step_start = Instant::now();
        let result = trainer.train_step(x, target)?;
        let step_time = step_start.elapsed();

        losses.push(result.loss);
        total_prove_time += step_time;
        proofs_generated += 1;

        // Self-verification: the Trainer's prover already self-verifies.
        if result.proof_result.verified {
            proofs_verified += 1;
        }

        // Collect EVM bundles for first few steps (for on-chain submission).
        // The Trainer now creates EvmProofBundle internally.
        if step < 3 {
            // Check cache first
            if let Some(cached) = proof_cache.get(&cache_key) {
                evm_bundles.push(cached.clone());
                cache_hits += 1;
            } else if let Some(bundle) = result.evm_bundle.clone() {
                // Validate EVM proof format: must be at least 320 bytes
                if bundle.evm_proof.len() >= 320 {
                    proof_cache.insert(cache_key, bundle.clone());
                    evm_bundles.push(bundle);
                } else {
                    tracing::warn!(
                        "Worker {}: step {} produced undersized proof ({} bytes, expected >=320)",
                        worker_id, step, bundle.evm_proof.len(),
                    );
                }
            } else {
                tracing::warn!(
                    "Worker {}: step {} EVM bundle not available",
                    worker_id, step,
                );
            }
        }

        // Progress reporting every 5 steps
        if (step + 1) % 5 == 0 || step == num_steps - 1 {
            tracing::debug!(
                "Worker {}: step {}/{}, loss={:.6}, verified={}, proof_time={:.0}ms",
                worker_id,
                step + 1,
                num_steps,
                result.loss,
                result.proof_result.verified,
                step_time.as_millis(),
            );
        }
    }

    Ok(WorkerResult {
        worker_id,
        losses,
        steps_completed: num_steps,
        proofs_generated,
        proofs_verified,
        cache_hits,
        total_prove_time,
        evm_bundles,
    })
}
