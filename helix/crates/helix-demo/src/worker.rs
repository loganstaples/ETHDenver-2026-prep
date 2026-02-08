//! Worker threads that perform real training with ZK proof generation.
//!
//! Uses small quantized values that stay within the circuit's ReLU lookup range.
//! The Trainer quantizes f64 values via `round(val * 1000)`, so weights of 0.001
//! become Fr(1), inputs of 0.001 become Fr(1), etc. The ReLU lookup range is ±128,
//! so we keep all intermediate activations well below that threshold.

use std::time::{Duration, Instant};

use anyhow::Result;
use helix_node::trainer::{MlpModel, Trainer};

/// EVM-ready proof bundle from a worker.
#[derive(Debug, Clone)]
pub struct EvmBundle {
    /// 320-byte EVM-formatted proof.
    pub proof_bytes: Vec<u8>,
    /// 8 x 32-byte big-endian public inputs.
    pub public_inputs_u256: Vec<[u8; 32]>,
    /// Step number.
    #[allow(dead_code)]
    pub step: u64,
    /// Loss at this step.
    #[allow(dead_code)]
    pub loss: f64,
}

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
    /// Total proof generation time.
    pub total_prove_time: Duration,
    /// EVM bundles for on-chain submission.
    pub evm_bundles: Vec<EvmBundle>,
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

    for step in 0..num_steps {
        let (x, target) = &dataset[step % dataset.len()];

        let step_start = Instant::now();
        let result = trainer.train_step(x, target)?;
        let step_time = step_start.elapsed();

        losses.push(result.loss);
        total_prove_time += step_time;
        proofs_generated += 1;

        // Collect EVM bundles for first few steps (for on-chain submission)
        if step < 3 {
            if let Some(ref evm_proof) = result.evm_proof {
                evm_bundles.push(EvmBundle {
                    proof_bytes: evm_proof.clone(),
                    public_inputs_u256: result.evm_public_inputs.clone(),
                    step: result.step,
                    loss: result.loss,
                });
            }
        }

        // Progress reporting every 5 steps
        if (step + 1) % 5 == 0 || step == num_steps - 1 {
            tracing::debug!(
                "Worker {}: step {}/{}, loss={:.6}, proof_time={:.0}ms",
                worker_id,
                step + 1,
                num_steps,
                result.loss,
                step_time.as_millis(),
            );
        }
    }

    Ok(WorkerResult {
        worker_id,
        losses,
        steps_completed: num_steps,
        proofs_generated,
        total_prove_time,
        evm_bundles,
    })
}
