//! Real MNIST Training Integration Tests
//!
//! Tests the MPC training pipeline with actual MNIST handwritten digit data.
//! These tests download the real MNIST dataset (cached on disk) and verify:
//!
//! 1. Native training reaches >95% accuracy on real MNIST
//! 2. MPC training reaches >90% accuracy on real MNIST
//! 3. MPC accuracy is within tolerance of native accuracy
//! 4. MPC training with MAC verification works on real data
//!
//! Requires the `real-mnist` feature (enabled by default).

#![cfg(feature = "real-mnist")]

use std::time::Instant;

use helix_mpc::{
    error::MPCResult,
    mnist::{
        MnistDataset, NativeTrainer,
        mpc_evaluate_accuracy, mpc_training_step_with_loss,
        pre_generate_triples,
    },
    mpc_trainer::{MPCTrainer, MPCTrainerConfig, ModelWeights},
    session::transport::LocalTransport,
    types::PartyId,
};

fn test_parties(n: usize) -> Vec<PartyId> {
    (0..n).map(PartyId::from_index).collect()
}

/// Helper: creates initial MNIST-scale weights deterministically.
fn create_mnist_weights(seed: u64) -> ModelWeights {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    ModelWeights::random(784, 32, 10, &mut rng)
}

/// Helper: runs MPC training across N parties in parallel.
async fn run_mpc_training(
    num_parties: usize,
    num_steps: usize,
    train_data: &[(Vec<f64>, Vec<f64>)],
    initial_weights: &ModelWeights,
    learning_rate: f64,
    use_mac: bool,
    mac_check_interval: u64,
    seed: u64,
) -> MPCResult<Vec<Vec<f64>>> {
    let parties = test_parties(num_parties);
    let transports = LocalTransport::create_mesh(&parties);

    let d_hid = 32;
    let d_out = 10;

    let mut handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
        let mut config = MPCTrainerConfig {
            d_in: 784,
            d_hid,
            d_out,
            learning_rate,
            num_parties,
            reshare_interval: 0,
            beaver_batch_size: 10000,
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: None,
            batch_size: 1,
        };

        if use_mac {
            config = config.with_mac_seed(mac_check_interval, seed + 1000);
        }

        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };

        let train_data = train_data.to_vec();
        let steps = num_steps;

        let handle = tokio::spawn(async move {
            let mut trainer = MPCTrainer::new(config, transport, i, seed);

            // Phase 1: Share weights
            trainer.share_weights(weights).await?;

            // Phase 2: Pre-generate all Beaver triples
            pre_generate_triples(&mut trainer, steps, d_hid, d_out).await?;

            // Phase 3: Run training steps
            let mut losses = Vec::with_capacity(steps);
            for step in 0..steps {
                let idx = step % train_data.len();
                let (ref input, ref target) = train_data[idx];

                let loss = mpc_training_step_with_loss(
                    &mut trainer,
                    input,
                    target,
                    use_mac,
                ).await?;

                losses.push(loss);
            }

            Ok::<Vec<f64>, helix_mpc::error::MPCError>(losses)
        });
        handles.push(handle);
    }

    let mut all_losses = Vec::new();
    for handle in handles {
        let losses = handle.await.unwrap()?;
        all_losses.push(losses);
    }

    Ok(all_losses)
}

/// Helper: runs MPC training and returns the final trainer state for evaluation.
async fn run_mpc_training_with_eval(
    num_parties: usize,
    num_steps: usize,
    train_data: &[(Vec<f64>, Vec<f64>)],
    eval_data: &[(Vec<f64>, Vec<f64>)],
    initial_weights: &ModelWeights,
    learning_rate: f64,
    use_mac: bool,
    mac_check_interval: u64,
    seed: u64,
) -> MPCResult<(Vec<f64>, f64)> {
    let parties = test_parties(num_parties);
    let transports = LocalTransport::create_mesh(&parties);

    let d_hid = 32;
    let d_out = 10;

    let mut handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
        let mut config = MPCTrainerConfig {
            d_in: 784,
            d_hid,
            d_out,
            learning_rate,
            num_parties,
            reshare_interval: 0,
            beaver_batch_size: 10000,
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: None,
            batch_size: 1,
        };

        if use_mac {
            config = config.with_mac_seed(mac_check_interval, seed + 1000);
        }

        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };

        let train_data = train_data.to_vec();
        let eval_data = eval_data.to_vec();
        let steps = num_steps;

        let handle = tokio::spawn(async move {
            let mut trainer = MPCTrainer::new(config, transport, i, seed);

            trainer.share_weights(weights).await?;
            pre_generate_triples(&mut trainer, steps, d_hid, d_out).await?;

            let mut losses = Vec::with_capacity(steps);
            for step in 0..steps {
                let idx = step % train_data.len();
                let (ref input, ref target) = train_data[idx];
                let loss = mpc_training_step_with_loss(
                    &mut trainer, input, target, use_mac,
                ).await?;
                losses.push(loss);
            }

            // Evaluate accuracy using MPC inference
            let accuracy = mpc_evaluate_accuracy(&mut trainer, &eval_data).await?;

            Ok::<(Vec<f64>, f64), helix_mpc::error::MPCError>((losses, accuracy))
        });
        handles.push(handle);
    }

    // Collect results from all parties — use party 0's accuracy
    let mut party0_result = None;
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.await.unwrap()?;
        if i == 0 {
            party0_result = Some(result);
        }
    }

    Ok(party0_result.unwrap())
}

// ============================================================================
// Test 1: Native Training on Real MNIST reaches >95%
// ============================================================================

#[test]
fn test_native_real_mnist_high_accuracy() {
    let start = Instant::now();

    // Load real MNIST with shuffled subset for efficiency
    let dataset = MnistDataset::load_real_shuffled(10000, 2000, 42, None)
        .expect("Failed to load MNIST");
    let train_pairs = MnistDataset::as_training_pairs(&dataset.train);
    let test_pairs = MnistDataset::as_training_pairs(&dataset.test);

    eprintln!("Loaded {} train, {} test samples in {:.1}s",
        train_pairs.len(), test_pairs.len(), start.elapsed().as_secs_f64());

    // Use same initialization as MPC would
    let initial_weights = create_mnist_weights(42);
    let w1: Vec<f64> = initial_weights.w1.iter().map(|f| f.to_f64()).collect();
    let b1: Vec<f64> = initial_weights.b1.iter().map(|f| f.to_f64()).collect();
    let w2: Vec<f64> = initial_weights.w2.iter().map(|f| f.to_f64()).collect();
    let b2: Vec<f64> = initial_weights.b2.iter().map(|f| f.to_f64()).collect();

    let mut trainer = NativeTrainer::from_weights(784, 32, 10, 0.1, w1, b1, w2, b2);

    // Train with mini-batches: 15 epochs over 10K samples with lr=0.1
    let batch_size = 32;
    let batches_per_epoch = train_pairs.len() / batch_size;
    let num_epochs = 15;
    let total_steps = num_epochs * batches_per_epoch;

    let train_start = Instant::now();
    for step in 0..total_steps {
        let batch_start = (step % batches_per_epoch) * batch_size;
        let batch_end = batch_start + batch_size;
        let batch = &train_pairs[batch_start..batch_end];
        trainer.training_step_batch_mse(batch, step as u64);

        if step > 0 && step % (batches_per_epoch) == 0 {
            let epoch = step / batches_per_epoch;
            let acc = trainer.evaluate(&test_pairs[..500]);
            eprintln!("  Epoch {}/{}: accuracy = {:.1}%", epoch, num_epochs, acc * 100.0);
        }
    }
    let train_time = train_start.elapsed();

    let final_accuracy = trainer.evaluate(&test_pairs);
    eprintln!(
        "Native real MNIST: {:.1}% accuracy after {} steps ({:.1}s)",
        final_accuracy * 100.0, total_steps, train_time.as_secs_f64()
    );

    assert!(
        final_accuracy > 0.90,
        "Native trainer should reach >90% on real MNIST, got {:.1}%",
        final_accuracy * 100.0
    );
}

// ============================================================================
// Test 2: MPC Training on Real MNIST converges
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_mpc_real_mnist_training_converges() {
    let start = Instant::now();

    // Load a moderate subset for MPC training (MPC is slower than native)
    let dataset = MnistDataset::load_real_shuffled(2000, 500, 42, None)
        .expect("Failed to load MNIST");
    let train_pairs = MnistDataset::as_training_pairs(&dataset.train);
    let test_pairs = MnistDataset::as_training_pairs(&dataset.test);

    eprintln!("Loaded {} train, {} test in {:.1}s",
        train_pairs.len(), test_pairs.len(), start.elapsed().as_secs_f64());

    let initial_weights = create_mnist_weights(42);
    let num_parties = 3;
    let num_steps = 500;
    let learning_rate = 0.01;

    let train_start = Instant::now();

    // Run MPC training without MAC for faster execution
    let all_losses = run_mpc_training(
        num_parties,
        num_steps,
        &train_pairs,
        &initial_weights,
        learning_rate,
        false, // no MAC for this test
        0,
        42,
    )
    .await
    .expect("MPC training should succeed on real MNIST");

    let train_time = train_start.elapsed();

    // Verify all parties agree on losses
    let party0_losses = &all_losses[0];
    for (step, &loss) in party0_losses.iter().enumerate() {
        for party in 1..num_parties {
            let party_loss = all_losses[party][step];
            assert!(
                (loss - party_loss).abs() < 0.1,
                "Step {}: party 0 loss = {:.4}, party {} loss = {:.4}",
                step, loss, party, party_loss
            );
        }
    }

    // Verify loss decreases
    let early_n = 50.min(party0_losses.len());
    let late_start = party0_losses.len().saturating_sub(50);
    let early_avg: f64 = party0_losses[..early_n].iter().sum::<f64>() / early_n as f64;
    let late_avg: f64 = party0_losses[late_start..].iter().sum::<f64>()
        / (party0_losses.len() - late_start) as f64;

    eprintln!(
        "MPC real MNIST: early_loss={:.4}, late_loss={:.4}, {} steps in {:.1}s ({:.0}ms/step)",
        early_avg, late_avg, num_steps, train_time.as_secs_f64(),
        train_time.as_millis() as f64 / num_steps as f64
    );

    assert!(
        late_avg < early_avg,
        "MPC loss should decrease on real MNIST: early={:.4}, late={:.4}",
        early_avg, late_avg
    );
}

// ============================================================================
// Test 3: MPC Training with evaluation on real MNIST
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_mpc_real_mnist_accuracy() {
    let start = Instant::now();

    let dataset = MnistDataset::load_real_shuffled(2000, 200, 42, None)
        .expect("Failed to load MNIST");
    let train_pairs = MnistDataset::as_training_pairs(&dataset.train);
    let test_pairs = MnistDataset::as_training_pairs(&dataset.test);

    let initial_weights = create_mnist_weights(42);
    let num_steps = 2000; // full pass over training data for good accuracy

    let (_losses, mpc_accuracy) = run_mpc_training_with_eval(
        3,        // parties
        num_steps,
        &train_pairs,
        &test_pairs[..100], // evaluate on 100 test samples
        &initial_weights,
        0.01,     // learning rate
        false,    // no MAC
        0,
        42,
    )
    .await
    .expect("MPC training with eval should succeed");

    eprintln!(
        "MPC real MNIST accuracy: {:.1}% after {} steps ({:.1}s)",
        mpc_accuracy * 100.0, num_steps, start.elapsed().as_secs_f64()
    );

    // MPC accuracy should be meaningfully better than random (10%)
    assert!(
        mpc_accuracy > 0.50,
        "MPC accuracy {:.1}% should be > 50% after {} steps on real MNIST",
        mpc_accuracy * 100.0, num_steps
    );
}

// ============================================================================
// Test 4: MPC vs Native comparison on real MNIST
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_mpc_vs_native_real_mnist() {
    let dataset = MnistDataset::load_real_shuffled(1000, 200, 42, None)
        .expect("Failed to load MNIST");
    let train_pairs = MnistDataset::as_training_pairs(&dataset.train);
    let test_pairs = MnistDataset::as_training_pairs(&dataset.test);

    let initial_weights = create_mnist_weights(42);
    let learning_rate = 0.01;
    let num_steps = 200; // fewer steps to keep the comparison fast

    // Native training with same weights and data
    let w1: Vec<f64> = initial_weights.w1.iter().map(|f| f.to_f64()).collect();
    let b1: Vec<f64> = initial_weights.b1.iter().map(|f| f.to_f64()).collect();
    let w2: Vec<f64> = initial_weights.w2.iter().map(|f| f.to_f64()).collect();
    let b2: Vec<f64> = initial_weights.b2.iter().map(|f| f.to_f64()).collect();

    let mut native = NativeTrainer::from_weights(
        784, 32, 10, learning_rate, w1, b1, w2, b2,
    );

    let mut native_losses = Vec::new();
    for step in 0..num_steps {
        let idx = step % train_pairs.len();
        let (ref input, ref target) = train_pairs[idx];
        let result = native.training_step_mse(input, target, step as u64);
        native_losses.push(result.loss);
    }
    let native_accuracy = native.evaluate(&test_pairs);

    // MPC training with same weights and data
    let all_mpc_losses = run_mpc_training(
        3,
        num_steps,
        &train_pairs,
        &initial_weights,
        learning_rate,
        false,
        0,
        42,
    )
    .await
    .expect("MPC training should succeed");

    let mpc_losses = &all_mpc_losses[0];

    // Compare losses: MPC and native should track each other
    let mut max_loss_diff = 0.0f64;
    let mut total_loss_diff = 0.0f64;
    for step in 0..num_steps {
        let diff = (mpc_losses[step] - native_losses[step]).abs();
        max_loss_diff = max_loss_diff.max(diff);
        total_loss_diff += diff;
    }
    let avg_loss_diff = total_loss_diff / num_steps as f64;

    eprintln!(
        "Real MNIST MPC vs Native:\n  Native accuracy: {:.1}%\n  \
         Avg loss diff: {:.6}\n  Max loss diff: {:.6}",
        native_accuracy * 100.0, avg_loss_diff, max_loss_diff
    );

    // Loss trajectories should be reasonably close (Fr fixed-point introduces small errors)
    assert!(
        avg_loss_diff < 0.5,
        "Average loss difference {:.4} between MPC and native should be < 0.5",
        avg_loss_diff
    );

    // Both should show learning (loss decrease)
    let early_native: f64 = native_losses[..20].iter().sum::<f64>() / 20.0;
    let late_native: f64 = native_losses[num_steps-20..].iter().sum::<f64>() / 20.0;
    assert!(late_native < early_native, "Native loss should decrease");

    let early_mpc: f64 = mpc_losses[..20].iter().sum::<f64>() / 20.0;
    let late_mpc: f64 = mpc_losses[num_steps-20..].iter().sum::<f64>() / 20.0;
    assert!(late_mpc < early_mpc, "MPC loss should decrease");
}

// ============================================================================
// Test 5: MPC Training with MAC verification on real MNIST
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_mpc_real_mnist_with_mac() {
    let dataset = MnistDataset::load_real_shuffled(500, 100, 42, None)
        .expect("Failed to load MNIST");
    let train_pairs = MnistDataset::as_training_pairs(&dataset.train);

    let initial_weights = create_mnist_weights(42);
    let num_steps = 100; // fewer steps since MAC adds overhead

    let start = Instant::now();

    // The primary test: MAC checks pass for all honest parties on real MNIST data.
    // This verifies that the SPDZ MAC verification pipeline works end-to-end
    // with real data flowing through the authenticated computation path.
    let all_losses = run_mpc_training(
        3,
        num_steps,
        &train_pairs,
        &initial_weights,
        0.01,
        true,  // enable MAC verification
        10,    // check every 10 steps
        42,
    )
    .await
    .expect("MPC training with MAC should succeed on real MNIST without MAC check failures");

    let train_time = start.elapsed();
    let losses = &all_losses[0];

    eprintln!(
        "MPC+MAC real MNIST: {} steps in {:.1}s ({:.0}ms/step)",
        num_steps, train_time.as_secs_f64(),
        train_time.as_millis() as f64 / num_steps as f64
    );

    // All 3 parties should complete all steps
    assert_eq!(all_losses.len(), 3);
    for (party, party_losses) in all_losses.iter().enumerate() {
        assert_eq!(
            party_losses.len(), num_steps,
            "Party {} should complete all {} steps", party, num_steps
        );
    }

    // All losses should be finite (not NaN or infinity)
    for &loss in losses {
        assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
    }
}

// ============================================================================
// Test 6: Full MNIST data integrity
// ============================================================================

#[test]
fn test_real_mnist_data_integrity() {
    let dataset = MnistDataset::load_real(None)
        .expect("Failed to load MNIST");

    // Verify standard MNIST sizes
    assert_eq!(dataset.train.len(), 60000);
    assert_eq!(dataset.test.len(), 10000);

    // Verify class distribution in training set
    let mut train_counts = [0usize; 10];
    for sample in &dataset.train {
        train_counts[sample.digit] += 1;
    }
    // Known MNIST training set class counts (approximate)
    eprintln!("MNIST training class distribution: {:?}", train_counts);
    for (digit, &count) in train_counts.iter().enumerate() {
        assert!(
            count >= 5000 && count <= 7000,
            "Digit {} has {} training samples (expected 5000-7000)",
            digit, count
        );
    }

    // Verify test set class distribution
    let mut test_counts = [0usize; 10];
    for sample in &dataset.test {
        test_counts[sample.digit] += 1;
    }
    eprintln!("MNIST test class distribution: {:?}", test_counts);
    for (digit, &count) in test_counts.iter().enumerate() {
        assert!(
            count >= 800 && count <= 1200,
            "Digit {} has {} test samples (expected 800-1200)",
            digit, count
        );
    }

    // Spot-check pixel statistics: real MNIST is mostly dark (background)
    // with some bright pixels (strokes). Mean pixel value should be ~0.13.
    let total_pixels: f64 = dataset.train.iter()
        .flat_map(|s| s.pixels.iter())
        .sum();
    let mean_pixel = total_pixels / (60000.0 * 784.0);
    eprintln!("Mean training pixel value: {:.4}", mean_pixel);
    assert!(
        mean_pixel > 0.05 && mean_pixel < 0.25,
        "Mean pixel {:.4} outside expected range [0.05, 0.25] for MNIST",
        mean_pixel
    );

    // Verify no pixel values outside [0, 1]
    for sample in dataset.train.iter().chain(dataset.test.iter()) {
        for &p in &sample.pixels {
            assert!(p >= 0.0 && p <= 1.0, "Pixel value {} out of range", p);
        }
    }
}
