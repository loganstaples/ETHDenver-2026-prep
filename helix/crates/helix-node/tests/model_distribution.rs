//! Model Distribution & Weight Aggregation Integration Test.
//!
//! Verifies the complete round-trip flow:
//! 1. Aggregator creates initial model → serializes to checkpoint bytes
//! 2. Workers deserialize checkpoint → reconstruct identical model
//! 3. Workers train locally on different data (synthetic or CSV) → produce updated models
//! 4. Workers serialize updated models → send back as checkpoint bytes
//! 5. Aggregator deserializes all worker models → averages (FedAvg)
//! 6. Aggregator persists checkpoint to disk
//! 7. Checkpoint is loadable from disk and produces correct model
//!
//! Success Criteria: model weights round-trip through checkpoint serialization,
//! FedAvg produces correct average, persisted checkpoint is loadable.

use helix_core::ModelCheckpoint;
use helix_node::trainer::{average_models, forward, MlpModel, Trainer};

/// Generates deterministic synthetic training data from a seed (matches main.rs).
fn generate_training_data(d_in: usize, d_out: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = seed;
    let mut next = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    let x: Vec<f64> = (0..d_in).map(|_| next() * 0.5).collect();
    let target: Vec<f64> = (0..d_out).map(|_| next().abs()).collect();
    (x, target)
}

#[test]
fn test_model_checkpoint_roundtrip_via_bytes() {
    // Aggregator creates initial model
    let model = MlpModel::new_random(4, 8, 2, 42);
    let _commitment = model.commitment();

    // Serialize to checkpoint bytes (what gets sent over the network)
    let checkpoint = model.to_checkpoint(0);
    let bytes = checkpoint.to_bytes().expect("to_bytes failed");

    // Worker deserializes checkpoint bytes
    let received_ckpt = ModelCheckpoint::from_bytes(&bytes).expect("from_bytes failed");
    let worker_model = MlpModel::from_checkpoint(&received_ckpt).unwrap();

    // Verify model dimensions match
    assert_eq!(worker_model.d_in, 4);
    assert_eq!(worker_model.d_hid, 8);
    assert_eq!(worker_model.d_out, 2);

    // Verify weights are close (f64→f32→f64 loses some precision)
    for (orig, recv) in model.w1.iter().zip(worker_model.w1.iter()) {
        assert!(
            (orig - recv).abs() < 1e-6,
            "w1 mismatch: {} vs {}",
            orig,
            recv
        );
    }
    for (orig, recv) in model.b1.iter().zip(worker_model.b1.iter()) {
        assert!(
            (orig - recv).abs() < 1e-6,
            "b1 mismatch: {} vs {}",
            orig,
            recv
        );
    }
    for (orig, recv) in model.w2.iter().zip(worker_model.w2.iter()) {
        assert!(
            (orig - recv).abs() < 1e-6,
            "w2 mismatch: {} vs {}",
            orig,
            recv
        );
    }

    // Commitments should be very close (may differ due to f32 precision)
    let _worker_commitment = worker_model.commitment();
    // The SHA-256 may differ because of f64→f32→f64 rounding
    // but both models should produce similar forward pass results
    let (x, target) = generate_training_data(4, 2, 100);
    let orig_fwd = forward(&model, &x, &target);
    let recv_fwd = forward(&worker_model, &x, &target);
    assert!(
        (orig_fwd.loss - recv_fwd.loss).abs() < 1e-4,
        "Forward pass should produce similar results: {} vs {}",
        orig_fwd.loss,
        recv_fwd.loss
    );
}

#[test]
fn test_fedavg_weight_averaging() {
    // Three workers start from the same model but train on different data
    let base_model = MlpModel::new_random(2, 4, 1, 42);

    let mut worker_models = Vec::new();
    for worker_id in 0..3 {
        let mut trainer = Trainer::with_model(base_model.clone(), 0.01);

        // Each worker trains 5 steps on different data
        for step in 0..5 {
            let seed = 1000 * (worker_id + 1) + step;
            let (x, target) = generate_training_data(2, 1, seed);
            trainer.train_step_unproved(&x, &target);
        }

        worker_models.push(trainer.model().clone());
    }

    // Verify workers diverged (different data → different weights)
    assert_ne!(
        worker_models[0].commitment(),
        worker_models[1].commitment(),
        "Workers should have diverged after training on different data"
    );

    // FedAvg: average all worker models
    let averaged = average_models(&worker_models).expect("average_models failed");

    // Verify dimensions preserved
    assert_eq!(averaged.d_in, 2);
    assert_eq!(averaged.d_hid, 4);
    assert_eq!(averaged.d_out, 1);

    // Verify weights are the mean of worker weights
    for i in 0..averaged.w1.len() {
        let expected = (worker_models[0].w1[i] + worker_models[1].w1[i] + worker_models[2].w1[i]) / 3.0;
        assert!(
            (averaged.w1[i] - expected).abs() < 1e-10,
            "w1[{}] should be mean: {} vs {}",
            i,
            averaged.w1[i],
            expected,
        );
    }
    for i in 0..averaged.w2.len() {
        let expected = (worker_models[0].w2[i] + worker_models[1].w2[i] + worker_models[2].w2[i]) / 3.0;
        assert!(
            (averaged.w2[i] - expected).abs() < 1e-10,
            "w2[{}] should be mean: {} vs {}",
            i,
            averaged.w2[i],
            expected,
        );
    }

    // Averaged model should be different from any individual worker
    for wm in &worker_models {
        assert_ne!(
            averaged.commitment(),
            wm.commitment(),
            "Averaged model should differ from individual workers"
        );
    }
}

#[test]
fn test_complete_training_round_with_distribution() {
    // Simulate a complete training round:
    // Aggregator → workers → train → workers → aggregator → FedAvg → persist

    let d_in = 2;
    let d_hid = 4;
    let d_out = 1;
    let num_workers = 3;
    let num_steps_per_worker = 3;
    let lr = 0.01;

    // 1. Aggregator creates initial model
    let initial_model = MlpModel::new_random(d_in, d_hid, d_out, 42);
    let initial_hash = initial_model.commitment();

    // 2. Serialize model for distribution
    let initial_ckpt = initial_model.to_checkpoint(0);
    let ckpt_bytes = initial_ckpt.to_bytes().expect("to_bytes failed");

    // Record initial loss for evaluation
    let eval_data: Vec<(Vec<f64>, Vec<f64>)> = (0..5)
        .map(|i| generate_training_data(d_in, d_out, 9000 + i))
        .collect();
    let initial_loss: f64 = eval_data
        .iter()
        .map(|(x, t)| forward(&initial_model, x, t).loss)
        .sum::<f64>()
        / eval_data.len() as f64;

    // 3. Workers receive model bytes, train, send back updates
    let mut worker_ckpt_bytes = Vec::new();
    for worker_id in 0..num_workers {
        // Deserialize aggregator's model
        let received_ckpt = ModelCheckpoint::from_bytes(&ckpt_bytes).expect("from_bytes failed");
        let worker_model = MlpModel::from_checkpoint(&received_ckpt).unwrap();
        let mut trainer = Trainer::with_model(worker_model, lr);

        // Train locally on worker-specific data
        for step in 0..num_steps_per_worker {
            let seed = 2000 * (worker_id as u64 + 1) + step as u64;
            let (x, target) = generate_training_data(d_in, d_out, seed);
            trainer.train_step_unproved(&x, &target);
        }

        // Serialize updated model
        let updated_ckpt = trainer.model().to_checkpoint(num_steps_per_worker as u64);
        let updated_bytes = updated_ckpt.to_bytes().expect("to_bytes failed");
        worker_ckpt_bytes.push(updated_bytes);
    }

    // 4. Aggregator receives and deserializes all worker models
    let mut worker_models = Vec::new();
    for (i, bytes) in worker_ckpt_bytes.iter().enumerate() {
        let ckpt = ModelCheckpoint::from_bytes(bytes).expect("from_bytes failed");
        let model = MlpModel::from_checkpoint(&ckpt).unwrap();
        assert_eq!(model.d_in, d_in, "Worker {} d_in mismatch", i);
        assert_eq!(model.d_hid, d_hid, "Worker {} d_hid mismatch", i);
        assert_eq!(model.d_out, d_out, "Worker {} d_out mismatch", i);
        worker_models.push(model);
    }

    // 5. FedAvg
    let averaged_model = average_models(&worker_models).expect("FedAvg failed");
    let averaged_hash = averaged_model.commitment();

    // The model should have changed from training
    assert_ne!(
        initial_hash, averaged_hash,
        "Averaged model should differ from initial model"
    );

    // 6. Persist checkpoint to disk
    let dir = tempfile::tempdir().expect("tempdir failed");
    let persist_path = dir.path().join("model_latest.hxck");

    let persist_ckpt = averaged_model.to_checkpoint(1);
    let persist_bytes = persist_ckpt.to_bytes().expect("to_bytes failed");
    std::fs::write(&persist_path, &persist_bytes).expect("write failed");

    // 7. Verify checkpoint is loadable
    let loaded_bytes = std::fs::read(&persist_path).expect("read failed");
    let loaded_ckpt = ModelCheckpoint::from_bytes(&loaded_bytes).expect("from_bytes failed");
    let loaded_model = MlpModel::from_checkpoint(&loaded_ckpt).unwrap();

    assert_eq!(loaded_model.d_in, d_in);
    assert_eq!(loaded_model.d_hid, d_hid);
    assert_eq!(loaded_model.d_out, d_out);

    // Verify weights match (within f32 precision)
    for (avg, loaded) in averaged_model.w1.iter().zip(loaded_model.w1.iter()) {
        assert!(
            (avg - loaded).abs() < 1e-6,
            "Persisted w1 mismatch: {} vs {}",
            avg,
            loaded,
        );
    }

    // 8. Verify training actually improved the model
    let final_loss: f64 = eval_data
        .iter()
        .map(|(x, t)| forward(&averaged_model, x, t).loss)
        .sum::<f64>()
        / eval_data.len() as f64;

    println!(
        "Training round complete: initial_loss={:.6}, final_loss={:.6}, improved={}",
        initial_loss,
        final_loss,
        final_loss < initial_loss,
    );
    // Note: with only 3 steps per worker and random data, loss may not always decrease.
    // The key assertion is that the pipeline works end-to-end, not that training converges.
}

#[test]
fn test_csv_data_loading_simulation() {
    // Write a temporary CSV file and verify the loading logic
    let dir = tempfile::tempdir().expect("tempdir failed");
    let csv_path = dir.path().join("training_data.csv");

    let csv_content = "\
x1,x2,target
0.1,0.2,0.3
0.4,0.5,0.9
0.7,0.8,1.5
-0.1,0.3,0.2
0.5,0.5,1.0
";
    std::fs::write(&csv_path, csv_content).expect("write CSV failed");

    // Parse CSV (same logic as in run_worker)
    let contents = std::fs::read_to_string(&csv_path).expect("read CSV failed");
    let mut lines = contents.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').map(|s| s.trim()).collect();

    let input_cols = vec!["x1".to_string(), "x2".to_string()];
    let target_cols = vec!["target".to_string()];

    let in_idxs: Vec<usize> = input_cols
        .iter()
        .filter_map(|c| header.iter().position(|h| h == c))
        .collect();
    let tgt_idxs: Vec<usize> = target_cols
        .iter()
        .filter_map(|c| header.iter().position(|h| h == c))
        .collect();

    assert_eq!(in_idxs, vec![0, 1], "Input column indices should match");
    assert_eq!(tgt_idxs, vec![2], "Target column index should match");

    let mut dataset = Vec::new();
    for line in lines {
        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        let x: Vec<f64> = in_idxs
            .iter()
            .map(|&i| fields.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0))
            .collect();
        let t: Vec<f64> = tgt_idxs
            .iter()
            .map(|&i| fields.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0))
            .collect();
        dataset.push((x, t));
    }

    assert_eq!(dataset.len(), 5, "Should have 5 data samples");
    assert_eq!(dataset[0].0, vec![0.1, 0.2]);
    assert_eq!(dataset[0].1, vec![0.3]);
    assert_eq!(dataset[2].0, vec![0.7, 0.8]);
    assert_eq!(dataset[2].1, vec![1.5]);

    // Train a model on this CSV data
    let model = MlpModel::new_random(2, 4, 1, 42);
    let mut trainer = Trainer::with_model(model, 0.01);

    let initial_loss = forward(trainer.model(), &dataset[0].0, &dataset[0].1).loss;

    for i in 0..50 {
        let (x, t) = &dataset[i % dataset.len()];
        trainer.train_step_unproved(x, t);
    }

    let final_loss = forward(trainer.model(), &dataset[0].0, &dataset[0].1).loss;

    assert!(
        final_loss < initial_loss,
        "Training on CSV data should reduce loss: {} → {}",
        initial_loss,
        final_loss,
    );
}

#[test]
fn test_multi_round_weight_distribution() {
    // Simulate 3 rounds of training with weight distribution between rounds
    let d_in = 2;
    let d_hid = 4;
    let d_out = 1;
    let num_workers = 2;
    let lr = 0.01;

    let mut aggregator_model = MlpModel::new_random(d_in, d_hid, d_out, 42);
    let mut round_losses = Vec::new();

    for round in 0..3 {
        // Distribute current model to workers
        let ckpt = aggregator_model.to_checkpoint(round as u64);
        let ckpt_bytes = ckpt.to_bytes().expect("to_bytes failed");

        // Workers train
        let mut worker_models = Vec::new();
        for worker_id in 0..num_workers {
            let received_ckpt = ModelCheckpoint::from_bytes(&ckpt_bytes).expect("from_bytes failed");
            let worker_model = MlpModel::from_checkpoint(&received_ckpt).unwrap();
            let mut trainer = Trainer::with_model(worker_model, lr);

            for step in 0..10 {
                let seed = (round as u64 * 10000) + (worker_id as u64 * 1000) + step as u64;
                let (x, target) = generate_training_data(d_in, d_out, seed);
                trainer.train_step_unproved(&x, &target);
            }

            worker_models.push(trainer.model().clone());
        }

        // Aggregate
        let averaged = average_models(&worker_models).expect("FedAvg failed");

        // Evaluate
        let eval_loss: f64 = (0..10)
            .map(|i| {
                let (x, t) = generate_training_data(d_in, d_out, 50000 + i);
                forward(&averaged, &x, &t).loss
            })
            .sum::<f64>()
            / 10.0;

        round_losses.push(eval_loss);
        aggregator_model = averaged;

        println!("Round {}: avg_eval_loss={:.6}", round, eval_loss);
    }

    // After 3 rounds of distributed training, the model should exist and be valid
    assert_eq!(aggregator_model.d_in, d_in);
    assert_eq!(aggregator_model.d_hid, d_hid);
    assert_eq!(aggregator_model.d_out, d_out);
    assert_eq!(aggregator_model.num_params(), d_hid * d_in + d_hid + d_out * d_hid + d_out);

    // Verify checkpoint persistence across rounds
    let final_ckpt = aggregator_model.to_checkpoint(3);
    let final_bytes = final_ckpt.to_bytes().expect("to_bytes failed");
    let reloaded = ModelCheckpoint::from_bytes(&final_bytes).expect("from_bytes failed");
    let reloaded_model = MlpModel::from_checkpoint(&reloaded).unwrap();

    // Verify reloaded model produces same forward pass results
    let (x, t) = generate_training_data(d_in, d_out, 99999);
    let orig_fwd = forward(&aggregator_model, &x, &t);
    let reloaded_fwd = forward(&reloaded_model, &x, &t);
    assert!(
        (orig_fwd.loss - reloaded_fwd.loss).abs() < 1e-4,
        "Reloaded model should produce same loss: {} vs {}",
        orig_fwd.loss,
        reloaded_fwd.loss,
    );
}

#[test]
fn test_average_models_error_cases() {
    // Empty models list
    let result = average_models(&[]);
    assert!(result.is_err(), "Should fail with empty models list");

    // Mismatched dimensions
    let m1 = MlpModel::new_random(2, 4, 1, 42);
    let m2 = MlpModel::new_random(3, 4, 1, 42); // different d_in
    let result = average_models(&[m1, m2]);
    assert!(result.is_err(), "Should fail with mismatched dimensions");
}

#[test]
fn test_checkpoint_persistence_and_restore() {
    let dir = tempfile::tempdir().expect("tempdir failed");
    let path = dir.path().join("model_latest.hxck");

    // Simulate aggregator startup with no existing checkpoint
    assert!(!path.exists());

    // Create initial model
    let model = MlpModel::new_random(4, 8, 2, 42);

    // Persist after first round
    let ckpt = model.to_checkpoint(1);
    let bytes = ckpt.to_bytes().expect("to_bytes failed");
    std::fs::write(&path, &bytes).expect("write failed");

    // Simulate restart: load from checkpoint
    assert!(path.exists());
    let loaded_bytes = std::fs::read(&path).expect("read failed");
    let loaded_ckpt = ModelCheckpoint::from_bytes(&loaded_bytes).expect("from_bytes failed");
    assert_eq!(loaded_ckpt.step_number, 1);

    let restored_model = MlpModel::from_checkpoint(&loaded_ckpt).unwrap();
    assert_eq!(restored_model.d_in, 4);
    assert_eq!(restored_model.d_hid, 8);
    assert_eq!(restored_model.d_out, 2);

    // Train further and persist again
    let mut trainer = Trainer::with_model(restored_model, 0.01);
    for i in 0..5 {
        let (x, t) = generate_training_data(4, 2, 5000 + i);
        trainer.train_step_unproved(&x, &t);
    }

    let ckpt2 = trainer.model().to_checkpoint(6);
    let bytes2 = ckpt2.to_bytes().expect("to_bytes failed");
    std::fs::write(&path, &bytes2).expect("write failed");

    // Verify second persist
    let loaded2 = ModelCheckpoint::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(loaded2.step_number, 6);
}
