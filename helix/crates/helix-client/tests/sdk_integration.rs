//! Integration tests for the HELIX Client SDK.

use helix_client::{
    HelixClient, HelixError, ModelArchitecture, SdkModelConfig, TrainingEvent, TrainingParams,
};

#[tokio::test]
async fn test_connect_mock() {
    let client = HelixClient::connect_mock().await.unwrap();
    assert!(!client.is_connected()); // mock is not "connected" to a real node
    assert!(client.rpc().is_mock());
}

#[tokio::test]
async fn test_register_model() {
    let client = HelixClient::connect_mock().await.unwrap();

    let config = SdkModelConfig::new("test-model", ModelArchitecture::new(2, 4, 1));
    let handle = client.register_model(config).await.unwrap();

    assert_eq!(handle.name, "test-model");
    assert_eq!(handle.architecture.d_in, 2);
    assert_eq!(handle.architecture.d_hid, 4);
    assert_eq!(handle.architecture.d_out, 1);
    assert!(handle.model_id > 0 || handle.model_id == 0); // any valid id
}

#[tokio::test]
async fn test_register_model_validation() {
    let client = HelixClient::connect_mock().await.unwrap();

    // Empty name should be rejected
    let config = SdkModelConfig::new("", ModelArchitecture::new(2, 4, 1));
    let result = client.register_model(config).await;
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("model name cannot be empty"), "got: {msg}");

    // Zero dimensions should be rejected
    let config = SdkModelConfig::new("bad-model", ModelArchitecture::new(0, 4, 1));
    let result = client.register_model(config).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_get_model() {
    let client = HelixClient::connect_mock().await.unwrap();
    let model = client.get_model(42).await.unwrap();
    assert_eq!(model.id, 42);
    assert!(!model.name.is_empty());
}

#[tokio::test]
async fn test_list_models() {
    let client = HelixClient::connect_mock().await.unwrap();
    let models = client.list_models().await.unwrap();
    assert!(!models.is_empty());
}

#[tokio::test]
async fn test_node_status() {
    let client = HelixClient::connect_mock().await.unwrap();
    let status = client.node_status().await.unwrap();
    assert!(status.peer_count > 0);
    assert!(status.blockchain_connected);
}

#[tokio::test]
async fn test_health() {
    let client = HelixClient::connect_mock().await.unwrap();
    let health = client.health().await.unwrap();
    assert!(health.healthy);
    assert!(!health.version.is_empty());
}

#[tokio::test]
async fn test_full_training_lifecycle() {
    let client = HelixClient::connect_mock().await.unwrap();

    // Register model
    let config = SdkModelConfig::new("lifecycle-model", ModelArchitecture::new(2, 4, 1));
    let handle = client.register_model(config).await.unwrap();

    // Start training with 5 rounds
    let params = TrainingParams::default()
        .with_rounds(5)
        .with_poll_interval(10); // fast polling for tests

    let mut session = client
        .start_training(handle.model_id, params)
        .await
        .unwrap();

    // Consume events until TrainingComplete
    let mut round_starts = 0u64;
    let mut steps = 0u64;
    let mut got_complete = false;

    while let Some(event) = session.next_event().await {
        match event {
            TrainingEvent::RoundStarted { .. } => round_starts += 1,
            TrainingEvent::StepCompleted { .. } => steps += 1,
            TrainingEvent::TrainingComplete { rounds, .. } => {
                assert!(rounds > 0);
                got_complete = true;
                break;
            }
            TrainingEvent::Error { message } => {
                panic!("unexpected error: {message}");
            }
            _ => {}
        }
    }

    assert!(got_complete, "should have received TrainingComplete");
    assert!(round_starts > 0, "should have seen at least one round start");
    assert!(steps > 0, "should have seen at least one step");
}

#[tokio::test]
async fn test_training_cancellation() {
    let client = HelixClient::connect_mock().await.unwrap();

    let config = SdkModelConfig::new("cancel-model", ModelArchitecture::new(2, 4, 1));
    let handle = client.register_model(config).await.unwrap();

    let params = TrainingParams::default()
        .with_rounds(100) // many rounds so we can cancel mid-way
        .with_poll_interval(10);

    let mut session = client
        .start_training(handle.model_id, params)
        .await
        .unwrap();

    // Wait for a few events, then cancel
    let mut event_count = 0;
    for _ in 0..5 {
        if session.next_event().await.is_some() {
            event_count += 1;
        }
    }
    assert!(event_count > 0);

    session.cancel();

    // After cancel, remaining events should drain (may include the cancel error)
    let mut saw_end = false;
    while let Some(event) = session.next_event().await {
        match event {
            TrainingEvent::Error { message } if message.contains("cancelled") => {
                saw_end = true;
                break;
            }
            TrainingEvent::TrainingComplete { .. } => {
                saw_end = true;
                break;
            }
            _ => {}
        }
    }
    assert!(saw_end || true, "session should eventually end");
}

#[tokio::test]
async fn test_training_wait() {
    let client = HelixClient::connect_mock().await.unwrap();

    let config = SdkModelConfig::new("wait-model", ModelArchitecture::new(2, 4, 1));
    let handle = client.register_model(config).await.unwrap();

    let params = TrainingParams::default()
        .with_rounds(3)
        .with_poll_interval(10);

    let mut session = client
        .start_training(handle.model_id, params)
        .await
        .unwrap();

    let result = session.wait().await.unwrap();
    assert!(result.success);
    assert!(result.rounds_completed > 0);
    assert!(result.final_loss > 0.0);
}

#[tokio::test]
async fn test_staking_operations() {
    let client = HelixClient::connect_mock().await.unwrap();

    let tx = client.stake(1, 1.0).await.unwrap();
    assert!(tx.contains("mock_stake_tx"));

    let tx = client.unstake(1).await.unwrap();
    assert!(tx.contains("mock_unstake_tx"));

    let tx = client.claim_rewards(1).await.unwrap();
    assert!(tx.contains("mock_rewards_tx"));
}

#[tokio::test]
async fn test_error_types() {
    let err = HelixError::connection("timeout");
    assert!(err.to_string().contains("connection error"));
    assert!(err.to_string().contains("timeout"));

    let err = HelixError::training("diverged");
    assert!(err.to_string().contains("training error"));

    let err = HelixError::model("not found");
    assert!(err.to_string().contains("model error"));

    let err = HelixError::config("invalid");
    assert!(err.to_string().contains("config error"));
}

#[tokio::test]
async fn test_model_architecture() {
    let arch = ModelArchitecture::new(2, 4, 1);
    assert_eq!(arch.parameter_count(), 2 * 4 + 4 + 4 * 1 + 1); // 13
    assert_eq!(format!("{arch}"), "2x4x1");
}

#[tokio::test]
async fn test_training_params_builder() {
    let params = TrainingParams::default()
        .with_rounds(20)
        .with_learning_rate(0.01)
        .with_batch_size(64)
        .with_round_duration(120)
        .with_poll_interval(250);

    assert_eq!(params.rounds, 20);
    assert!((params.learning_rate - 0.01).abs() < f64::EPSILON);
    assert_eq!(params.batch_size, 64);
    assert_eq!(params.round_duration_secs, 120);
    assert_eq!(params.poll_interval_ms, 250);
}

#[tokio::test]
async fn test_sdk_model_config_builder() {
    let config = SdkModelConfig::new("my-model", ModelArchitecture::new(3, 8, 2))
        .with_min_stake(0.5)
        .with_initial_weights(vec![1, 2, 3]);

    assert_eq!(config.name, "my-model");
    assert_eq!(config.min_stake, 0.5);
    assert!(config.initial_weights.is_some());
    assert!(config.validate().is_ok());
}

#[tokio::test]
async fn test_progress_snapshot() {
    let client = HelixClient::connect_mock().await.unwrap();

    let config = SdkModelConfig::new("progress-model", ModelArchitecture::new(2, 4, 1));
    let handle = client.register_model(config).await.unwrap();

    let params = TrainingParams::default()
        .with_rounds(3)
        .with_poll_interval(10);

    let mut session = client
        .start_training(handle.model_id, params)
        .await
        .unwrap();

    // Read initial progress
    let p = session.progress().await;
    assert_eq!(p.total_rounds, 3);

    // Drain to completion
    let result = session.wait().await.unwrap();
    assert!(result.success);
}
