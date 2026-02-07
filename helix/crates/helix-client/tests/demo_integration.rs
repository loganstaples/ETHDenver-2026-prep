//! Demo integration tests for helix-client
//!
//! Tests cover: dashboard endpoints, auth middleware, rate limiting,
//! RPC circuit breaker, recovery manager flows, HelixClient facade,
//! benchmark correctness, config validation, and dashboard data consistency.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt; // for `oneshot`

use helix_client::dashboard::{
    create_dashboard_router, create_dashboard_router_with_state,
    demo_nodes, demo_metrics, demo_events, demo_training_status, populate_demo_network,
    DashboardConfig, DashboardState, NetworkStatus,
};
use helix_client::rpc::client::{
    HelixRpcConfig, HelixRpcClient, RpcCircuitBreaker, RpcCircuitBreakerState,
};
use helix_client::demo::recovery::{
    CircuitBreaker, CircuitState, ErrorCategory, RecoveryConfig, RecoveryManager,
};
use helix_client::HelixClient;
use helix_client::config::ConfigProfile;
use helix_client::benchmark::{BenchmarkResults, BenchmarkType};

// ============================================================================
// Dashboard endpoint tests
// ============================================================================

#[tokio::test]
async fn test_dashboard_endpoints_return_valid_json() {
    let config = DashboardConfig::default();
    let app = create_dashboard_router(config);

    let endpoints = [
        "/",
        "/health",
        "/api/status",
        "/api/network",
        "/api/training",
        "/api/nodes",
        "/api/metrics",
        "/api/events",
    ];

    for path in &endpoints {
        let req = Request::builder()
            .uri(*path)
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Endpoint {} should return 200",
            path
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body)
            .unwrap_or_else(|e| panic!("Endpoint {} should return valid JSON: {}", path, e));

        // Basic sanity: the response should not be null
        assert!(!json.is_null(), "Endpoint {} returned null", path);
    }
}

// ============================================================================
// Dashboard auth tests
// ============================================================================

#[tokio::test]
async fn test_dashboard_auth_rejects_bad_token() {
    let config = DashboardConfig {
        auth_token: Some("secret-token-123".to_string()),
        allowed_origins: Vec::new(),
        rate_limit_per_second: 100,
    };
    let app = create_dashboard_router(config);

    // Request without token → 401
    let req = Request::builder()
        .uri("/api/status")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    // Request with wrong token → 401
    let req = Request::builder()
        .uri("/api/status")
        .header("Authorization", "Bearer wrong-token")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    // Request with valid token → 200
    let req = Request::builder()
        .uri("/api/status")
        .header("Authorization", "Bearer secret-token-123")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_dashboard_health_bypasses_auth() {
    let config = DashboardConfig {
        auth_token: Some("locked-down".to_string()),
        allowed_origins: Vec::new(),
        rate_limit_per_second: 100,
    };
    let app = create_dashboard_router(config);

    // /health should work without a token
    let req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ============================================================================
// Dashboard rate limiting tests
// ============================================================================

#[tokio::test]
async fn test_dashboard_rate_limiting() {
    let config = DashboardConfig {
        auth_token: None,
        allowed_origins: Vec::new(),
        rate_limit_per_second: 3, // very low limit for testing
    };
    let app = create_dashboard_router(config);

    // Fire requests rapidly
    let mut statuses = Vec::new();
    for _ in 0..6 {
        let req = Request::builder()
            .uri("/api/status")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        statuses.push(resp.status());
    }

    // At least some should be 429 (the first 3 pass, rest should be rate-limited)
    let ok_count = statuses.iter().filter(|s| **s == StatusCode::OK).count();
    let rate_limited = statuses
        .iter()
        .filter(|s| **s == StatusCode::TOO_MANY_REQUESTS)
        .count();

    assert!(ok_count >= 1, "At least 1 request should succeed");
    assert!(rate_limited >= 1, "At least 1 request should be rate-limited");
}

// ============================================================================
// Dashboard state wiring tests
// ============================================================================

#[tokio::test]
async fn test_dashboard_serves_updated_state() {
    let state = DashboardState::with_defaults();

    // Update training status
    state
        .update_training(helix_client::dashboard::TrainingStatus {
            model_id: 42,
            current_round: 7,
            total_rounds: 20,
            proofs_submitted: 14,
            proofs_verified: 14,
            error_bound: 12.3,
            max_error: 500.0,
            is_active: true,
        })
        .await;

    let app = create_dashboard_router_with_state(state);

    let req = Request::builder()
        .uri("/api/training")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["model_id"], 42);
    assert_eq!(json["current_round"], 7);
    assert_eq!(json["is_active"], true);
}

// ============================================================================
// RPC circuit breaker tests
// ============================================================================

#[test]
fn test_rpc_circuit_breaker_opens_after_threshold() {
    let mut cb = RpcCircuitBreaker::new(3, Duration::from_secs(30));

    // Initially closed
    assert_eq!(cb.state, RpcCircuitBreakerState::Closed);
    assert!(cb.allow_request());

    // Record failures below threshold
    cb.record_failure();
    cb.record_failure();
    assert_eq!(cb.state, RpcCircuitBreakerState::Closed);
    assert!(cb.allow_request());

    // Third failure trips the breaker
    cb.record_failure();
    assert_eq!(cb.state, RpcCircuitBreakerState::Open);
    assert!(!cb.allow_request());
}

#[test]
fn test_rpc_circuit_breaker_success_resets() {
    let mut cb = RpcCircuitBreaker::new(3, Duration::from_secs(30));

    cb.record_failure();
    cb.record_failure();
    // A success should reset the count
    cb.record_success();
    assert_eq!(cb.state, RpcCircuitBreakerState::Closed);
    assert_eq!(cb.failure_count, 0);

    // Now it takes 3 more failures to trip again
    cb.record_failure();
    cb.record_failure();
    assert_eq!(cb.state, RpcCircuitBreakerState::Closed);
}

#[test]
fn test_rpc_circuit_breaker_half_open_after_timeout() {
    let mut cb = RpcCircuitBreaker::new(1, Duration::from_millis(1));

    // Trip the breaker
    cb.record_failure();
    assert_eq!(cb.state, RpcCircuitBreakerState::Open);

    // Wait for reset timeout
    std::thread::sleep(Duration::from_millis(5));

    // Should transition to HalfOpen
    assert!(cb.allow_request());
    assert_eq!(cb.state, RpcCircuitBreakerState::HalfOpen);

    // A success should close it
    cb.record_success();
    assert_eq!(cb.state, RpcCircuitBreakerState::Closed);
}

// ============================================================================
// Recovery Manager circuit breaker tests
// ============================================================================

#[tokio::test]
async fn test_recovery_circuit_breaker_transitions() {
    let cb = CircuitBreaker::new(3, 2, Duration::from_millis(50));

    // Closed initially
    assert_eq!(cb.state().await, CircuitState::Closed);
    assert!(cb.allow().await);

    // Record failures → trips to Open
    cb.record_failure().await;
    cb.record_failure().await;
    cb.record_failure().await;
    assert_eq!(cb.state().await, CircuitState::Open);
    assert!(!cb.allow().await);

    // Wait for reset timeout → HalfOpen
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert!(cb.allow().await);
    assert_eq!(cb.state().await, CircuitState::HalfOpen);

    // Success in HalfOpen → needs success_threshold (2)
    cb.record_success().await;
    // Still HalfOpen (need 2 successes)
    assert_eq!(cb.state().await, CircuitState::HalfOpen);
    cb.record_success().await;
    // Now Closed
    assert_eq!(cb.state().await, CircuitState::Closed);
}

#[tokio::test]
async fn test_recovery_circuit_breaker_halfopen_failure_reopens() {
    let cb = CircuitBreaker::new(1, 2, Duration::from_millis(10));

    // Trip the breaker
    cb.record_failure().await;
    assert_eq!(cb.state().await, CircuitState::Open);

    // Wait for half-open
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(cb.allow().await);
    assert_eq!(cb.state().await, CircuitState::HalfOpen);

    // Failure in half-open → back to Open
    cb.record_failure().await;
    assert_eq!(cb.state().await, CircuitState::Open);
}

// ============================================================================
// Recovery Manager error cycle tests
// ============================================================================

#[tokio::test]
async fn test_recovery_manager_error_cycle() {
    let manager = RecoveryManager::new(RecoveryConfig {
        auto_recovery: true,
        max_recovery_attempts: 3,
        recovery_delay: Duration::from_millis(10),
        recovery_timeout: Duration::from_secs(5),
        graceful_degradation: true,
        min_workers: 1,
        health_check_interval: Duration::from_secs(1),
        auto_restart_nodes: false,
    });

    // Record an error
    manager
        .record_error(ErrorCategory::NetworkError, "Connection refused", Some("rpc"))
        .await;

    let stats = manager.stats().await;
    assert_eq!(stats.total_errors, 1);

    // Attempt recovery
    let result = manager
        .attempt_recovery(ErrorCategory::NetworkError, Some("rpc"))
        .await;
    assert!(result.is_ok());

    let stats = manager.stats().await;
    assert_eq!(stats.successful_recoveries, 1);
}

#[tokio::test]
async fn test_recovery_manager_degradation_on_repeated_errors() {
    let manager = RecoveryManager::new(RecoveryConfig {
        auto_recovery: true,
        max_recovery_attempts: 2,
        recovery_delay: Duration::from_millis(1),
        recovery_timeout: Duration::from_secs(5),
        graceful_degradation: true,
        min_workers: 1,
        health_check_interval: Duration::from_secs(1),
        auto_restart_nodes: false,
    });

    // Record enough errors to trigger degradation
    for _ in 0..3 {
        manager
            .record_error(ErrorCategory::ProofError, "proof gen failed", None)
            .await;
    }

    // Recovery should succeed but activate degradation
    let result = manager
        .attempt_recovery(ErrorCategory::ProofError, None)
        .await;
    assert!(result.is_ok());
    assert!(manager.is_degraded().await);
}

// ============================================================================
// DemoConfig validation tests
// ============================================================================

#[test]
fn test_quick_demo_config_validation() {
    let config = RecoveryConfig::demo_mode();

    assert!(config.auto_recovery);
    assert_eq!(config.max_recovery_attempts, 2);
    assert!(config.graceful_degradation);
    assert_eq!(config.min_workers, 1);
    assert!(config.recovery_delay < Duration::from_secs(1));
    assert!(config.recovery_timeout <= Duration::from_secs(10));
}

#[test]
fn test_strict_config_validation() {
    let config = RecoveryConfig::strict();

    assert!(config.auto_recovery);
    assert!(config.max_recovery_attempts >= 3);
    assert!(!config.graceful_degradation);
    assert!(config.min_workers >= 2);
}

// ============================================================================
// RPC config defaults
// ============================================================================

#[test]
fn test_rpc_config_has_max_backoff() {
    let config = HelixRpcConfig::default();
    assert_eq!(config.max_backoff_ms, 10_000);
    assert_eq!(config.retry_delay_ms, 500);
    assert_eq!(config.max_retries, 3);
}

// ============================================================================
// HelixClient facade tests
// ============================================================================

#[test]
fn test_helix_client_from_profile() {
    let client = HelixClient::from_profile(ConfigProfile::Local)
        .expect("Local profile should produce a valid client");

    assert_eq!(client.config().profile, ConfigProfile::Local);
    // Starts in mock mode
    assert!(!client.is_connected());
}

#[test]
fn test_helix_client_with_dashboard() {
    let dashboard = DashboardState::with_defaults();
    let client = HelixClient::from_profile(ConfigProfile::Local)
        .unwrap()
        .with_dashboard(dashboard);

    assert!(client.dashboard_state().is_some());
}

#[test]
fn test_helix_client_rejects_invalid_config() {
    let mut config = helix_client::config::HelixConfig::default();
    config.training.batch_size = 0; // invalid
    assert!(HelixClient::new(config).is_err());
}

// ============================================================================
// Benchmark NaN safety tests
// ============================================================================

#[test]
fn test_benchmark_results_nan_safety() {
    let samples = vec![1.0, f64::NAN, 3.0, 2.0, f64::NAN];
    // Should not panic
    let results = BenchmarkResults::from_samples(
        BenchmarkType::ProofGen,
        5,
        0,
        samples,
    );
    // Mean will be NaN due to NaN inputs, but the important thing is no panic
    assert!(!results.p50_ms.is_infinite());
}

#[test]
fn test_benchmark_percentile_sorted() {
    // Known sorted input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    let samples: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    let results = BenchmarkResults::from_samples(
        BenchmarkType::ProofVerify,
        10,
        0,
        samples,
    );

    assert_eq!(results.min_ms, 1.0);
    assert_eq!(results.max_ms, 10.0);
    assert_eq!(results.mean_ms, 5.5);
    // P50 of [1..10]: index = (10 * 50 / 100) = 5 → element at [5] = 6.0
    assert_eq!(results.p50_ms, 6.0);
    // Std dev of 1..10 = sqrt(8.25) ≈ 2.872
    assert!((results.std_dev_ms - 2.872).abs() < 0.01);
}

// ============================================================================
// Config validation tests
// ============================================================================

#[test]
fn test_config_default_validates() {
    let config = helix_client::config::HelixConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_rejects_zero_batch_size() {
    let mut config = helix_client::config::HelixConfig::default();
    config.training.batch_size = 0;
    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("Batch size"));
}

#[test]
fn test_config_rejects_negative_learning_rate() {
    let mut config = helix_client::config::HelixConfig::default();
    config.training.learning_rate = -0.5;
    let err = config.validate().unwrap_err();
    assert!(err.to_string().contains("Learning rate"));
}

// ============================================================================
// Dashboard data consistency tests
// ============================================================================

#[tokio::test]
async fn test_dashboard_defaults_and_fallback_match() {
    // Verify that with_defaults() and handler fallbacks return identical data.
    // We test this by:
    // 1. Getting data from an empty-state dashboard (handler fallbacks)
    // 2. Comparing against the demo_*() helper functions directly

    let empty_config = DashboardConfig::default();
    let app = create_dashboard_router(empty_config);

    // -- /api/nodes --
    let req = Request::builder().uri("/api/nodes").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let handler_nodes: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let helper_nodes = serde_json::to_value(&demo_nodes()).unwrap();
    assert_eq!(handler_nodes, helper_nodes, "/api/nodes fallback should match demo_nodes()");

    // -- /api/metrics --
    let req = Request::builder().uri("/api/metrics").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let handler_metrics: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(handler_metrics, demo_metrics(), "/api/metrics fallback should match demo_metrics()");

    // -- /api/events --
    let req = Request::builder().uri("/api/events").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let handler_events: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let helper_events = serde_json::to_value(&demo_events()).unwrap();
    assert_eq!(handler_events, helper_events, "/api/events fallback should match demo_events()");

    // -- /api/training --
    let req = Request::builder().uri("/api/training").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let handler_training: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let helper_training = serde_json::to_value(&demo_training_status()).unwrap();
    assert_eq!(handler_training, helper_training, "/api/training fallback should match demo_training_status()");

    // -- /api/network (populate_demo_network) --
    let req = Request::builder().uri("/api/network").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let handler_network: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let mut expected_network = NetworkStatus::default();
    populate_demo_network(&mut expected_network);
    // uptime_seconds will differ, so just check the populated fields
    assert_eq!(handler_network["node_id"], "helix-node-a1b2c3d4");
    assert_eq!(handler_network["status"], "online");
    assert_eq!(handler_network["peer_count"], 4);
    assert_eq!(handler_network["block_height"], 12_345_678);
    assert_eq!(handler_network["chain_id"], 31337);
}
