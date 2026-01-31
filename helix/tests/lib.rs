//! HELIX Integration Test Library
//!
//! This crate provides comprehensive end-to-end integration testing for the HELIX
//! trustless AI training infrastructure. It includes:
//!
//! - Test harness for multi-crate integration testing
//! - Fixtures for deterministic test data generation
//! - Mock implementations for network simulation and adversarial testing
//! - Performance metrics and regression tracking
//! - Custom assertions for cryptographic verification
//! - Comprehensive performance benchmark suite
//!
//! # Organization
//!
//! - `common/` - Shared test infrastructure
//!   - `fixtures.rs` - Test data and configuration fixtures
//!   - `harness.rs` - Test harness with phase tracking and timeouts
//!   - `mocks.rs` - Mock implementations for network, workers, and verifiers
//!   - `metrics.rs` - Performance metrics and regression detection
//!   - `assertions.rs` - Custom assertions for ZK and MPC testing
//!
//! - `integration/` - Integration test modules
//!   - `end_to_end.rs` - Full pipeline tests
//!   - `mpc_zkp_integration.rs` - MPC + ZK proof integration
//!   - `adversarial.rs` - Byzantine node simulation
//!   - `network_partition.rs` - Network failure recovery
//!   - `verification_consistency.rs` - Native vs EVM verification
//!   - `training_convergence.rs` - Model training correctness
//!   - `checkpoint_resume.rs` - State persistence and recovery
//!   - `cross_platform.rs` - Platform compatibility
//!
//! - `benches/` - Performance benchmarks
//!   - `harness.rs` - Benchmark harness with standardized reporting
//!   - `native_baseline.rs` - Native computation baselines
//!   - `gkr_prover.rs` - GKR prover benchmarks
//!   - `halo2_prover.rs` - Halo2 prover benchmarks
//!   - `overhead_report.rs` - Automated overhead calculation
//!   - `metal_vs_cpu.rs` - GPU vs CPU comparison
//!   - `memory_profile.rs` - Memory usage profiling
//!   - `scaling.rs` - Scaling analysis
//!   - `performance_regression.rs` - CI/CD performance tracking

pub mod common;
pub mod benches;
