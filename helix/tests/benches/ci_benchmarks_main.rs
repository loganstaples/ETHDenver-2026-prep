//! CI Benchmark Runner Binary
//!
//! This is the main entry point for the CI-compatible benchmark runner.
//!
//! # Usage
//!
//! ```bash
//! cargo run --package helix-integration-tests --bin ci_benchmarks
//! ```

use helix_integration_tests::benches::ci_runner;

fn main() {
    ci_runner::main();
}
