# HELIX Build & Test Targets
# ============================

.PHONY: build test test-integration test-e2e test-unit check fmt clippy clean

# Build the entire workspace
build:
	cargo build --workspace

# Run all unit tests (no feature gates)
test-unit:
	cargo test --workspace

# Run the full E2E integration test suite
# Requires: Foundry (forge, anvil) installed and in PATH
test-integration:
	@echo "=== HELIX E2E Integration Tests ==="
	@echo "Prerequisites: forge, anvil must be in PATH"
	@echo ""
	@echo "Building contracts..."
	@cd contracts && forge build
	@echo ""
	@echo "Running integration tests (this may take several minutes)..."
	cargo test --test e2e_full_round --features integration -- --test-threads=1
	cargo test --test e2e_mpc_training --features integration -- --test-threads=1
	cargo test --test e2e_byzantine --features integration -- --test-threads=1
	cargo test --test e2e_multi_round --features integration -- --test-threads=1
	cargo test --test e2e_client_sdk --features integration -- --test-threads=1
	cargo test --test e2e_failure_recovery --features integration -- --test-threads=1
	@echo ""
	@echo "=== All E2E integration tests passed ==="

# Run on-chain verification tests (subset of integration)
test-e2e: test-integration

# Run all tests (unit + integration)
test: test-unit test-integration

# Quick compile check
check:
	cargo check --workspace

# Format code
fmt:
	cargo fmt --all

# Run clippy
clippy:
	cargo clippy --workspace -- -D warnings

# Clean build artifacts
clean:
	cargo clean
