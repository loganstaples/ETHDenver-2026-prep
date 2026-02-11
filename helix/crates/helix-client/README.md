# helix-client

CLI and client library for interacting with the HELIX decentralized verifiable ML training network.

## Purpose

User-facing tools for training management, network interaction, on-chain operations, and wallet management.

## Commands

```bash
helix init          # Initialize node, create config and wallet
helix join          # Join training network (real RPC + on-chain staking)
helix status        # Check training/network/proof status (live data via RPC)
helix query <type>  # Query model, round, stake, proof, worker, metrics
helix export        # Export training data, proofs, metrics to JSON
helix train         # Run training rounds with ZK proof generation
helix demo          # 90-second demo with real halo2 proofs
helix benchmark     # Performance benchmarks (real ZK proofs or simulated)
helix health        # Health monitoring (HTTP, TCP, JSON-RPC probes)
helix dashboard     # Launch REST API (axum + auth + rate limiting)
helix visualize     # Interactive terminal UI (ratatui, 4 tabs)
```

## Library Usage

```rust
use helix_client::{HelixClient, TrainingOrchestrator, NetworkOrchestrator};
use helix_client::rpc::client::UnifiedRpcClient;

// Connect to a running HELIX node (falls back to mock in debug builds)
let rpc = UnifiedRpcClient::connect_or_mock(config).await;
let status = rpc.get_training_status().await?;

// Full E2E orchestration: Anvil + deploy + train + prove + submit
let orchestrator = TrainingOrchestrator::new(orch_config);
let result = orchestrator.run().await?;
```

## Features

| Feature | Flag | Description |
|---------|------|-------------|
| On-chain integration | `chain` | ethers-rs contract calls, deployment via forge |
| Hardware wallets | `hardware-wallet` | Ledger Nano S/X/Plus via HID |
| Mock fallback | `mock-fallback` | Auto-fallback to simulated RPC for demos |

## Architecture

```
CLI (main.rs) --> UnifiedRpcClient --> {HelixRpcClient | MockRpcClient}
                                  --> ChainClient --> HelixCoordinatorV2 (on-chain)
              --> TrainingOrchestrator --> Anvil + forge + MLTrainingProverV2
              --> BenchmarkRunner --> RealTrainingExecutor (halo2 proofs)
              --> SecureWallet --> AES-256-GCM + Argon2id + OS Keychain
```

## Key Modules

- **`rpc/`** -- JSON-RPC client with circuit breaker + mock fallback; ethers-rs chain client
- **`wallet/`** -- BIP-39/BIP-44 HD wallet, Ledger support, OS keychain (macOS/Linux/Windows), audit logging
- **`demo/`** -- Real ZK proof generation via `MLTrainingProverV2`, 90-second timed demo
- **`benchmark.rs`** -- Real halo2 proof benchmarks via `BenchmarkRunner::with_real_proofs()`
- **`orchestration.rs`** -- E2E training: Anvil + deploy + register + stake + prove + submit
- **`health.rs`** -- Real HTTP, TCP, JSON-RPC, filesystem, and process health probes

## Security

- Wallet encryption: Argon2id KDF + AES-256-GCM (via `SecureWallet`)
- Legacy wallet uses real `sha3::Keccak256` and `OsRng` (deprecated; use `SecureWallet` for production)
- Private key defaults read from `HELIX_PRIVATE_KEY` env var; known Anvil keys rejected on non-local networks
- Init command generates cryptographically random keystore passwords
- All key material `zeroize`d on drop

## Building

```bash
cargo build -p helix-client                    # Default (no chain features)
cargo build -p helix-client --features chain   # With on-chain integration
cargo test -p helix-client                     # Run all tests
cargo test -p helix-client --features chain    # Including chain integration tests
```
