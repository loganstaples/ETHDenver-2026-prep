# HELIX

**Decentralized Verifiable Machine Learning Training**

## Overview

HELIX enables trustless, distributed training of machine learning models where workers train on secret-shared weights using MPC. SPDZ information-theoretic MACs provide mathematically certain cheater detection, individual bad actor identification, and on-chain settlement — all without anyone ever seeing the model weights.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          HELIX Platform                             │
├────────────┬────────────┬────────────┬────────────┬────────────────┤
│ helix-core │ helix-avm  │ helix-mpc  │ helix-node │ helix-prover   │
│            │            │            │            │                │
│ Types      │ Arithmetic │ SPDZ MACs  │ Network    │ Checkpoint     │
│ Tensors    │ Quantize   │ Beaver     │ Gossip     │ Proofs         │
│ Error      │ Gradients  │ Secret     │ Sync       │ Aggregation    │
│ Tracking   │ VM Exec    │ Sharing    │ Transport  │ Serialization  │
│ Data       │ Bounds     │ NN Layers  │ Roles      │                │
│ Pipeline   │ NN Layers  │ Sessions   │            │                │
├────────────┴────────────┼────────────┴────────────┼────────────────┤
│      helix-circuits     │     helix-client        │  helix-demo    │
│      Halo2 circuits     │     Orchestration SDK   │  E2E demo      │
└─────────────────────────┴─────────────────────────┴────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                      Smart Contracts (Solidity)                     │
├─────────────────────────────────────────────────────────────────────┤
│  HelixCoordinatorV3  │  ModelRegistry   │  Halo2Verifier           │
│  Staking / Slashing  │  TrainingRound   │  RLCAggregationVerifier  │
│  HelixToken (ERC20)  │  Rewards         │  PoseidonHasher          │
│  HelixCoordinatorV4 (MPC-primary, optional ZK)                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                        Dashboard (Next.js)                          │
├─────────────────────────────────────────────────────────────────────┤
│  Marketplace     │  Portfolio         │  MPC Training              │
│  Inference       │  Network Monitor   │  Model Detail              │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Rust 1.75+
- Node.js 18+
- Foundry (for contracts)

### Build

```bash
# Build all Rust crates
cargo build --workspace

# Build contracts
cd contracts && forge build

# Build dashboard
cd dashboard && npm install && npm run build
```

### Run Tests

```bash
# Run all Rust tests
cargo test --workspace

# Run contract tests
cd contracts && forge test

# Run dashboard
cd dashboard && npm run dev
```

## Crates

| Crate | Description |
|-------|-------------|
| `helix-core` | Core types, tensors, error tracking, data pipeline |
| `helix-avm` | Approximate VM, quantization, gradients, NN layers |
| `helix-mpc` | MPC training engine: SPDZ MACs, Beaver triples, secret sharing, sessions |
| `helix-circuits` | Halo2 ZK circuits for ML training steps and state transitions |
| `helix-prover` | Proof generation, batch proving, aggregation |
| `helix-node` | P2P network, gossip protocol, node roles, transport |
| `helix-client` | CLI tool for job orchestration and on-chain interaction |
| `helix-demo` | End-to-end demo runner (MNIST training) |

## Key Features

### Information-Theoretic Security

SPDZ MACs cannot be broken even with unlimited computing power — strictly stronger than computational assumptions used by ZK proofs or TEEs:

```rust
use helix_mpc::security::mac::SpdzMac;

// Every secret-shared value carries a MAC
// Cheating is detected with mathematical certainty
let authenticated_share = SpdzMac::authenticate(share, mac_key);
```

### Individual Cheater Identification

Pairwise MAC verification pinpoints the exact bad actor — no group punishment:

```rust
use helix_mpc::mac_verification::PairwiseMacVerifier;

let verifier = PairwiseMacVerifier::new(parties);
let blame = verifier.identify_cheater(&mac_failures);
// Specific worker is slashed on-chain
```

### Zero Weight Leakage

Model weights exist only as secret shares during training. No single worker, coordinator, or contract ever sees the full model:

```rust
use helix_mpc::sharing::additive::AdditiveSharing;

let shares = AdditiveSharing::split(&weights, num_workers);
// Each worker holds one share — useless alone
```

### Error Bound Tracking

All computations track numerical error margins through the pipeline:

```rust
use helix_core::ErrorBound;

let bound = ErrorBound::new(1e-7);
let result = bound.propagate_through_matmul(m, k, n);
```

### Self-Healing Training

When a cheater is detected, they're removed and training continues with remaining honest workers — no restart needed.

## Dashboard

The Next.js dashboard serves as both a marketplace and training platform:

- **Marketplace**: Browse, buy, and rate public models with search and sorting
- **Portfolio**: Personal dashboard with revenue charts, model management, activity feed
- **Training**: MPC training orchestrator with 13-phase workflow, MAC verification, cheater detection
- **Inference**: MNIST digit classifier with drawing canvas, MPC attestation details, per-model fees
- **Network**: Worker node monitoring with reputation, staking status, uptime metrics
- **Model Detail**: Version history, weight download, accuracy stats

## License

MIT License
