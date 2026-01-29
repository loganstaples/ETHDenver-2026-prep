# HELIX

**Decentralized Verifiable Machine Learning Training**

## Overview

HELIX enables trustless, distributed training of machine learning models with cryptographic proofs ensuring computational integrity. Every gradient update, aggregation step, and model update is verified through zero-knowledge proofs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HELIX Platform                            │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  helix-core  │  helix-avm   │ helix-prover │   helix-node      │
│              │              │              │                   │
│  Types       │  Arithmetic  │  Chunking    │   Network         │
│  Tensors     │  Quantization│  Parallel    │   Gossip          │
│  Error       │  Gradients   │  Aggregation │   Sync            │
│  Tracking    │  VM Executor │  IVC         │   Roles           │
│  Data        │  Bounds      │  Keys        │                   │
│  Pipeline    │  NN Layers   │  Serialize   │                   │
└──────────────┴──────────────┴──────────────┴───────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     Smart Contracts (Solidity)                   │
├─────────────────────────────────────────────────────────────────┤
│  HelixCoordinator  │  TrainingRound  │  ProofVerifier          │
│  ErrorBoundRegistry│  TokenRewards   │  ModelRegistry          │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        Dashboard (Next.js)                       │
├─────────────────────────────────────────────────────────────────┤
│  TrainingProgress  │  ProofExplorer  │  ErrorBoundsViz         │
│  NodeNetworkView   │  ModelInteraction│ OnChainExplorer        │
└─────────────────────────────────────────────────────────────────┘
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
| `helix-prover` | ZK proof generation, aggregation, IVC |
| `helix-node` | P2P network, gossip protocol, node roles |

## Key Features

### Error Bound Tracking

All computations track error bounds through the pipeline:

```rust
use helix_core::ErrorBound;

let bound = ErrorBound::new(1e-7);
let result = bound.propagate_through_matmul(m, k, n);
```

### Quantization

INT4/INT8 quantization with calibration:

```rust
use helix_avm::quantization::{QuantConfig, QuantScheme};

let config = QuantConfig {
    weight_bits: 8,
    activation_bits: 8,
    scheme: QuantScheme::Symmetric,
    ..Default::default()
};
```

### Distributed Training

Byzantine-tolerant gradient aggregation:

```rust
use helix_node::roles::{ComputeNode, AggregatorNode};

let compute = ComputeNode::new(config);
let aggregator = AggregatorNode::new(agg_config);
```

### ZK Proofs

Generate proofs for training rounds:

```rust
use helix_prover::{ProofRequest, BatchProver};

let prover = BatchProver::new(config);
let proofs = prover.prove_batch(&witnesses)?;
```

## Dashboard

The Next.js dashboard provides real-time visualization:

- **Training Progress**: Loss/accuracy charts, round tracking
- **Proof Explorer**: Browse and verify proofs
- **Error Bounds**: Layer-wise error propagation
- **Network View**: Node topology and metrics
- **On-Chain Explorer**: Contract state and transactions

## License

MIT License
