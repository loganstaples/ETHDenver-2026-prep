# HELIX Presentation Materials

## Elevator Pitch

**HELIX** enables trustless, decentralized training of AI models where every computation is cryptographically verified. No more "trust me" in AI—HELIX proves it.

## Problem

1. **AI Training is Opaque**: No way to verify that models were trained correctly
2. **Centralized Control**: Training happens on proprietary infrastructure
3. **No Accountability**: Model behavior can't be traced to training decisions

## Solution

HELIX provides:

1. **Verifiable Training**: ZK proofs for every gradient update
2. **Decentralized Compute**: Distributed training across participant nodes
3. **On-Chain Accountability**: Training history recorded on blockchain

## How It Works

```
1. Model Owner submits training job to HELIX Coordinator contract
2. Compute Nodes claim training rounds and process batches
3. Each node generates ZK proof of correct gradient computation
4. Aggregator collects gradients and generates aggregation proof
5. Proofs verified on-chain, rewards distributed
6. Model updated with proven gradients
```

## Key Innovations

### Approximate Virtual Machine (AVM)

- Tracks numerical error bounds through all operations
- Enables verification of ML computations
- Supports quantized (INT4/INT8) operations

### Recursive Proof Aggregation

- Combines thousands of computation proofs
- Efficient on-chain verification
- Constant verification cost regardless of training size

### Byzantine-Tolerant Aggregation

- Handles malicious or faulty nodes
- Gradient outlier detection
- Configurable threshold for consensus

## Demo Highlights

1. **Training Dashboard**: Real-time loss/accuracy visualization
2. **Proof Explorer**: Browse verified computation proofs
3. **Network View**: Live node topology and status
4. **On-Chain Explorer**: Track rewards and model updates

## Technical Stack

| Layer | Technology |
|-------|------------|
| Smart Contracts | Solidity, Foundry |
| Proof System | Custom ZK (Halo2-inspired) |
| Node Runtime | Rust, Tokio |
| Dashboard | Next.js, React |
| Storage | IPFS, Ethereum |

## Metrics

- **207+ Tests** passing across all crates
- **9 Dashboard routes** with full visualization
- **6 Contract modules** for on-chain coordination
- **Error bounds** tracked through entire pipeline

## Roadmap

### Phase 1: MVP (Current)
- ✅ Core error bound algebra
- ✅ Basic ZK proof generation
- ✅ P2P network implementation
- ✅ Dashboard visualization

### Phase 2: Optimization
- GPU-accelerated proving
- Proof batching and compression
- Enhanced tokenomics

### Phase 3: Production
- Mainnet deployment
- Model marketplace
- Enterprise integrations

## Team

Built for ETHDenver 2026

## Contact

- GitHub: [helix-ml/helix](https://github.com/helix-ml/helix)
- Twitter: @helix_ml
