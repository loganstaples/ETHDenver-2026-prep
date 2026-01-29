# HELIX: The Approximate Intelligence Protocol

> Verifiable AI training at the speed of approximation.

HELIX is a ZK-verified decentralized AI training network powered by a probabilistic execution virtual machine that makes verification practical by proving computation is *within bounds* rather than bit-exact.

## Architecture

```
helix/
├── crates/                    # Rust workspace
│   ├── helix-core/            # Shared types, traits, utilities
│   ├── helix-avm/             # Approximate Virtual Machine
│   ├── helix-circuits/        # Halo2 ZK circuits
│   ├── helix-prover/          # Proof generation orchestration
│   ├── helix-node/            # Training node implementation
│   └── helix-client/          # CLI and client library
├── contracts/                 # Solidity smart contracts
├── dashboard/                 # Next.js frontend
├── proto/                     # Protocol buffer definitions
└── docs/                      # Documentation
```

## Quick Start

```bash
# Build all Rust crates
cargo build

# Run tests
cargo test

# Build contracts
cd contracts && forge build

# Run dashboard
cd dashboard && npm run dev
```

## The Core Insight

Neural network training is already approximate by design—SGD uses noisy gradients, quantized training works, and convergence proofs assume bounded noise. HELIX embraces this reality, making ZK proofs practical by proving bounded computation rather than exact equality.

## License

MIT
