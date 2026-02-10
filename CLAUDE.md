# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HELIX is a decentralized verifiable machine learning training protocol (ETHDenver 2026). It enables trustless, distributed training where every gradient update and aggregation step is verified through zero-knowledge proofs.

## Build & Test Commands

### Rust Workspace (helix/)
```bash
cargo build --workspace                 # Build all crates
cargo test --workspace                  # Run all tests
cargo test -p helix-circuits            # Test specific crate
cargo test -p helix-prover test_name    # Run single test
cargo check -p helix-avm                # Quick compile check
cargo bench -p helix-avm                # Run benchmarks
```

### Smart Contracts (helix/contracts/)
```bash
forge build                                    # Build contracts
forge test                                     # Run all tests
forge test --match-contract HelixCoordinatorV2 # Run specific test contract
forge test -vvv                                # Verbose test output
forge script script/Deploy.s.sol --broadcast  # Deploy with real verifier
forge script script/Deploy.s.sol --sig "runWithMock()" --broadcast  # Deploy with mock verifier
```

### Dashboard (helix/dashboard/)
```bash
npm install && npm run dev   # Run development server
npm run build                # Production build
npm run lint                 # Lint check
```

### Integration Tests (helix/tests/)
```bash
cargo test --test end_to_end -- --test-threads=1
cargo test --test verification_consistency
```

## Architecture

### Rust Crates (helix/crates/)

| Crate | Purpose |
|-------|---------|
| `helix-core` | Types, tensors, error tracking, data pipeline |
| `helix-avm` | Approximate VM, quantization, gradients, NN layers |
| `helix-circuits` | Halo2 circuits for ML training steps, IVC |
| `helix-prover` | Proof generation, batch proving, aggregation |
| `helix-node` | P2P network, gossip protocol, node roles |
| `helix-mpc` | Multi-party computation for weight privacy |
| `helix-client` | Client SDK for orchestration |

**Crate dependencies flow**: core → avm/circuits → prover → node

### Smart Contracts (helix/contracts/src/)

- **core/**: `HelixCoordinatorV2` (main coordinator), `ModelRegistry`, `TrainingRound`
- **verification/**: `Halo2Verifier` (BN254 pairing-based KZG verification), `BoundsChecker`, `AggregationVerifier`
- **token/**: `HelixToken` (ERC20), `Staking`, `Rewards`
- **governance/**: `TrainingDAO`

### Key Data Flow

1. Model owner registers model via `HelixCoordinatorV2.registerModel()`
2. Participants stake tokens via `stake(modelId)`
3. Owner starts training round via `startRound(modelId, duration)`
4. Participants generate ZK proofs of gradient computation (Rust prover)
5. Proofs submitted via `submitProof()` with 8 public inputs: `[oldHashLo, oldHashHi, newHashLo, newHashHi, loss, errorBound, stepNumber, errorChecksum]`
6. `Halo2Verifier` validates proof; invalid proofs trigger slashing
7. Valid proofs update model commitment and accumulate error bounds

### Circuit-Contract Interface

Public inputs from `MLTrainingStepV2Circuit` (8 elements):
- Indices 0-1: Old weight hash (split into lo/hi 128-bit halves)
- Indices 2-3: New weight hash (split into lo/hi 128-bit halves)
- Index 4: Computed loss value
- Index 5: Error bound for this step
- Index 6: Training step number
- Index 7: Error checksum (SHA-256 commitment to error state)

The contract reconstructs commitments via `_hashPair(lo, hi)` using keccak256.

## Key Patterns

- **Error Bound Tracking**: All computations track numerical error margins through the entire pipeline
- **Quantization**: INT4/INT8 support with error accounting
- **Byzantine-Tolerant Aggregation**: Gradient outlier detection in aggregator nodes
- **Economic Security**: Staking with slashing for invalid proofs

### Solidity Specifics
- All contracts use Solidity `^0.8.19`
- `via_ir = true` in foundry.toml prevents stack-too-deep errors
- OpenZeppelin used for ERC20, AccessControl, SafeERC20, ReentrancyGuard

## Prerequisites

- Rust 1.75+
- Node.js 18+
- Foundry (for contracts)
