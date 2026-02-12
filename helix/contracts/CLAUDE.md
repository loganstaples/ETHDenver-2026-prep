# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HELIX is a decentralized verifiable machine learning training protocol. This repository contains the Solidity smart contracts for on-chain coordination, along with the Rust backend crates in the parent directory.

## Build & Test Commands

### Contracts (this directory)
```bash
forge build                                    # Build contracts
forge test                                     # Run all tests
forge test --match-contract HelixCoordinatorV2 # Run specific test contract
forge test -vvv                                # Verbose test output
forge script script/Deploy.s.sol --broadcast  # Deploy with real verifier
forge script script/Deploy.s.sol --sig "runWithMock()" --broadcast  # Deploy with mock verifier
```

### Rust Crates (parent directory)
```bash
cargo build --workspace                        # Build all crates
cargo test --workspace                         # Run all tests
cargo test -p helix-circuits                   # Test specific crate
cargo test -p helix-prover test_name           # Run single test
cargo check -p helix-avm                       # Quick compile check
```

### Dashboard (../dashboard)
```bash
npm install && npm run dev                     # Run development server
npm run build                                  # Production build
```

## Architecture

### Contract Layer (`src/`)

**Core Protocol** (`src/core/`):
- `HelixCoordinatorV2.sol` - Main coordinator managing model registration, training rounds, staking, and proof submission with slashing
- `ModelRegistry.sol` - Model state commitments and version history
- `TrainingRound.sol` - Round state machine and gradient submission lifecycle

**ZK Verification** (`src/verification/`):
- `Halo2Verifier.sol` - BN254 pairing-based KZG proof verification (production)
- `PoseidonHasher.sol` - Poseidon hash matching Rust circuit (error checksum verification)
- `AggregationVerifier.sol` - Gradient aggregation proof verification

**Token & Incentives** (`src/token/`):
- `HelixToken.sol` - ERC20 with 100M cap and roles-based minting
- `Staking.sol` - Stake deposits with unbonding periods
- `Rewards.sol` - Per-round reward distribution

**Governance** (`src/governance/`):
- `TrainingDAO.sol` - Proposal-based governance with stake-weighted voting

### Rust Crates (parent `crates/`)

| Crate | Purpose |
|-------|---------|
| `helix-core` | Types, tensors, error tracking, benchmarking, data pipeline |
| `helix-avm` | Approximate VM, quantization, gradients, NN layers, circuit bridge |
| `helix-circuits` | Halo2 circuits for ML training steps, IVC, Freivalds verification |
| `helix-prover` | Proof generation pipeline, batch proving, aggregation |
| `helix-node` | P2P network, gossip protocol, node roles |
| `helix-mpc` | Multi-party computation for weight privacy |
| `helix-client` | Client SDK |

### Key Data Flow

1. Model owner registers model via `HelixCoordinatorV2.registerModel()`
2. Participants stake tokens via `stake(modelId)`
3. Owner starts training round via `startRound(modelId, duration)`
4. Participants generate ZK proofs of gradient computation (Rust prover)
5. Proofs submitted via `submitProof()` with 7 public inputs: `[oldHashLo, oldHashHi, newHashLo, newHashHi, loss, errorBound, stepNumber]`
6. `Halo2Verifier` validates proof; invalid proofs trigger slashing
7. Valid proofs update model commitment and accumulate error bounds

### Circuit-Contract Interface

Public inputs from `MLTrainingStepV2Circuit`:
- Indices 0-1: Old weight hash (split into lo/hi 128-bit halves)
- Indices 2-3: New weight hash (split into lo/hi 128-bit halves)
- Index 4: Computed loss value
- Index 5: Error bound for this step
- Index 6: Training step number

The contract reconstructs commitments via `_hashPair(lo, hi)` using keccak256.

## Key Patterns

- All contracts use Solidity `^0.8.19`
- `via_ir = true` in foundry.toml prevents stack-too-deep errors
- OpenZeppelin used for ERC20, AccessControl, SafeERC20, ReentrancyGuard
- Staking uses unbonding periods and slashing for economic security
- Error bounds tracked per model via `accumulatedErrorBound[modelId]`
