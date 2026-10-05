# HELIX

### Trustless Distributed ML Training with Information-Theoretic Security

*ETHDenver 2026*

---

HELIX is a decentralized machine learning training protocol where **no single party ever sees the model weights**. Workers train collaboratively on secret-shared weights using Multi-Party Computation (MPC), with SPDZ information-theoretic MACs providing mathematical certainty that any cheating is detected immediately, the exact cheater is identified, and they are slashed on-chain. Training self-heals and continues after removing bad actors.

No trusted hardware. No trusted coordinators. No computational assumptions. **Mathematical impossibility of undetected cheating.**

---

## Table of Contents

- [The Problem](#the-problem)
- [How HELIX Solves It](#how-helix-solves-it)
- [Architecture](#architecture)
- [Protocol Walkthrough](#protocol-walkthrough)
- [Live Demo](#live-demo)
- [Web Dashboard](#web-dashboard)
- [Technical Deep Dive](#technical-deep-dive)
- [Performance](#performance)
- [Project Scale](#project-scale)
- [Getting Started](#getting-started)

---

## The Problem

Decentralized ML training today forces a choice between privacy, verification, and efficiency. No existing system delivers all three:

| Project | What it does | Privacy | Verification | Overhead |
|---------|-------------|---------|-------------|----------|
| **EZKL / Giza / Modulus** | ZK proofs for inference | None (weights visible) | Strong (per-step proofs) | 100-1000x |
| **Gensyn** | Decentralized training | None (validators see everything) | Weak (random spot checks) | Low |
| **Ritual** | TEE-based inference | Hardware-dependent (SGX broken multiple times) | Trust Intel/AMD | Low |
| **Bittensor** | Decentralized inference | None | Validator re-computation | 2x+ |
| **io.net** | GPU marketplace | None | **None** | None |

The fundamental limitation: ZK proofs for every training step incur 100-1000x computational overhead, making them impractical for real training workloads. TEEs require trusting hardware manufacturers. Probabilistic checks miss targeted attacks.

## How HELIX Solves It

HELIX replaces ZK-every-step with **MPC + SPDZ MACs** -- achieving stronger security guarantees at a fraction of the cost:

| | HELIX |
|---|---|
| **Privacy** | Additive secret sharing -- weights exist only as shares, never reconstructed until the owner collects at the end |
| **Verification** | SPDZ MACs -- information-theoretic security that cannot be broken even with unlimited computing power |
| **Cheater ID** | Pairwise MAC verification pinpoints the exact bad actor, not just "someone cheated" |
| **Self-Healing** | Remove the cheater, redistribute shares from checkpoint, continue training |
| **Overhead** | 5-15x native (communication-bound, not proof-bound) |
| **On-Chain Cost** | ~50-80K gas per checkpoint via multi-party attestation |
| **ZK Proofs** | Optional add-on for external verifiability -- never required for security |

### Why Information-Theoretic Security Matters

Zero-knowledge proofs are *computationally* secure -- they rely on the assumption that certain math problems are hard to solve. If those assumptions break (quantum computing, new algorithms), the proofs break.

SPDZ MACs are *information-theoretically* secure -- they cannot be broken **even with unlimited computing power**. The guarantee is not "this would take a long time." The guarantee is mathematical impossibility. A cheating worker has a `1/p` probability of evading detection, where `p` is the size of the BN254 prime field (~2^254). That's not a bet anyone can win.

---

## Architecture

```
Model Owner                        Workers (3+ nodes)                    Smart Contract
     |                                   |                                    |
     |-- Register training job --------->|                                    |
     |                                   |-- Stake tokens & join ------------>|
     |-- Split weights into N shares --->|                                    |
     |-- Publish Pedersen commitment ----|------------------------------------>|
     |                                   |                                    |
     |                            [MPC Training Loop]                         |
     |                             Forward pass (Beaver triple multiplication)|
     |                             ReLU (garbled circuits via oblivious transfer)
     |                             Backpropagation (Beaver triples)           |
     |                             Weight update (LOCAL -- zero communication)|
     |                             SPDZ MAC verification (every step)         |
     |                                   |                                    |
     |                            [Periodic Checkpoints]                      |
     |                             Joint Pedersen commitment (no reconstruction)
     |                             All workers sign attestation               |
     |                                   |-- Submit attestation + sigs ------>|
     |                                   |                                    |
     |                            [If MAC Check Fails]                        |
     |                             Pairwise identification -> exact cheater   |
     |                             Honest majority signs blame report         |
     |                                   |-- Submit blame report ------------>| SLASH
     |                             Remove cheater, resume from checkpoint     |
     |                                   |                                    |
     |<-- Workers encrypt final shares --|                                    |
     |-- Reconstruct trained model       |                                    |
     |-- Verify accuracy                 |      Release payment to workers -->|
```

### Crate Architecture

```
helix-core          Foundation: types, tensors, error tracking, data pipeline
    |
helix-avm           Training engine: forward/backward pass, autodiff, quantization
    |
helix-mpc           Core MPC engine (37 modules):
    |                  - Additive secret sharing over BN254 Fr
    |                  - SPDZ MAC generation & batch verification
    |                  - Beaver triple generation via OT (MASCOT-style)
    |                  - Garbled circuit ReLU via x25519 OT
    |                  - Pedersen commitments on BN254 G1
    |                  - Pairwise cheater identification
    |                  - Checkpoint attestation & recovery
    |                  - Full E2E training pipeline
    |
helix-circuits      Halo2 circuits (optional ZK: StateTransitionCircuit)
helix-prover        KZG proof pipeline (optional, only when ZK enabled)
helix-node          P2P networking, gossip protocol, session coordination
helix-client        Owner SDK, full orchestration, dashboard HTTP/WS API
helix-demo          ETHDenver demo binary with terminal UI
```

### Smart Contracts

```
HelixCoordinatorV4          MPC-primary coordinator (1,520 lines)
  - registerTrainingJob()       Job registration with optional ZK config
  - stakeAndJoin()              Per-job worker staking
  - submitCheckpoint()          Multi-party signed attestation verification
  - submitCheckpointWithProof() Optional ZK proof verification (Halo2Verifier)
  - reportMACFailure()          Slash cheaters, 10% bounty to reporters
  - completeTraining()          Proportional payment distribution
  - pauseTraining() / stopTraining() / resumeTraining()
  - commitInferenceResult()     On-chain inference attestation
  - Worker Pool: registerInPool(), assignPoolWorkers()
  - Risk-based ZK: auto-activate when workers drop below threshold

HelixToken                  ERC20 with staking hooks
Staking / Rewards           Per-job staking, proportional rewards
Halo2Verifier               BN254 pairing-based KZG verification
TrainingDAO                 Governance
```

---

## Protocol Walkthrough

### Phase 1: Job Setup

The model owner registers a training job on-chain, deposits payment, and configures parameters (architecture, checkpoint frequency, ZK settings). Workers discover the job, stake tokens, and join. The owner's CLI:

1. Splits model weights into N additive secret shares: `W = s1 + s2 + ... + sN`
2. Encrypts each share to the corresponding worker's x25519 public key (DH + AES-GCM)
3. Distributes encrypted shares over direct TCP connections
4. Publishes a Pedersen commitment `C0 = g^W * h^r` on-chain (verifiable, reveals nothing)

Workers collaboratively pre-generate Beaver triples via oblivious transfer (MASCOT-style protocol). **No trusted dealer** -- every party contributes randomness via OT, and nobody sees the full triple. The global SPDZ MAC key `alpha` is secret-shared: `alpha = alpha_1 + alpha_2 + ... + alpha_N`. Nobody knows the full key.

### Phase 2: MPC Training

For each training step, all computation happens on secret shares:

- **Matrix multiplications** (forward and backward pass): Use Beaver triples. For each multiplication `a * b`, workers open `(a - alpha)` and `(b - beta)` where `(alpha, beta, gamma)` is a pre-computed triple with `gamma = alpha * beta`. From the opened values, each worker locally computes their share of `a*b`. One communication round per layer.

- **ReLU activation**: Uses garbled circuits via x25519 oblivious transfer. One worker garbles the comparison circuit; others evaluate. Each worker gets their share of `max(0, x)` without anyone learning `x`.

- **Weight updates**: Each worker locally updates their share: `si_new = si_old - lr * grad_i`. **Zero communication needed** -- additive secret sharing is linear, so subtracting a share of the gradient from a share of the weights yields a share of the updated weights.

- **SPDZ MAC verification** (configurable, default every step): Workers run a commit-then-reveal sigma protocol. Each party computes `sigma_i`, commits `hash(sigma_i || nonce)`, then reveals. If `sum(sigma_i) = 0`, all computations were honest. If non-zero: **cheater detected**.

The loss value is publicly revealed after each step (a single scalar that doesn't leak weights), providing real-time training progress.

### Phase 3: Checkpoints

At configurable intervals, workers jointly compute a **Pedersen commitment** to current weights without reconstructing them:

1. Each worker `i` computes: `Ci = g^(si) * h^(ri)`
2. Combined commitment: `C = product(Ci) = g^W * h^r`
3. All workers sign an attestation: job ID, step range, commitment, loss, participants
4. One worker submits the attestation + signatures to `HelixCoordinatorV4`
5. Contract verifies: all registered workers signed, steps are sequential, data is valid

**Cost: ~50-80K gas** per checkpoint, compared to ~7.5M gas for a ZK proof verification.

### Phase 4: Cheater Detection & Slashing

When a MAC check fails:

1. **Immediate halt** -- no more computation on potentially corrupted state
2. **Pairwise identification** -- for each pair `(i, j)`, workers do a 2-party MAC cross-check:
   - Cheater fails with *every* honest party
   - Honest parties pass *all* their mutual checks
   - With 3 workers and 1 cheater: unambiguous identification
3. **Blame report** -- honest majority signs evidence, submits to contract
4. **On-chain slashing** -- cheater loses full stake (10% bounty to reporters, rest to treasury)
5. **Resume training** -- remaining honest workers redistribute the cheater's share from the last checkpoint and continue. No restart. No data loss.

### Phase 5: Completion

Final attestation goes on-chain. Workers encrypt their final shares to the owner's public key. The owner reconstructs `W_final = s1 + s2 + ... + sN` (the **only** time full weights exist outside of shares), verifies accuracy on a test set, and payment is released to workers proportional to their participation.

### Optional ZK Proof Layer

ZK proofs are **not required** for security -- SPDZ MACs already provide information-theoretic guarantees. But for users who want *external verifiability* (anyone can verify without trusting the worker set):

- `--zk-mode off` (default): Pure MPC + attestations. Fastest and cheapest.
- `--zk-mode always`: Lightweight StateTransitionCircuit proof at every checkpoint. Proves commitment transitions are valid without re-executing the full forward/backward pass. 1-3 seconds per proof.
- `--zk-mode risk`: ZK proofs auto-activate when workers drop below a threshold (e.g., after slashing). A safety net for questionable quorums.

Proofs are 1,856 bytes (SHPLONK, Keccak256Transcript), verified on-chain via the Halo2Verifier (BN254 pairing-based KZG).

---

## Live Demo

The `helix-demo` binary runs the complete pipeline end-to-end on MNIST (784 -> 32 -> 10 MLP, ~25K parameters):

```bash
# Core demo (~2-3 min): MPC training + cheater detection + on-chain settlement
cargo run -p helix-demo --features chain -- \
  --steps 200 --workers 3 --simulate-cheater

# With optional ZK proofs
cargo run -p helix-demo --features chain -- \
  --steps 200 --workers 3 --simulate-cheater --zk-mode always

# Risk-based ZK (auto-activates when cheater detected)
cargo run -p helix-demo --features chain -- \
  --steps 200 --workers 3 --simulate-cheater --zk-mode risk

# Real MNIST data (downloads on first run)
cargo run -p helix-demo -- \
  --steps 500 --workers 3 --real-mnist --simulate-cheater

# MPC-only (no on-chain settlement)
cargo run -p helix-demo -- --steps 200 --workers 3 --simulate-cheater
```

### What the Demo Shows

1. **Phase 1** -- Generate synthetic (or real) MNIST dataset, display model architecture (784 -> 32 -> 10, 25,824 parameters)
2. **Phase 2** -- 3 MPC workers initialized with secret-shared weights and pre-generated Beaver triples. Training progresses with a live progress bar showing step number, loss, active workers, and per-step MAC verification status
3. **Cheater injection** -- At the midpoint, Worker 2 begins submitting corrupted gradient shares
4. **Immediate detection** -- SPDZ MAC verification fails at the very next step. Pairwise identification runs. Worker 2 is unambiguously identified
5. **Recovery** -- Worker 2 is excluded. Training resumes with Workers 1 and 3 from the last checkpoint
6. **Phase 3** -- Final model weights reconstructed by summing shares. Accuracy evaluated on held-out test set. Per-class accuracy breakdown displayed
7. **Phase 4** -- Native (non-MPC) baseline trained for comparison, showing the MPC accuracy gap is minimal
8. **Phase 5** (with `--features chain`) -- Anvil node booted, HelixCoordinatorV4 deployed, workers stake and join, all checkpoints submitted on-chain with multi-party ECDSA signatures, training finalized on-chain
9. **Summary** -- Complete metrics: steps, throughput, loss curve, accuracy, MAC checks passed, Beaver triples consumed, gas used, security properties

The demo includes colored terminal output, ASCII loss curves, progress bars, and dramatic cheater detection alerts.

---

## Web Dashboard

A full Next.js dashboard provides a production-grade training interface:

### Pages

- **Marketplace** (home) -- Browse public models, filter by architecture/owner, view pricing and reputation scores
- **Portfolio Dashboard** -- Your models, revenue tracking, activity feed, accuracy charts
- **Train** -- Configure and launch MPC training jobs with real-time monitoring:
  - Model selection (from registry or upload new)
  - Worker pool assignment and configuration
  - Training parameters: steps, learning rate, batch size, checkpoint frequency
  - ZK mode toggle (Off / Always / Risk)
  - Cheater simulation flag for demos
  - Live loss/accuracy charts (Recharts), phase progression, MAC verification status
  - Cinematic cheater detection toast notifications with slashing details
  - Cost estimation panel: per-step fees, checkpoint gas, total budget
  - Pause / Stop / Resume controls
- **Inference** -- Run inference on trained models directly from the browser:
  - Draw digits on an HTML5 canvas or upload images
  - MPC-distributed inference with on-chain attestation of results
  - Probability distribution visualization
- **Network** -- Public worker directory with reputation scores, resource availability, and status
- **Models** -- Model detail pages with version history, owner controls, weight download, 0G storage integration
- **Settings** -- Wallet configuration, chain selection, trusted node management

### Key Features

- **Wallet integration** via wagmi/viem (MetaMask, WalletConnect)
- **On-chain model registry** with NFT minting for trained models
- **Encrypted weight storage** on 0G Network (decentralized storage)
- **Multi-wallet support** with session recovery
- **Real-time WebSocket** progress streaming from the Rust backend

```bash
cd helix/dashboard && npm install && npm run dev
```

---

## Technical Deep Dive

### MPC Primitives (implemented from scratch in Rust)

| Primitive | Implementation |
|-----------|---------------|
| **Secret sharing** | Additive shares over BN254 Fr field; weights split as `W = s1 + s2 + ... + sN` |
| **SPDZ MACs** | Global key `alpha` secret-shared; MAC tags `gamma(x) = alpha * x` verified via sigma protocol with commit-then-reveal |
| **Beaver triples** | Distributed generation via x25519 oblivious transfer (MASCOT-style); no trusted dealer |
| **Garbled circuits** | ReLU comparison via garbled circuits with x25519-based OT for label transfer |
| **Pedersen commitments** | BN254 G1 curve; joint commitment without weight reconstruction |
| **Cheater identification** | Pairwise MAC cross-checks; cheater fails with all honest parties |
| **Share redistribution** | Post-cheater recovery from checkpoint commitment + honest party shares |

### Smart Contract Design (HelixCoordinatorV4)

- **Per-job staking**: Workers stake collateral specific to each training job
- **Multi-party attestation**: Checkpoint verification via ECDSA signature validation from all active workers
- **Individual slashing**: MAC failure reports with majority signature trigger full stake slash; 10% bounty to reporters
- **Proportional payment**: Workers paid based on steps participated (pro-rata for early stops)
- **Global worker pool**: Workers register once with an endpoint and stake, get assigned to jobs on demand
- **Risk-based ZK**: `zkActivatedByRisk` flag flips when active workers drop below `minWorkersForMpc`; once activated, checkpoints require ZK proofs
- **Inference attestation**: On-chain commitment of MPC inference results with multi-party signatures
- **Lifecycle management**: Active -> Paused -> Resumed -> Completed/Stopped with correct payment handling at each transition

### Halo2 ZK Layer (optional)

- PSE halo2 fork 0.4.0 with KZG commitments on BN254
- **StateTransitionCircuit**: Proves `C_old -> C_new` transition is valid (commitment verification + weight delta bounds, not full forward/backward pass)
- EVM-compatible proofs: 1,856 bytes, SHPLONK scheme, Keccak256Transcript
- **LazyZkProver**: Defers SRS loading and keygen until first proof is actually needed -- zero overhead when ZK is disabled
- On-chain verification via BN254 pairing-based Halo2Verifier

---

## Performance

### MNIST (784 -> 32 -> 10, 3 Workers)

The timing and gas figures below are design targets from `SYSTEM_DESIGN.md`; the accuracy row is measured.

| Metric | Value |
|--------|-------|
| Native training step | 5-10ms |
| MPC training step | 50-120ms |
| **Overhead** | **10-15x** (vs 100-1000x for ZK-every-step) |
| SPDZ MAC verification | 2-5ms per step |
| Garbled circuit ReLU | 10-30ms per step |
| Checkpoint gas (attestation) | ~50-80K (~$0.10-0.50 on L2) |
| Checkpoint gas (ZK proof) | ~7.5M (optional) |
| 200-step training | ~30-60 seconds |
| Measured test accuracy (dashboard, 5,000 real MNIST samples, 500–1,000 steps) | 85–89% |

### Where the Overhead Comes From

The 10-15x overhead is dominated by **MPC communication** (Beaver triple openings and share exchanges), not by proof generation. This is fundamentally different from ZK-every-step approaches where the bottleneck is circuit proving. Communication overhead scales with network latency, not model complexity -- meaning HELIX's overhead ratio improves for larger models where computation dominates communication.

---

## Project Scale

| Component | Scale |
|-----------|-------|
| Solidity contracts | ~13,000 lines |
| Dashboard (TypeScript/React) | ~28,000 lines |
| Passing Rust tests | ~4,400+ |
| Passing contract tests | ~640+ |
| MPC sub-modules | 37 modules in helix-mpc |

### Project Structure

```
helix/
  crates/
    helix-core/         Types, tensors, error tracking, data pipeline
    helix-avm/          Training engine: forward, backward, autodiff, quantization
    helix-mpc/          MPC engine: 37 modules covering secret sharing, Beaver
                        triples, SPDZ MACs, garbled circuits, Pedersen commitments,
                        cheater identification, checkpoints, recovery, and more
    helix-circuits/     Halo2 circuits (StateTransitionCircuit for optional ZK)
    helix-prover/       KZG proof pipeline (optional, only when ZK enabled)
    helix-node/         P2P networking, gossip protocol, session coordination
    helix-client/       Owner SDK, orchestration, dashboard HTTP/WebSocket API
    helix-demo/         Demo binary: runner, display, evaluator, risk, ZK prover
  contracts/
    src/core/           HelixCoordinatorV4 (MPC coordinator, attestation, slashing)
    src/token/          HelixToken (ERC20), Staking, Rewards
    src/verification/   Halo2Verifier (BN254 KZG), PoseidonHasher
    src/governance/     TrainingDAO
  dashboard/            Next.js frontend: marketplace, training, inference, network
  docs/                 SYSTEM_DESIGN.md, IMPLEMENTATION_STAGES.md
```

---

## Getting Started

### Prerequisites

- Rust 1.75+
- Node.js 18+
- Foundry (for smart contracts)

### Build

```bash
cd helix

# Build all Rust crates
cargo build --workspace

# Build contracts
cd contracts && forge build && cd ..

# Build dashboard
cd dashboard && npm install && npm run build
```

### Test

```bash
# All Rust tests (~4,400)
cargo test --workspace

# Contract tests (~640)
cd contracts && forge test

# Specific crate
cargo test -p helix-mpc
cargo test -p helix-demo
```

### Run Demo

```bash
# Quick demo: MPC training + cheater simulation
cargo run -p helix-demo -- --steps 200 --workers 3 --simulate-cheater

# Full demo with on-chain settlement
cargo run -p helix-demo --features chain -- --steps 200 --workers 3 --simulate-cheater

# Dashboard
cd dashboard && npm run dev
```

---

## What Makes HELIX Different

**1. Information-theoretic security.** SPDZ MACs provide mathematical impossibility of undetected cheating -- not computational hardness, not hardware trust, not probabilistic sampling. This is strictly stronger than any ZK proof system, any TEE, any spot-check mechanism.

**2. Individual cheater identification.** When a MAC check fails, pairwise verification pinpoints the exact bad actor. No group punishment. No ambiguity. The cheater is slashed; honest workers continue.

**3. Zero weight leakage.** Model weights exist only as additive secret shares during the entire training process. No worker, no coordinator, no smart contract ever sees the full model. Only the owner reconstructs at the very end.

**4. Self-healing.** Detect the cheater, remove them, redistribute shares from checkpoint, keep training. No restart. No data loss. Graceful degradation as long as honest majority holds.

**5. Practical efficiency.** 5-15x overhead instead of 100-1000x. The bottleneck is MPC communication, not proof generation. Weight updates require zero communication (additive sharing is linear).

**6. Minimal on-chain cost.** Multi-party attestation checkpoints at ~50-80K gas vs ~7.5M gas for ZK proof verification. Orders of magnitude cheaper, enabling frequent checkpoints without breaking the bank.

**7. ZK as a feature, not a dependency.** The system is fully secure without any ZK proofs. ZK is available as an optional add-on for external third-party verifiability -- a nice feature for some use cases, but never a security requirement.

---

## License

MIT
