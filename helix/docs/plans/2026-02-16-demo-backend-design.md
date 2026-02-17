# ETHDenver Demo Backend - Design Document

**Date**: 2026-02-16
**Goal**: Fully working end-to-end backend for live ETHDenver demo
**Constraint**: One day to complete. Nothing fake.

## Architecture

```
Browser (localhost:3000)
    ↕ HTTP/WS
Rust API Server - dashboard.rs (localhost:3001)
    ↕ spawns FullOrchestrator
    ↕ TCP connections to workers
Worker 0 (:9001/9002) | Worker 1 (:9003/9004) | Worker 2 (:9005/9006)
    ↕ MPC protocol over TCP
Anvil local Ethereum (localhost:8545)
    ↕ V4 contract interactions
```

## Priority Order

1. **Training works end-to-end**: Submit data → MPC train → get model back
2. **Weight privacy**: Workers only see secret shares, never full weights
3. **On-chain**: All lifecycle events on V4 contract via local Anvil
4. **ZK works**: Configurable optional proofs (off/always/risk-based)

## Demo Flow

1. Start Anvil, deploy V4 contracts
2. Spawn 3 worker processes (separate processes, one command)
3. Start Rust API server (dashboard)
4. Start Next.js dashboard
5. User opens browser, uploads training data
6. Clicks "Start Training" with config (workers, steps, ZK mode)
7. Watches real-time loss curve via WebSocket
8. Can toggle ZK mode from UI
9. Sees on-chain events (staking, checkpoints, slashing)
10. Downloads trained model when complete

## Key Requirements

- User does ZERO computation (browser only)
- Workers NEVER see full weights (additive secret sharing)
- Only the end user reconstructs the trained model
- Multiple workers run as separate OS processes from one terminal command
- Everything on-chain via local Anvil
- Real MNIST training (784→32→10, 25K params)

## Components (Existing → Verified)

| Component | File | Action |
|-----------|------|--------|
| API Server | dashboard.rs | Verify endpoints, fix training flow |
| Orchestrator | full_orchestration.rs | Verify TCP worker support |
| Worker Entry | worker_entry.rs | Verify TCP data+control channels |
| Spawn Workers | main.rs (spawn-workers cmd) | Verify process spawning |
| ZK Layer | zk_proof_layer.rs | Verify lazy init works |
| V4 Contract | HelixCoordinatorV4.sol | Verify deploy + lifecycle |
| Dashboard UI | dashboard/src/ | Wire to backend, verify forms |

## Approach

Verify & Fix: Smoke-test each component, fix what's broken, add what's missing.
