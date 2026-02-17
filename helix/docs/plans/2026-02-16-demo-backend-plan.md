# ETHDenver Demo Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Get the HELIX backend fully working end-to-end: user submits training data from web dashboard → MPC training with real secret sharing → model returned, all on-chain.

**Architecture:** Dashboard (Next.js :3000) → Rust API server (Axum :3001) → FullOrchestrator spawns MPC parties as tokio tasks with TCP transport → V4 contract on Anvil (:8545). Workers are real MPC parties communicating over real TCP with real SPDZ MACs and Beaver triples. The user does zero computation.

**Tech Stack:** Rust (tokio, axum, helix-mpc, ethers), Solidity (Foundry), Next.js (TypeScript, Wagmi, RainbowKit, Recharts)

---

## Critical Bugs Found During Analysis

1. **`network-mpc` feature disabled by default** — TCP transport never activates; all MPC training falls back to LocalTransport (in-memory channels). The training still works but workers aren't using real TCP.
2. **Dashboard port stride bug** — `dashboard.rs:955` generates worker ports as `9001, 9002, 9003` but `spawn-workers` uses stride-2 ports `9001, 9003, 9005` (data + control channels). Mismatch if ever combined.
3. **Anvil private key empty** — `dashboard.rs:982` sets `private_key: String::new()` for chain config. The orchestrator may fail if it tries to use an empty private key before the Anvil-default-key fallback kicks in.

---

### Task 1: Enable TCP Transport and Fix Config Bugs

**Files:**
- Modify: `helix/crates/helix-client/Cargo.toml` (line 116)
- Modify: `helix/crates/helix-client/src/dashboard.rs` (lines 955-957, 982)

**Step 1: Enable network-mpc in default features**

In `helix/crates/helix-client/Cargo.toml`, change:
```toml
default = ["chain"]
```
to:
```toml
default = ["chain", "network-mpc"]
```

This makes `cfg!(feature = "network-mpc")` true in `full_orchestration.rs:549`, enabling TCP transport when worker endpoints are configured.

**Step 2: Fix dashboard port stride to match spawn-workers convention**

In `dashboard.rs`, change the worker endpoint generation (line 955-957):
```rust
// Generate worker endpoints (in-process workers on localhost)
let worker_endpoints: Vec<String> = (0..req.num_workers)
    .map(|i| format!("127.0.0.1:{}", 9001 + i))
    .collect();
```
to:
```rust
// Generate worker endpoints with stride-2 ports (data channel on even, control on +1)
let worker_endpoints: Vec<String> = (0..req.num_workers)
    .map(|i| format!("127.0.0.1:{}", 9001 + (i as u16) * 2))
    .collect();
```

**Step 3: Set Anvil default owner key in dashboard**

In `dashboard.rs`, line 982 (`private_key: String::new()`) should already be overridden by the block at lines 998-1018 which sets it to Anvil account 0. Verify this actually runs by checking that the `#[cfg(feature = "chain")]` block at 998 is reached. If `private_key` is empty when it shouldn't be, fix the initialization order.

**Step 4: Compile check**

Run: `cargo check -p helix-client --features "chain,network-mpc"`
Expected: Compiles without errors (warnings OK)

**Step 5: Commit**

```bash
git add helix/crates/helix-client/Cargo.toml helix/crates/helix-client/src/dashboard.rs
git commit -m "fix: enable TCP transport by default, fix dashboard port stride"
```

---

### Task 2: Verify Full Compilation and Fix Any Errors

**Files:**
- Potentially any file in helix-client, helix-mpc

**Step 1: Full workspace compile**

Run: `cargo build -p helix-client`
Expected: Successful build (warnings OK)

If there are errors, fix them one by one. Common issues:
- Missing imports when `network-mpc` is enabled
- Type mismatches in TCP transport code paths
- Feature-gate issues

**Step 2: Run helix-client unit tests**

Run: `cargo test -p helix-client --lib -- --test-threads=4`
Expected: Tests pass (some chain-related tests may fail without Anvil, that's OK)

**Step 3: Commit any fixes**

---

### Task 3: Smoke Test the Dashboard API Server

**Step 1: Start the dashboard server**

Run: `cargo run -p helix-client -- dashboard --port 3001 --host 0.0.0.0 --cors`
Expected: Server starts, prints "Dashboard is running" message

**Step 2: Test health endpoint**

Run (in another terminal): `curl http://localhost:3001/health`
Expected: 200 OK response

**Step 3: Test training/start endpoint with minimal request**

Run:
```bash
curl -X POST http://localhost:3001/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "architecture": [784, 32, 10],
    "num_workers": 3,
    "num_steps": 50,
    "learning_rate": 0.01,
    "checkpoint_freq": 25,
    "mac_interval": 5,
    "zk_mode": "off",
    "zk_checkpoint_freq": 0,
    "min_workers_for_mpc": 2,
    "train_size": 100,
    "test_size": 50,
    "use_real_mnist": false,
    "payment_eth": 0.1,
    "stake_per_worker_eth": 0.05,
    "simulate_cheater": false,
    "seed": 42
  }'
```
Expected: 200 with `{"session_id": "...", "status": "starting"}`

**Step 4: Poll session status**

Run: `curl http://localhost:3001/api/training/sessions`
Expected: JSON array with session, status progressing through phases

**Step 5: Watch server logs for training progress**

Expected: See Phase 1-13 progress in server output. Training should complete with accuracy > 0.

**Step 6: Check model download**

Run: `curl http://localhost:3001/api/training/sessions/<SESSION_ID>/model`
Expected: JSON with `w1`, `b1`, `w2`, `b2` weight arrays

**Step 7: Fix any issues found**

Common issues to look for:
- Anvil not starting (check if `anvil` is in PATH)
- Contract deployment failures (contract size limits, gas)
- MPC training errors (transport binding failures)
- Missing progress events (WebSocket not broadcasting)

---

### Task 4: Verify On-Chain Lifecycle

**Step 1: Start Anvil manually**

Run: `anvil --accounts 20`
Expected: Anvil running on localhost:8545 with 20 funded accounts

**Step 2: Run mpc-train with explicit Anvil**

Run:
```bash
cargo run -p helix-client -- mpc-train \
  --architecture 784,32,10 \
  --steps 50 \
  --checkpoint-freq 25 \
  --rpc-url http://localhost:8545 \
  --seed 42
```

Expected: Full 13-phase pipeline executes:
- Phase 3: Chain bootstrap (connects to Anvil)
- Phase 4: V4 contract deployed
- Phase 5: Job registered on-chain
- Phase 6: Workers staked
- Phase 8: MPC training completes
- Phase 9: Checkpoints submitted on-chain
- Phase 11: Training finalized on-chain
- Phase 12: Accuracy evaluation
- Final weights saved

**Step 3: Verify contract state**

After training completes, use `cast` to check contract state:
```bash
cast call <COORDINATOR_ADDRESS> "getJob(uint256)(address,bytes32,uint256,uint256,uint256,uint256,uint256,bytes32,uint256,bool,bool)" 1 --rpc-url http://localhost:8545
```

**Step 4: Fix any issues found**

---

### Task 5: Verify ZK Proof Integration

**Step 1: Run mpc-train with ZK enabled**

Run:
```bash
cargo run -p helix-client -- mpc-train \
  --architecture 784,32,10 \
  --steps 50 \
  --checkpoint-freq 25 \
  --zk-mode always \
  --zk-proofs \
  --seed 42
```

Expected: Same as Task 4 but with ZK proof events:
- "ZK proof: generating for checkpoint N..."
- "ZK proof: step N | M bytes | Xms | verified=true"

**Step 2: Run with risk-based ZK**

Run:
```bash
cargo run -p helix-client -- mpc-train \
  --architecture 784,32,10 \
  --steps 50 \
  --checkpoint-freq 25 \
  --zk-mode risk \
  --simulate-cheater \
  --seed 42
```

Expected: ZK activates when cheater is detected and worker count drops

**Step 3: Fix any ZK issues**

Common issues:
- SRS loading failure (check halo2 setup)
- Proof generation timeout
- StateTransitionCircuit errors

---

### Task 6: Wire Next.js Dashboard to Backend

**Files:**
- Modify: `helix/dashboard/.env.local` (create if needed)
- Potentially modify: `helix/dashboard/src/lib/api.ts`
- Potentially modify: `helix/dashboard/src/app/train/page.tsx`

**Step 1: Create/verify .env.local**

Create `helix/dashboard/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:3001/api
NEXT_PUBLIC_WS_URL=ws://localhost:3001/ws
NEXT_PUBLIC_ENABLE_TESTNETS=true
```

**Step 2: Start the Next.js dashboard**

Run: `cd helix/dashboard && npm install && npm run dev`
Expected: Dashboard loads on localhost:3000

**Step 3: Navigate to /train page**

Open http://localhost:3000/train in browser.
Expected: Training configuration form appears with:
- Architecture (fixed MNIST)
- Worker count slider
- Steps input
- ZK mode selector (Off / Always / Risk)
- Start Training button

**Step 4: Start a training session from the UI**

With the Rust dashboard server running on :3001:
1. Configure training (3 workers, 50 steps, ZK off)
2. Click "Start Training"
Expected: Session starts, live progress appears

**Step 5: Verify WebSocket real-time updates**

Expected: Loss curve updates in real-time, step counter progresses, phases shown

**Step 6: Verify model download**

When training completes:
Expected: Download button appears, clicking downloads the trained model weights as JSON

**Step 7: Fix any frontend-backend integration issues**

Common issues:
- CORS errors (ensure --cors flag on dashboard server)
- API URL mismatch (check .env.local)
- WebSocket connection failures
- JSON field name mismatches between Rust and TypeScript types
- Missing fields in API responses that the frontend expects

---

### Task 7: Test Complete Demo Flow End-to-End

**Step 1: Start all services**

Terminal 1: `anvil --accounts 20`
Terminal 2: `cargo run -p helix-client -- dashboard --port 3001 --host 0.0.0.0 --cors`
Terminal 3: `cd helix/dashboard && npm run dev`

**Step 2: Run the full demo**

1. Open http://localhost:3000/train
2. Upload training data OR use built-in MNIST
3. Configure: 3 workers, 100 steps, ZK off, checkpoint every 25 steps
4. Start training
5. Watch real-time loss curve
6. When complete, download model
7. Verify model has reasonable weights (not zeros/NaN)

**Step 3: Test with cheater simulation**

Repeat with "Simulate Cheater" toggle ON.
Expected: Cheater detected mid-training, slashed on-chain, training continues

**Step 4: Test ZK modes**

Repeat with ZK mode set to "Always".
Expected: ZK proofs generated at each checkpoint, shown in UI

Repeat with ZK mode set to "Risk-Based".
Expected: ZK activates only if conditions trigger it

**Step 5: Document any remaining issues and fix them**

---

### Task 8: Polish and Final Verification

**Step 1: Verify weight privacy**

Confirm in the code/logs that:
- Workers only ever receive encrypted shares (x25519 + AES-256-GCM)
- No worker ever has access to full model weights
- Only the model owner (orchestrator) reconstructs final weights
- Reconstruction happens AFTER training completes, not during

**Step 2: Verify on-chain completeness**

Check that ALL lifecycle events are on-chain:
- Job registration (registerTrainingJob)
- Worker staking (stakeAndJoin)
- Checkpoint attestations (submitCheckpoint with signatures)
- MAC failure reports if cheater (reportMACFailure)
- Training completion (completeTraining)
- Payment distribution

**Step 3: Clean up demo output**

Ensure terminal output is clean and impressive:
- Clear phase indicators
- Loss curve in terminal (for CLI demo)
- Colored status messages
- Final summary with accuracy, time, gas used

**Step 4: Final commit**

Commit all fixes with a clean message summarizing what was done.
