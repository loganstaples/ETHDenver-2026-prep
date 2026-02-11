# rpc/ Module Review

**Score: A- (88%)**
**Verdict:** Solid RPC layer with real JSON-RPC, on-chain integration, and circuit breakers. TrainingProofInputs aligned to 8-element V2 contract. Added `get_model()` and `get_round()` forwarding to UnifiedRpcClient.

---

## Architecture

```
rpc/
  mod.rs      # Re-exports (feature-gated chain module)
  client.rs   # HelixRpcClient + MockRpcClient + UnifiedRpcClient (~2,000 lines)
  chain.rs    # ChainClient — ethers-rs contract wrapper (761 lines)
```

Total: ~2,800 lines across 3 files.

---

## Per-File Analysis

### mod.rs — Re-exports
Feature-gated re-export of `chain` module (only when `chain` feature enabled). Clean separation of JSON-RPC concerns (client.rs) from on-chain concerns (chain.rs).

### client.rs — RPC Clients (~2,000 lines)

**HelixRpcClient (Real)**
- JSON-RPC 2.0 over HTTP via reqwest
- ~20 RPC methods: health, training, proofs, models, rounds, workers, staking
- Exponential backoff with jitter: base_delay * 2^attempt (capped at max_backoff) + random 0-25%
- Auth token support (Bearer header)
- Circuit breaker: Closed (normal) -> Open (after N failures, rejects calls) -> HalfOpen (after timeout, tests one call) -> Closed (on success)
- `connect()` performs health check and caches capabilities
- `subscribe_*()` methods use polling with `tokio::select!` + watch channels for shutdown

**MockRpcClient (Simulated)**
- Deterministic training simulation with randomized loss/error values
- `init_workers(count)` creates dummy workers with Ethereum-like addresses
- `advance_round()`: decreases loss by 8-12%, increases error by 3-5%
- `slash_worker()`, `fail_worker()`, `recover_worker()` for demo scenarios
- All state behind `Arc<RwLock<T>>` for thread safety

**UnifiedRpcClient (Wrapper)**
- `connect(config)`: strict mode — fails if real node unavailable
- `connect_or_mock(config)`: fallback mode (debug/demo only)
- `mock_only()`: pure mock mode
- Routes all methods to real or mock client
- Stores optional `ChainClient` for on-chain operations (behind `chain` feature)

**Real-Time Trackers**
- `RealTimeProofTracker`: proof status with callback support, last update time
- `RealTimeTrainingTracker`: training status with loss/error history, ETA estimation

### chain.rs — On-Chain Contract Wrapper (761 lines)

**ChainClient**
- Wraps `HelixCoordinatorV2` via `abigen!` (compile-time ABI generation)
- `LocalWallet` + `Provider<Http>` + `SignerMiddleware` for auto-signing
- Circuit breaker on every contract call (check before, record success/failure after)
- Methods: register_model, get_model_state, start_round, get_round_state, stake, unstake, get_stake, submit_proof, submit_proof_raw, query_proof_submitted, query_round_completed
- `deploy_with_forge()`: runs `forge script`, parses broadcast JSON for contract addresses

**Data Types**
- `TrainingProofInputs`: 8 public inputs (old_hash_lo/hi, new_hash_lo/hi, loss, error_bound, step_number, error_checksum)
- `ChainModelState`: current_round, current_commitment, active
- `ChainRoundState`: model_commitment, new_commitment, is_completed, deadline, prover
- `ChainStakeInfo`: amount, locked_until, slashed
- `ForgeDeployResult`: all deployed contract addresses

---

## Strengths

1. **Circuit breaker pattern** — Both `HelixRpcClient` and `ChainClient` implement proper circuit breakers. Prevents cascading failures when nodes go down. State machine transitions are correct (Closed -> Open on N failures, Open -> HalfOpen after timeout, HalfOpen -> Closed on success or Open on failure).

2. **Exponential backoff with jitter** — Prevents thundering herd on retries. Jitter (0-25% of capped delay) distributes retry timing.

3. **Real contract interaction** — `ChainClient` sends actual `eth_sendTransaction` via ethers-rs. Model registration, staking, proof submission, event querying all work against deployed contracts.

4. **`deploy_with_forge()` is genuinely useful** — Parses forge broadcast JSON to extract contract addresses. Handles both V2 and V3 coordinators, MockVerifier and Halo2Verifier. Returns typed `ForgeDeployResult`.

5. **Clean mock/real separation** — `UnifiedRpcClient` provides transparent routing. Library users don't need to know whether they're hitting a real node or mock.

---

## Weaknesses

### HIGH

~~`TrainingProofInputs.to_vec()` returns 7 elements; V2 contract expects 8~~ **RESOLVED**
- Added `error_checksum: U256` field. `to_vec()` returns 8 elements. `from_bytes()` updated. Tests verified.

**Hardcoded chain ID 31337 in broadcast path** (`chain.rs:666-669`)
- `deploy_with_forge()` reads from `broadcast/Deploy.s.sol/31337/run-latest.json`
- Fails on any non-Anvil chain (e.g., Sepolia = 11155111, Mainnet = 1)
- **Fix:** Accept chain_id parameter, or detect from provider via `eth_chainId`.

**Model ID parsing from event logs is fragile** (`chain.rs:register_model`)
- Assumes model ID is in `log.topics[1]`
- Falls back to 0 if missing (silent incorrect result)
- **Fix:** Use ethers `decode_log` with the ModelRegistered event ABI for type-safe parsing.

### MEDIUM

**Mock `advance_round()` unbounded error accumulation**
- Error grows by 3-5% per round without limit
- Production should halt at `max_error_bound`
- **Fix:** Add max check and return false when exceeded.

**`TrainingPhase` and `ProofPhase` have no transition validation**
- Any phase can follow any other (e.g., `RoundComplete` -> `Initializing`)
- **Fix:** Add `can_transition_to()` method or use a state machine crate.

**Polling subscriptions lack backpressure**
- `subscribe_training_status()` polls at fixed interval
- If callback is slow, polls accumulate
- **Fix:** Skip poll if previous callback hasn't completed.

### NICE-TO-HAVE

- No gas estimation on contract calls
- No nonce management (relies on provider default)
- MockRpcClient uses `rand::random()` which isn't deterministic in tests — could use seeded RNG
- `HelixRpcConfig` defaults to `127.0.0.1:9545` — should match Anvil's actual default port (8545)

---

## Test Coverage

- `TrainingProofInputs::to_vec()` — unit test verifies 8-element output (including error_checksum)
- `TrainingProofInputs::from_bytes()` — unit test verifies field mapping (including error_checksum)
- Circuit breaker state transitions — tested in `tests/demo_integration.rs` and `tests/e2e_chain_integration.rs`
- Full on-chain lifecycle — 15+ tests in `e2e_chain_integration.rs` (register, stake, submit, events)

**Gap:** No test verifies that `submit_proof()` (typed API) correctly constructs the contract call. Only `submit_proof_raw()` is exercised in e2e tests.
