# config/ Module Review

**Score: B (78%)**
**Verdict:** Comprehensive configuration system with profiles, validation, and TOML persistence. Well-designed for both demo and production use. Missing config migration and some validation edge cases.

---

## Architecture

```
config/
  mod.rs        # HelixConfig + profiles + validation (1,069 lines)
  training.rs   # TrainingJobConfig + model architectures (937 lines)
```

Total: ~2,006 lines across 2 files.

---

## Per-File Analysis

### mod.rs — HelixConfig (1,069 lines)

**Network Profiles:**
- `Local`: localhost RPC (127.0.0.1:8545), no staking, test mode
- `Anvil`: Anvil RPC (127.0.0.1:9545), chain ID 31337, minimal staking
- `Sepolia`: Sepolia testnet RPC, chain ID 11155111, real staking thresholds
- `Mainnet`: Ethereum mainnet (placeholder), chain ID 1, production staking
- `Custom`: User-defined values

**Sub-Configs:**
- `NodeConfig`: node_id, host, port, role (Worker/Aggregator/Validator/Observer)
- `NetworkConfig`: bootstrap_peers, max_peers, gossip_interval, discovery_interval
- `TrainingConfig`: max_rounds, batch_size, learning_rate, optimizer, error_budget
- `RpcConfig`: endpoint, timeout, max_retries, retry_delay, auth_token
- `ContractConfig`: coordinator_address, verifier_address, token_address, staking_address
- `StakingConfig`: min_stake, lock_period, slash_percent, auto_restake
- `ProofConfig`: circuit_k, verification_mode, timeout
- `StorageConfig`: data_dir, max_cache_size, checkpoint_interval
- `TelemetryConfig`: enabled, metrics_port, log_level

**Features:**
- TOML load/save (`load()`, `save()`, `load_or_create()`)
- `ProfileManager` with caching (avoids re-reading TOML on each access)
- Validation: checks learning_rate > 0, batch_size > 0, valid profile, etc.
- `from_profile()` constructor for quick setup
- Supports config file path override

### training.rs — TrainingJobConfig (937 lines)

**Model Architectures:**
- MLP (multi-layer perceptron)
- Transformer
- BERT
- CNN (convolutional)
- RNN (recurrent)
- Custom

**Hyperparameters:**
- Optimizer: SGD, Adam, AdamW, RMSProp, LAMB
- LR Scheduler: Constant, StepDecay, CosineAnnealing, WarmupCosine, OneCycle, ReduceOnPlateau
- Early stopping: patience, min_delta, metric
- Gradient clipping: max_norm or max_value

**Data Config:**
- Source: local file, IPFS, S3, HTTP
- Format: CSV, Parquet, Binary, Custom
- Train/val/test splits (default 80/10/10)
- Batch size, shuffle, num_workers

**Proof Settings:**
- Circuit K value
- Verification mode
- Freivalds checks
- Error budget

**Demo Presets:**
- `demo_mlp()`: small MLP, 10 rounds, batch 32
- `demo_transformer()`: small transformer, 5 rounds, batch 16

**Features:**
- TOML load/save
- `generate_example_toml()` for config file creation
- Validation (rejects batch_size=0, negative learning_rate, etc.)
- Checkpoint config (interval, path, max_kept)
- Resource limits (max_memory_mb, max_gpu_memory_mb, max_time_seconds)

---

## Strengths

1. **Profile system** — 4 pre-defined profiles cover the full deployment spectrum (local dev -> Anvil testing -> Sepolia testnet -> mainnet). `Custom` allows full override. This makes onboarding easy.

2. **Comprehensive sub-configs** — Every major subsystem (node, network, training, RPC, contracts, staking, proof, storage, telemetry) has its own config struct. Clean separation of concerns.

3. **TOML persistence** — Human-readable config files that users can edit. `load_or_create()` generates defaults on first run. This is the right choice for CLI tools.

4. **Training job flexibility** — 6 model architectures, 6 optimizers, 6 LR schedulers, early stopping, gradient clipping, checkpoint management. Covers real training scenarios.

5. **Demo presets** — `demo_mlp()` and `demo_transformer()` provide zero-config starting points for demos. Reduces friction.

6. **Config validation** — Catches common errors (negative learning rate, zero batch size, invalid profile) before they cause runtime failures.

---

## Weaknesses

### HIGH

**No config versioning or migration**
- Config struct changes (new fields, renamed fields, removed fields) break existing TOML files
- Users with saved configs from earlier versions will get deserialization errors
- **Fix:** Add `version: u32` field. Implement migration functions for each version increment. Use `serde(default)` for new optional fields.

**Sepolia and Mainnet contract addresses are empty/placeholder**
- `ContractConfig` for non-Anvil profiles has empty addresses
- **Impact:** Switching to Sepolia/Mainnet profile fails without manual config
- **Fix:** Maintain a registry of deployed contract addresses per network. Load from embedded JSON or remote registry.

### MEDIUM

**`ProfileManager` cache has no expiry**
- Once a profile is cached, it's never invalidated
- If the TOML file changes on disk, `ProfileManager` returns stale config
- **Fix:** Add file modification time check, or TTL-based cache expiry.

**No environment variable override**
- Common pattern: `HELIX_RPC_ENDPOINT=http://... helix train` to override config without editing files
- **Fix:** Add `from_env()` method that overlays environment variables on TOML config. Or use `config` crate.

**`TrainingConfig` defaults are demo-oriented**
- `max_rounds: 100`, `learning_rate: 0.001`, `error_budget: 1000.0`
- For production training these may be inappropriate
- **Fix:** Tie defaults to profile (demo profiles use small values, production uses conservative values).

### NICE-TO-HAVE

- No config diff/merge tool (useful when upgrading)
- No config validation warnings (only errors) — could warn about suboptimal settings
- `generate_example_toml()` could include comments explaining each field
- No schema documentation (config fields not documented in help system)

---

## Test Coverage

**Tested (from `tests/demo_integration.rs`):**
- `TrainingJobConfig::demo_mode()` validates successfully
- `TrainingJobConfig::strict()` with batch_size=0 rejected
- `TrainingJobConfig` with learning_rate < 0 rejected
- `HelixConfig::from_profile(Local)` creates valid config

**Gaps:**
- No test for TOML round-trip (save then load produces same config)
- No test for profile-specific defaults (Anvil chain ID = 31337)
- No test for `ProfileManager` caching behavior
- No test for `load_or_create()` creating defaults
- No test for training job TOML parsing with all field types
