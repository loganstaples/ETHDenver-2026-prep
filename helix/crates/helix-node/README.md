# helix-node

Complete node software for participating in the HELIX decentralized ML training network. Implements P2P networking, distributed training coordination with BFT consensus, ZK proof generation/verification, and on-chain proof submission.

## Purpose

Full node implementation supporting three roles:

- **Compute Node** — Execute ML training steps, generate Halo2 KZG proofs, submit gradient shares
- **Aggregator Node** — Collect gradients, verify proofs, run Byzantine-tolerant aggregation, submit to chain
- **Verifier Node** — Spot-check proofs, maintain network integrity via verification policies

## Architecture

```
helix-node/
├── src/
│   ├── main.rs              Binary entry point (worker/aggregator modes)
│   ├── config.rs             NodeConfig with validation, JSON load/save
│   ├── identity.rs           ed25519 identity with persistent keypair
│   ├── trainer.rs            2-layer MLP training + ZK proof generation
│   ├── sc_client.rs          HelixCoordinatorV2 ethers bindings
│   ├── round_commit.rs       Proof collection → Merkle aggregation → on-chain submit
│   ├── network/              P2P networking stack (~7,500 lines)
│   │   ├── runner.rs         Network orchestrator with reputation integration
│   │   ├── transport.rs      TCP/TLS with connection pooling (max 256 peers)
│   │   ├── wire.rs           HELX magic, CRC32, bincode protocol (frame size enforced)
│   │   ├── gossip.rs         Epidemic gossip with LRU dedup
│   │   ├── discovery.rs      Bootstrap peer discovery
│   │   ├── reputation.rs     Multi-dimensional peer scoring with decay/banning
│   │   ├── sybil.rs          Stake-weighted Sybil resistance (on-chain verified)
│   │   ├── rate_limit.rs     Token bucket + auto-blacklisting
│   │   ├── eclipse.rs        Eclipse attack prevention
│   │   └── partition_detect.rs  Network partition detection
│   ├── training/             Distributed training coordination (~9,000 lines)
│   │   ├── orchestrator.rs   Multi-node training round lifecycle
│   │   ├── consensus.rs      BFT 2-phase commit with hiding commitments
│   │   ├── verification.rs   Halo2 KZG proof verification (3 policies)
│   │   ├── aggregation.rs    6 Byzantine-tolerant aggregation strategies
│   │   └── ...               Round state, checkpoints, fault tolerance
│   ├── roles/                Compute, Aggregator, Verifier state machines
│   ├── api/                  HTTP REST (Axum) + JSON-RPC (13 methods)
│   ├── storage/              Local (atomic writes) + IPFS backends
│   └── data/                 Merkle-verified data loading + on-chain commitments
└── tests/                    25 integration tests (multi-worker, distributed, Halo2)
```

## Key Data Flow

1. **Startup**: Load config → create/load persistent identity → connect to chain → start P2P stack
2. **Worker mode**: Connect to aggregator → send heartbeats → on `RoundStart`: train → prove → submit gradient
3. **Aggregator mode**: Start round → broadcast to workers → collect gradients + proofs → BFT consensus (commit/reveal) → aggregate (Krum) → verify proofs → submit to chain
4. **On-chain**: `HelixCoordinatorV2` → `Halo2Verifier.verify()` → accept or slash

## Configuration

```json
{
  "listen_addr": "0.0.0.0:9000",
  "data_dir": "./helix-data",
  "role": "Compute",
  "rpc_url": "http://localhost:8545",
  "bootstrap_peers": ["192.168.1.10:9000"],
  "api": { "enabled": true, "port": 3000, "auth_token": "..." }
}
```

Private keys are loaded from the `HELIX_PRIVATE_KEY` environment variable (never stored in config files).

## Running a Node

```bash
# Compute worker
helix-node --role compute --config node.toml

# Aggregator
helix-node --role aggregator --config node.toml
```

## API

### HTTP REST (Axum)

- `GET /health` — Node health status (no auth)
- `GET /metrics` — Prometheus metrics (no auth)
- `GET /api/round/status` — Current round info (Bearer auth)
- `POST /api/round/start` — Trigger new round (Bearer auth)
- `GET /api/peers` — Connected peer list (Bearer auth)

### JSON-RPC

13 methods including `helix_getStatus`, `helix_startRound`, `helix_submitGradient`, `helix_getMetrics`, etc.

## Security Features

- **Message signing**: ed25519 signatures on all P2P messages (with `crypto-sign` feature)
- **Reputation system**: Multi-dimensional scoring (Responsiveness, Validity, Bandwidth, Uptime) with decay and banning — wired into the network receive loop
- **Sybil resistance**: Stake-weighted peer selection with on-chain stake verification
- **Rate limiting**: Per-peer token bucket with automatic blacklisting
- **Eclipse prevention**: Diverse peer selection across network regions
- **Partition detection**: Graduated response to network splits
- **Hiding commitments**: BFT consensus uses nonce-blinded SHA-256 commitments
- **Frame size limits**: Wire protocol enforces `MAX_MESSAGE_SIZE` on both encode and decode
- **Connection pool limits**: Configurable max peers (default 256) prevents resource exhaustion
- **Proof verification cache**: Binds proof bytes + public inputs + round ID + error bound

## Testing

```bash
cargo test -p helix-node              # Run all tests (~374)
cargo test -p helix-node -- --test-threads=1  # Sequential (for integration tests)
```

**374 tests** across unit and integration tests covering:
- Network security modules (gossip, rate limiting, reputation, Sybil, eclipse, partition)
- BFT consensus protocol
- Proof verification (all 3 policies)
- Training orchestrator lifecycle (17 tests)
- Gradient aggregation (12 tests, all 6 strategies)
- HTTP API (10 tests: auth, rate limiting, endpoints)
- Role state machines (30 tests)
- JSON-RPC server (30 tests)
- Integration tests (25 tests: multi-worker, distributed training, Halo2 verification)

## Build

```bash
cargo build -p helix-node                          # Debug build
cargo build -p helix-node --release                # Release (faster proofs)
cargo build -p helix-node --features crypto-sign   # With ed25519 signing
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `tokio` | Async runtime |
| `ethers` | Ethereum interaction (abigen) |
| `tokio-rustls` | TLS transport |
| `ed25519-dalek` | Message signing |
| `axum` | HTTP API server |
| `bincode` | Wire protocol serialization |
| `sha2` | Merkle trees, commitments |
| `helix-core` | Tensors, error tracking |
| `helix-prover` | Halo2 proof generation/verification |
| `helix-mpc` | Multi-party computation bridge |
