# helix-node/src/network/ — Technical Review

## Overview

The `network/` module implements a complete P2P networking stack for the HELIX decentralized training network. It comprises 13 submodules handling transport, gossip propagation, peer discovery, wire protocol, and a comprehensive security suite (rate limiting, Sybil resistance, eclipse prevention, partition detection, reputation management). This is the largest module in `helix-node` at ~7,500+ lines across 14 files.

## Architecture

### Module Tree

```
network/
├── mod.rs               (65 lines)   Re-exports all submodules
├── messages.rs          (646 lines)  Message types, signing, verification
├── runner.rs            (832 lines)  Main network orchestrator
├── transport.rs         (765 lines)  TCP/TLS with connection pooling
├── wire.rs              (639 lines)  Binary wire protocol (HELX magic, CRC32)
├── gossip.rs            (555 lines)  Epidemic gossip with LRU dedup
├── discovery.rs         (374 lines)  Basic peer discovery
├── mdns_discovery.rs    (443 lines)  mDNS local discovery
├── sync.rs              (387 lines)  State synchronization
├── eclipse.rs           (859 lines)  Eclipse attack prevention
├── partition_detect.rs  (1027 lines) Network partition detection
├── rate_limit.rs        (827 lines)  Token bucket rate limiting
├── reputation.rs        (700 lines)  Multi-dimensional peer reputation
└── sybil.rs             (671 lines)  Stake-weighted Sybil resistance
```

### Key Types

| Type | File | Purpose |
|------|------|---------|
| `NetworkRunner` | runner.rs | Wires transport + gossip + discovery + security into receive loop |
| `NetworkMessage` | messages.rs | Signed message envelope with ed25519 (crypto-sign feature) |
| `MessagePayload` | messages.rs | Enum of all message types (Discovery/Training/Gradient/Sync/Heartbeat/Consensus) |
| `TcpTransport` | transport.rs | TCP/TLS connections with length-prefixed framing |
| `ConnectionPool` | transport.rs | Peer connection management |
| `WireCodec` | wire.rs | Binary codec with HELX magic, version, CRC32 |
| `GossipProtocol` | gossip.rs | Epidemic gossip with configurable fanout/TTL |
| `PeerDiscovery` | discovery.rs | Join/response peer exchange |
| `MdnsDiscovery` | mdns_discovery.rs | mDNS-based local peer discovery |
| `EclipseResistantPeerManager` | eclipse.rs | Diverse peer sourcing by IP prefix/ASN |
| `PartitionDetector` | partition_detect.rs | BFS-based partition group detection |
| `RateLimiter` | rate_limit.rs | Token bucket with auto-blacklisting |
| `ReputationManager` | reputation.rs | Multi-dimension scoring with decay |
| `SybilResistantSelector` | sybil.rs | Stake-weighted peer selection |

### Data Flow

```
Inbound:
  TCP/TLS → WireCodec.decode() → NetworkRunner.receive_loop()
    → verify_signature() → rate_limiter.check() → gossip.process()
    → dispatch to handler (training, gradient, sync, heartbeat, consensus)

Outbound:
  handler → gossip.broadcast()/unicast() → outbound_queue
    → WireCodec.encode() → TcpTransport.send()
```

### Dependencies

- Internal: `helix-core` (types), `helix-mpc` (MPCError)
- External: `tokio` (async), `tokio-rustls` (TLS), `mdns-sd` (mDNS), `bincode` (wire), `sha2` (message hashing), `ed25519-dalek` (signing, feature-gated), `dashmap` (concurrent maps)

## Per-Module Analysis

### `messages.rs` — Message Types & Signing

**What it does**: Defines `NetworkMessage` as the signed envelope and `MessagePayload` as the discriminated union of all message types. Signing uses ed25519 over SHA-256 hash of canonical fields (sender, payload, timestamp, nonce).

**Strengths**:
- Clean separation of transport envelope from payload (`messages.rs:22-55`)
- ed25519 signing with `crypto-sign` feature gate (`messages.rs:64-97`) — messages are authenticated when the feature is enabled
- `PeerKeyRegistry` for tracking peer public keys (`messages.rs:125-165`)
- Nonce and timestamp included in signed data, preventing replay

**Weaknesses**:
- **No timestamp validation** (`messages.rs:85`): Timestamp is signed but never checked for staleness. An attacker replaying a legitimately signed message within the nonce window would succeed.
  - **Impact**: Replay attacks possible within gossip TTL window
  - **Fix**: Add `MAX_MESSAGE_AGE` check in `verify_signature()`, reject messages older than e.g. 60 seconds
- **PeerKeyRegistry unbounded** (`messages.rs:130`): `HashMap<PeerId, PublicKey>` grows without bound as peers come and go.
  - **Impact**: Memory leak in long-running nodes
  - **Fix**: Use LRU cache or tie to active peer set with eviction on disconnect

**Tests**: 10 tests covering sign/verify roundtrip, payload serialization, nonce generation. Good coverage.

### `runner.rs` — Network Runner

**What it does**: `NetworkRunner` is the central orchestrator. Its `receive_loop` deserializes incoming messages, verifies signatures, checks rate limits, processes through gossip dedup, then dispatches to the appropriate handler based on `MessagePayload` variant.

**Strengths**:
- Builder pattern for clean construction (`runner.rs:45-120`)
- Correct pipeline: deserialize → verify sig → rate limit → gossip dedup → dispatch (`runner.rs:280-420`)
- Signature verification statistics tracked (`runner.rs:445-460`)

**Weaknesses**:
- **Single receive loop** (`runner.rs:280`): One `tokio::select!` loop handles all connections. No prioritization between message types.
  - **Impact**: Heartbeat/consensus messages delayed by flood of gradient data
  - **Fix**: Separate channels per message priority, process high-priority first
- **No graceful shutdown** (`runner.rs:280-420`): The receive loop runs until the channel closes. No shutdown signal or drain period.
  - **Impact**: In-flight messages lost on shutdown
  - **Fix**: Add `CancellationToken` with drain period before exit
- **Error logging but no action** (`runner.rs:350-380`): Signature verification failures are logged but the peer isn't penalized via reputation system.
  - **Impact**: Malicious peers can spam invalid signatures without consequence
  - **Fix**: Call `reputation_manager.record_event(peer, InvalidMessage)` on verification failure

**Tests**: 2 tests (builder, basic message routing). Under-tested relative to complexity.

### `transport.rs` — TCP/TLS Transport

**What it does**: TCP transport with optional TLS (self-signed or CA), 4-byte length-prefixed framing, and `ConnectionPool` for managing connections by peer ID.

**Strengths**:
- Length-prefixed framing is correct and simple (`transport.rs:180-220`)
- TLS support with both self-signed and proper certificate modes (`transport.rs:272-340`)
- Connection pooling avoids reconnection overhead (`transport.rs:400-460`)

**Weaknesses**:
- **Self-signed TLS in production path** (`transport.rs:272-302`): `generate_self_signed_cert()` creates TLS that doesn't prevent MitM — any attacker can also generate a self-signed cert.
  - **Impact**: MitM possible on first connection (no TOFU or pinning)
  - **Fix**: Implement certificate pinning (store expected peer cert hashes) or mutual TLS with pre-shared certs. The TLS transport in helix-mpc (`FingerprintVerifier`) already does this correctly.
- **No connection limits** (`transport.rs:400`): `ConnectionPool` has no maximum size.
  - **Impact**: Attacker opens thousands of connections, exhausts file descriptors
  - **Fix**: Add `max_connections` to pool, reject new connections when full (rate_limit.rs has flood detection but it's not wired to transport)
- **No keepalive or health checks** (`transport.rs:400-460`): Pool doesn't detect dead connections until next send fails.
  - **Impact**: Stale connections waste resources, first send after partition fails
  - **Fix**: Add TCP keepalive (`TcpSocket::set_keepalive()`) and periodic ping

**Tests**: 3 tests (config, connection pool creation). No actual network I/O tests.

### `wire.rs` — Wire Protocol

**What it does**: Binary wire protocol with `HELX` magic bytes, version field, flags byte, CRC32 checksum, and bincode payload. Also defines `BinaryGradient` for quantized gradient serialization.

**Strengths**:
- Clean protocol design: magic + version + flags + length + CRC32 + payload (`wire.rs:25-80`)
- Custom CRC32 implementation for zero external deps (`wire.rs:400-500`)
- `BinaryGradient` with quantization scale for efficient gradient wire format (`wire.rs:300-380`)

**Weaknesses**:
- **Custom CRC32 instead of crc32fast** (`wire.rs:400-500`): Rolling your own CRC32 is ~100 lines that could be a single dependency. The implementation looks correct but is untested against edge cases.
  - **Impact**: Potential checksum bugs, slower than SIMD-accelerated crc32fast
  - **Fix**: Replace with `crc32fast::hash()` (3 lines, SIMD-optimized)
- **No payload size limit** (`wire.rs:130`): `FrameReader` reads whatever length the header says.
  - **Impact**: Attacker sends header claiming 4GB payload, OOM
  - **Fix**: Add `MAX_FRAME_SIZE` check (e.g. 64MB) before allocation. Reject oversized frames.
- **Version field unused** (`wire.rs:35`): Version is written but never checked on decode.
  - **Impact**: No forward/backward compatibility handling
  - **Fix**: Check version on decode, reject unknown versions or add migration logic

**Tests**: 6 tests covering encode/decode roundtrip, CRC validation, binary gradient. Decent coverage.

### `gossip.rs` — Gossip Protocol

**What it does**: Epidemic-style message propagation. When a message arrives, it's checked against an LRU dedup cache; if new, it's queued for forwarding to `fanout` random peers (default 6) with a TTL decrement. `LazyPushGossip` variant pushes metadata only, requiring explicit pull for full messages.

**Strengths**:
- LRU-based dedup cache with configurable capacity (`gossip.rs:65-85`) — properly evicts old entries
- Configurable fanout and max_hops prevent unbounded propagation (`gossip.rs:42-60`)
- `LazyPushGossip` for bandwidth-efficient dissemination of large messages (`gossip.rs:350-420`)
- Clean separation of gossip logic from transport (`gossip.rs:119-168`)

**Weaknesses**:
- **Random peer selection without topology awareness** (`gossip.rs:140-155`): Peers selected uniformly at random from connected set.
  - **Impact**: In clustered networks, messages may take many hops to reach distant clusters
  - **Fix**: Bias selection toward peers in different subnets/ASNs (leverage eclipse.rs diversity data)
- **No priority gossip** (`gossip.rs:119`): All messages treated equally in the outbound queue.
  - **Impact**: High-priority consensus messages delayed behind gradient floods
  - **Fix**: Add priority queue for outbound, prioritize consensus > heartbeat > training > gradient

**Tests**: 8 tests including LRU eviction, promotion, TTL enforcement, dedup. Well tested.

### `discovery.rs` — Peer Discovery

**What it does**: Basic peer discovery with known peer bootstrap, join request/response protocol, stale peer cleanup, and capability-based filtering.

**Strengths**:
- Capability filtering allows nodes to discover peers with specific roles (`discovery.rs:180-220`)
- Stale peer cleanup with configurable timeout (`discovery.rs:250-300`)

**Weaknesses**:
- **Bootstrap-only discovery** (`discovery.rs:45-80`): Relies on a known set of bootstrap peers. No DHT or structured overlay.
  - **Impact**: Bootstrap nodes are single point of failure. No peer discovery after bootstrap nodes go offline.
  - **Fix**: Implement Kademlia DHT (libp2p-kad integration behind `dht` feature flag)
- **No peer exchange protocol** (`discovery.rs:100-140`): Join response returns the responding node's full peer list, not curated recommendations.
  - **Impact**: Privacy leak (full peer list exposed) and potential for eclipse attacks (malicious node controls all returned peers)
  - **Fix**: Return random subset of peers, validate returned peers independently

**Tests**: 2 tests. Under-tested for the complexity of the discovery protocol.

### `mdns_discovery.rs` — mDNS Discovery

**What it does**: Local network peer discovery using mDNS (DNS-SD) via the `mdns-sd` crate. `CombinedDiscovery` wraps mDNS + future DHT.

**Strengths**:
- Zero-configuration local discovery — ideal for demos (`mdns_discovery.rs:40-80`)
- `CombinedDiscovery` abstraction ready for DHT integration (`mdns_discovery.rs:300-380`)
- Service type includes port and role metadata (`mdns_discovery.rs:85-120`)

**Weaknesses**:
- **Blocking mDNS recv in async context** (`mdns_discovery.rs:176-179`): Uses `spawn_blocking` for each `recv_timeout()`.
  - **Impact**: Thread pool pressure under high discovery rates; each blocked thread is a tokio blocking thread
  - **Fix**: Use a dedicated mDNS thread with a channel back to async, or use mdns-sd's built-in event loop
- **No mDNS response validation** (`mdns_discovery.rs:140-170`): All mDNS responses are trusted.
  - **Impact**: Any device on LAN can advertise as a HELIX node
  - **Fix**: Require handshake with ed25519 identity verification after mDNS discovery (connect → verify identity → accept)

**Tests**: 3 tests (service registration, combined discovery, trait impl). Adequate for demo scope.

### `eclipse.rs` — Eclipse Attack Prevention

**What it does**: Prevents eclipse attacks by maintaining peer diversity across IP prefixes, ASNs, and source categories. Uses hash bucketing, eviction scoring, and anchor peer protection.

**Strengths**:
- Multi-dimensional diversity: IP prefix, ASN, source category (`eclipse.rs:80-140`)
- Anchor peers protected from eviction (`eclipse.rs:300-350`) — long-lived trusted connections survive rotation
- Diversity score metric combining all factors (`eclipse.rs:504-528`)
- Peer rotation for churn resistance (`eclipse.rs:450-500`)

**Weaknesses**:
- **ASN lookup is simulated** (`eclipse.rs:200-230`): `lookup_asn()` uses IP prefix hashing as a proxy, not actual BGP/WHOIS data.
  - **Impact**: Eclipse protection based on fake ASN data; attacker with multiple IPs in same prefix gets different "ASNs"
  - **Fix**: Integrate `maxminddb` for GeoIP/ASN lookups, or use a simple prefix-to-ASN mapping table. For demo, current approach is adequate.
- **Hash bucket assignment is deterministic** (`eclipse.rs:160-190`): Peer's bucket is `hash(peer_id) % num_buckets`.
  - **Impact**: Attacker can grind peer IDs to target specific buckets, filling them with malicious peers
  - **Fix**: Use keyed hash (HMAC with node-specific secret) so buckets aren't predictable to attackers

**Tests**: 5 tests covering diversity scoring, anchor protection, eviction. Good coverage.

### `partition_detect.rs` — Partition Detection

**What it does**: Detects network partitions using BFS-based connected component analysis, heartbeat monitoring, and consensus agreement tracking. Recommends actions: Continue, PauseTraining, Halt.

**Strengths**:
- BFS partition group finding is algorithmically correct (`partition_detect.rs:200-280`)
- Multi-signal detection: heartbeats + gossip reach + consensus disagreement (`partition_detect.rs:350-450`)
- Graduated response: continue → pause → halt based on severity (`partition_detect.rs:500-560`)
- Detailed `PartitionReport` with affected peers and recommendations (`partition_detect.rs:40-80`)

**Weaknesses**:
- **Heartbeat-only detection** (`partition_detect.rs:300-340`): Relies on missed heartbeats to detect partitions. Slow detection (multiple missed heartbeat intervals).
  - **Impact**: Partition not detected for 30-60 seconds (typical heartbeat interval × missed threshold)
  - **Fix**: Add probe-based detection (periodic small messages to random peers, detect failures faster)
- **No partition healing** (`partition_detect.rs:500-560`): Detects partitions and recommends actions, but doesn't attempt reconnection.
  - **Impact**: Manual intervention needed to reconnect after partition heals
  - **Fix**: Add reconnection attempts when partition detected — try alternative routes, re-bootstrap
- **Connected component analysis assumes full peer list** (`partition_detect.rs:200-280`): BFS over `known_peers` adjacency, but nodes only know their direct neighbors.
  - **Impact**: Can't detect partitions in parts of the network it's not directly connected to
  - **Fix**: Aggregate partition reports from multiple peers (gossip partition status)

**Tests**: 9 tests including partition detection, BFS correctness, action recommendations. Well tested.

### `rate_limit.rs` — Rate Limiting

**What it does**: Token bucket rate limiter with per-peer and global limits, message type costs, exponential backoff cooldown, adaptive rate adjustment, connection flood detection, and auto-blacklisting.

**Strengths**:
- Token bucket with O(1) check and refill (`rate_limit.rs:92-202`)
- Per-message-type cost differentiation (`rate_limit.rs:432-437`) — gradient messages cost more than heartbeats
- Adaptive rate adjustment based on utilization (`rate_limit.rs:500-560`)
- Connection flood detection with auto-blacklist (`rate_limit.rs:600-680`)

**Weaknesses**:
- **Blacklist is permanent** (`rate_limit.rs:620`): Once blacklisted, a peer stays blacklisted forever (no TTL, no appeal).
  - **Impact**: Legitimate peers temporarily misbehaving (e.g. during catchup) permanently banned
  - **Fix**: Add TTL-based blacklist entries (e.g. 1 hour) with escalating duration for repeat offenders
- **Message type costs are hardcoded** (`rate_limit.rs:432-437`): Fixed cost map, not configurable.
  - **Impact**: Can't adjust for different deployment scenarios (e.g. higher gradient costs on bandwidth-limited networks)
  - **Fix**: Load costs from `RateLimitConfig`

**Tests**: 5 tests covering token bucket, cooldown, flood detection. Adequate.

### `reputation.rs` — Peer Reputation

**What it does**: Multi-dimensional reputation system scoring peers on Responsiveness, Validity, Bandwidth, and Uptime. Records 11 behavior event types, applies weighted scoring, time-based decay, and automatic banning below threshold.

**Strengths**:
- Four-dimension scoring provides nuanced assessment (`reputation.rs:68-79`)
- Time-based decay prevents permanent grudges (`reputation.rs:300-340`)
- 11 distinct behavior events with dimension-specific impacts (`reputation.rs:120-200`)
- Configurable ban threshold and decay rate (`reputation.rs:45-65`)

**Weaknesses**:
- **No positive reputation recovery path** (`reputation.rs:300-340`): Decay moves scores toward neutral, but a banned peer can never get unbanned (ban is checked before events are processed).
  - **Impact**: Legitimate peers that temporarily misbehave are permanently excluded
  - **Fix**: Add unban after cooldown period + minimum positive interactions from other peers
- **Not wired to runner.rs** (`runner.rs:350`): Reputation manager exists but signature failures and other events in the receive loop don't update reputation scores.
  - **Impact**: Reputation system is decorative — events aren't actually recorded during normal operation
  - **Fix**: Wire `record_event()` calls into runner's message processing pipeline at each decision point

**Tests**: 8 tests covering scoring, decay, banning, multi-dimension. Good coverage.

### `sybil.rs` — Sybil Resistance

**What it does**: Stake-weighted peer selection to prevent Sybil attacks. Supports quadratic weighting (sqrt of stake) to limit whale influence, penalty system with decay, and fairness scoring.

**Strengths**:
- Quadratic weighting is a good balance between Sybil resistance and fairness (`sybil.rs:108-120`)
- Penalty decay prevents permanent punishment (`sybil.rs:250-290`)
- Fairness score metric for monitoring selection quality (`sybil.rs:350-400`)
- Selection cooldown prevents rapid peer cycling (`sybil.rs:420-450`)

**Weaknesses**:
- **Stake data is manually set** (`sybil.rs:80-100`): `update_stake(peer, amount)` must be called externally. No on-chain stake verification.
  - **Impact**: Peers can claim any stake amount; Sybil resistance is only as good as the caller's verification
  - **Fix**: Wire to `SCClient.get_stake()` for on-chain stake verification, or require stake proofs in handshake
- **No minimum stake enforcement** (`sybil.rs:108`): Any positive stake qualifies for selection.
  - **Impact**: Very cheap Sybil attack (1 wei per identity)
  - **Fix**: Enforce `min_stake` from contract's `models[modelId].minStake`

**Tests**: 7 tests covering selection weights, penalties, fairness. Good coverage.

### `sync.rs` — State Synchronization

**What it does**: Manages state synchronization between peers. Tracks `NetworkState` with Checkpoints, provides sync status (synced/syncing/behind), and handles state requests.

**Strengths**:
- Checkpoint-based sync with version tracking (`sync.rs:40-80`)
- Clean sync status enum with clear semantics (`sync.rs:120-150`)

**Weaknesses**:
- **No actual sync protocol** (`sync.rs:200-350`): State comparison and checkpoint management exist, but there's no pull-based sync mechanism that fetches missing state from peers.
  - **Impact**: Nodes that fall behind have no way to catch up
  - **Fix**: Implement request-response sync: detect behind → request missing checkpoints from peers → apply in order
- **No conflict resolution** (`sync.rs:200-350`): Multiple checkpoint versions can exist with no merge strategy.
  - **Impact**: Forked state possible after partition
  - **Fix**: Use highest-round-wins with proof verification for conflict resolution

**Tests**: 5 tests covering status tracking, checkpoint comparison. Adequate.

## Strengths Summary

1. **Comprehensive security suite**: Eclipse, Sybil, rate limiting, reputation, partition detection — this is unusually thorough for a prototype. Most projects skip all of these.
2. **Clean layering**: Transport → Wire → Gossip → Runner is well-separated and each layer is independently testable.
3. **Correct gossip implementation**: LRU dedup with TTL, configurable fanout, lazy push variant — textbook epidemic gossip.
4. **Binary wire protocol**: HELX magic, CRC32, bincode — efficient and debuggable (magic bytes help with protocol identification).
5. **ed25519 message authentication**: Feature-gated but complete — sign/verify with nonce prevents forgery.

## Weaknesses Summary (Prioritized)

### Critical

1. **Reputation system not wired** (`runner.rs`, `reputation.rs`): The reputation manager exists with correct logic but isn't called from the network receive loop. Peer misbehavior has no consequences.
   - **Fix**: Add `reputation_manager.record_event()` calls at each decision point in runner.rs (sig failure, rate limit hit, invalid message, etc.)

2. **No frame size limit** (`wire.rs:130`): An attacker can claim any payload size in the wire frame header, causing OOM.
   - **Fix**: Add `const MAX_FRAME_SIZE: u32 = 64 * 1024 * 1024;` check before allocating read buffer.

### High Priority

3. **Sybil stakes not verified on-chain** (`sybil.rs:80-100`): Stake amounts are trusted from caller, not verified against smart contract.
   - **Fix**: Periodically query `SCClient.get_stake()` for all peers, update SybilResistantSelector.

4. **No timestamp staleness check** (`messages.rs:85`): Signed timestamps not validated, enabling delayed replay.
   - **Fix**: Reject messages with `|now - timestamp| > MAX_AGE` (e.g. 60s).

5. **Transport connection limits missing** (`transport.rs:400`): No cap on `ConnectionPool` size.
   - **Fix**: Add `max_connections` field, reject when full, integrate with rate_limit flood detection.

### Nice to Have

6. **Custom CRC32** (`wire.rs:400-500`): ~100 lines that could be `crc32fast::hash()`.
7. **mDNS blocking recv** (`mdns_discovery.rs:176`): Dedicated thread would be cleaner.
8. **No DHT discovery** (`discovery.rs`): Bootstrap-only is fragile for production.

## Testing Assessment

| Module | Tests | Coverage | Assessment |
|--------|-------|----------|------------|
| messages.rs | 10 | High | Sign/verify, serialization, nonce |
| runner.rs | 2 | Low | Needs integration tests with mock transport |
| transport.rs | 3 | Low | No actual network I/O tests |
| wire.rs | 6 | Medium | Encode/decode/CRC roundtrip |
| gossip.rs | 8 | High | LRU, TTL, dedup, fanout |
| discovery.rs | 2 | Low | Basic join/response only |
| mdns_discovery.rs | 3 | Medium | Service registration, combined |
| eclipse.rs | 5 | Medium | Diversity, anchors, eviction |
| partition_detect.rs | 9 | High | BFS, heartbeat, actions |
| rate_limit.rs | 5 | Medium | Token bucket, flood, cooldown |
| reputation.rs | 8 | High | Scoring, decay, banning |
| sybil.rs | 7 | High | Weights, penalties, fairness |
| sync.rs | 5 | Medium | Status, checkpoints |

**Total**: ~73 tests. Security modules are well-tested. Transport and runner are under-tested.

**Missing test categories**:
- Network I/O integration tests (actual TCP connections)
- Multi-peer gossip propagation tests
- Concurrent message handling under load
- TLS handshake tests
- Partition detection with real topology simulation

## Demo Readiness

| Feature | Status | Notes |
|---------|--------|-------|
| TCP transport | Ready | Works for local demo |
| TLS | Ready | Self-signed adequate for demo |
| Gossip propagation | Ready | Tested, correct |
| mDNS discovery | Ready | Perfect for local demo |
| Message signing | Ready | crypto-sign feature |
| Wire protocol | Ready | HELX + CRC32 + bincode |
| Rate limiting | Ready | Prevents demo disruption |
| Eclipse/Sybil/Reputation | Ready | Impressive for demo, but not wired to runner |
| Partition detection | Ready | Can demo detection (not healing) |
| State sync | Partial | Status tracking works, no actual sync |
| DHT discovery | Not Ready | mDNS only for demo |

**Demo target**: 3+ nodes on local network communicating via gossip, exchanging training messages. **Achievable with current code.**

## Summary

### Health Score: **B** (73/100)

The networking stack is architecturally sound with impressive breadth — few prototypes include eclipse prevention, Sybil resistance, AND partition detection. The individual modules are well-implemented with correct algorithms (LRU gossip dedup, BFS partition detection, token bucket rate limiting, quadratic stake weighting). The main weakness is **integration**: the security modules exist but aren't fully wired into the runner's message processing pipeline, making them decorative rather than functional. The transport layer lacks connection limits and the wire protocol lacks frame size limits, creating DoS vectors. For ETHDenver demo purposes, this is more than sufficient — the local mDNS + gossip + TCP transport path works correctly.
