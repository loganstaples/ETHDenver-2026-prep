# Session Module - Technical Review

**Updated**: 2026-02-11 — Critical OOM and HMAC replay issues resolved. Health score upgraded from C to B.

**Review Date**: 2026-02-10
**Files**: 11 (mod.rs, channel.rs, transport.rs, network.rs, secure_channel.rs, establishment.rs, key_rotation.rs, multiplexer.rs, party_selection.rs, manager.rs, integration_tests.rs)
**Total Lines**: ~8,280
**Health Score**: B (70-75% production-ready)

---

## 1. Overview

The `session/` module implements the complete networking and session management stack for MPC training. It handles party discovery, authenticated key exchange, encrypted channels, message multiplexing, and session lifecycle management.

**Layer Stack (bottom to top):**
```
┌─────────────────────────────────┐
│ manager.rs         (lifecycle)  │  Session phases, participant registry
├─────────────────────────────────┤
│ multiplexer.rs     (batching)   │  Stream-based multiplexing, flow control
├─────────────────────────────────┤
│ party_selection.rs (routing)    │  Latency/reliability scoring, quorum
├─────────────────────────────────┤
│ key_rotation.rs    (PFS)        │  Automatic key rotation, chain derivation
├─────────────────────────────────┤
│ establishment.rs   (handshake)  │  Commit-reveal DH key exchange
├─────────────────────────────────┤
│ secure_channel.rs  (encryption) │  AES-256-GCM with X25519 DH
├─────────────────────────────────┤
│ channel.rs / network.rs         │  LocalChannel, NetworkChannel
├─────────────────────────────────┤
│ transport.rs       (transport)  │  Local, TCP, TLS, Authenticated
└─────────────────────────────────┘
```

**Crypto primitives used:** X25519 (DH), Ed25519 (signatures), AES-256-GCM (encryption), SHA-256 (KDF, commitments), HMAC-SHA256 (message authentication), rustls (TLS 1.3).

---

## 2. File-by-File Analysis

### 2.1 transport.rs (1,712 lines) — Transport Abstractions

The foundation layer. Defines `MPCTransport` trait and four implementations.

#### MPCTransport Trait (lines 31-53)
```rust
#[async_trait]
pub trait MPCTransport: Send + Sync {
    async fn send(&self, to: &str, data: &[u8]) -> Result<()>;
    async fn recv(&self, from: &str) -> Result<Vec<u8>>;
    async fn broadcast(&self, data: &[u8]) -> Result<()>;
    fn peers(&self) -> Vec<String>;
    fn party_id(&self) -> String;
}
```

#### LocalTransport (lines 59-162)
In-memory mesh via `tokio::sync::mpsc` channels. `create_mesh()` creates full bidirectional graph. Correct and complete for testing.

#### TcpTransport (feature-gated `network-mpc`)
Binds TCP listener, connects to peers with deterministic ordering (smaller party ID connects first). Length-prefixed framing with 64 MB max message size.

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 1 | High | transport.rs (TCP) | Handshake exchange (`party_id`, `version`, `seed`) is **unauthenticated** — vulnerable to MITM if TLS not enabled | Add Ed25519 signature over handshake message |
| 2 | Medium | transport.rs (TCP) | Seed contribution not signed — handshake seed used for key derivation can be manipulated by active attacker | Sign seed with party's long-term key |

#### TlsTransport (feature-gated `network-mpc`)
Wraps TCP with rustls. Self-signed certs via rcgen with certificate fingerprint pinning.

**Strengths:**
- `FingerprintVerifier` implementing `ServerCertVerifier` (lines 708-755) — checks SHA-256 fingerprint of peer cert
- Prevents MITM even with self-signed certificates
- Proper TLS 1.3 configuration

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 3 | Medium | transport.rs (TLS) | No certificate revocation — if peer's private key is compromised, fingerprint pinning still trusts it | Add revocation list or key rotation trigger |
| 4 | Low | transport.rs (TLS) | Fingerprint stored as `[u8; 32]` — no rotation mechanism if fingerprint needs updating | Add `update_fingerprint()` method tied to key rotation |

#### AuthenticatedTransport (lines 1279-1412)
Wraps any `MPCTransport` with HMAC-SHA256 and sequence numbers.

**Strengths:**
- HMAC computed over `session_key || sequence_le_bytes || payload` (line ~1350)
- Sequence tracking: accepts `seq >= expected`, updates `expected = seq + 1` (prevents replay)
- **Constant-time comparison** (lines 1271-1277): XOR accumulation over 32 bytes
- From-seeds deterministic key derivation

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 5 | Low | transport.rs:1279-1412 | If session key reused across party pairs, identical sequences produce identical HMACs. Not exploitable in practice but violates best practice. | Derive per-peer HMAC keys: `HMAC(session_key, peer_id)` |

---

### 2.2 channel.rs — Channel Abstractions

Defines `MPCChannel` trait and `LocalChannel` (in-memory broadcast).

**Assessment:** Clean abstraction layer. `LocalChannel` properly routes messages by (sender, receiver) pair. No significant issues.

---

### 2.3 network.rs (754 lines) — TCP Network Channel

TCP/TLS network channel with HMAC authentication and framed messages.

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| C1 | ~~**CRITICAL**~~ | network.rs:333-336 | ~~**OOM vulnerability**: Reads 4-byte length prefix, then allocates `vec![0u8; len]` without checking against `max_message_size`.~~ | ✅ **RESOLVED** (2026-02-11): Added 64MB max message size check before allocation in network.rs. |
| C2 | ~~**CRITICAL**~~ | network.rs (HMAC) | ~~HMAC doesn't bind to sequence number — same message accepted multiple times. No `expected_seq` tracking per peer.~~ | ✅ **RESOLVED** (2026-02-11): HMAC now includes sequence number. Per-peer sequence tracking rejects replayed messages. |
| 6 | High | network.rs (TLS) | `use_tls` flag enables TLS but **no cert pinning** — self-signed certs not validated. TlsTransport has fingerprint pinning, but NetworkChannel's TLS doesn't use it. | Wire in `FingerprintVerifier` from TlsTransport |
| 7 | Medium | network.rs (HMAC) | HMAC computed on bincode serialization — bincode format can change between versions, causing false HMAC failures on upgrade | Sign raw wire format (length prefix + payload bytes) instead |

---

### 2.4 secure_channel.rs (304 lines) — Encrypted Channel

AES-256-GCM encrypted wrapper using X25519 DH-derived session keys.

**Architecture:**
1. Generate ephemeral X25519 key pair
2. Exchange public keys with peers (deterministic ordering by party ID)
3. Compute shared secret via DH
4. Derive AES key: `SHA-256(shared_secret || sorted_party_ids)`
5. Per-peer cipher with counter-based nonces

**Strengths:**
- Counter-based nonces prevent reuse
- Per-peer cipher isolation
- Deterministic key ordering prevents deadlock during exchange

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 8 | High | secure_channel.rs:167-183 | **Weak KDF**: Single `SHA-256(shared_secret || ids)` — no salt, no info string. Should use HKDF-SHA256 per NIST SP 800-56C. | Replace with `hkdf::Hkdf::<Sha256>::new(salt, &shared_secret).expand(info, &mut key)` |
| 9 | Medium | secure_channel.rs:185-190 | Nonce uses only 8 of 12 bytes (bytes [4..12], bytes [0..4] are zero). Reduces nonce space from 96 to 64 bits. | Use full 96-bit counter, or document why 64-bit is sufficient (2^64 messages before wrap) |
| 10 | Medium | secure_channel.rs:113-120 | `send_encrypted()` takes `&mut self` but nonce counters need concurrent access from multiple tasks. No interior mutability provided. | Use `AtomicU64` for nonce counters, or `Mutex<HashMap<peer, u64>>` |

---

### 2.5 establishment.rs (725 lines) — Session Key Exchange

Commit-reveal Diffie-Hellman key exchange with Ed25519 authentication.

**Protocol (3 phases):**
1. **Commit**: Each party generates ephemeral DH keypair + nonce, broadcasts `SHA-256(pubkey || nonce)` commitment
2. **Reveal**: Each party broadcasts `(pubkey, nonce, ed25519_signature(pubkey || nonce))`
3. **Finalize**: Verify all commitments match reveals, verify signatures, compute pairwise DH shared secrets, derive session key as `SHA-256(session_id || all_contributions)`

**Strengths:**
- Commit-reveal prevents adaptive key choice (party can't choose key based on others' keys)
- Ed25519 authentication prevents impersonation
- Timestamp freshness check
- Forward secrecy via ephemeral DH keys

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 11 | High | establishment.rs (~323) | Standard `!=` for signature verification — should use constant-time comparison to prevent timing side-channel | Use `ed25519_dalek::Signature::verify()` which is constant-time internally (may already be OK, but verify) |
| 12 | High | establishment.rs (~342) | DH public keys used in `finalize()` without re-validation — if a party sends a valid commitment + signature but for a **different** key than the one used for DH, the session key is wrong but no error raised | Re-derive pubkey from received data and compare with what was used for DH |
| 13 | Medium | establishment.rs (state machine) | Phase transitions are unilateral — any single party can call `finalize()` even if not all parties have revealed. No consensus on when phase 3 begins. | Add `all_revealed()` check before allowing finalize, or require explicit "ready" messages |
| 14 | Medium | establishment.rs (timestamps) | Timestamp freshness check uses local clock — clock skew between parties can cause false rejection. No NTP or bounded-drift tolerance. | Add configurable clock skew tolerance (e.g., 30 seconds) |
| 15 | Low | establishment.rs | Ephemeral `StaticSecret` held in memory until `finalize()` completes — longer exposure window | Zeroize as soon as DH computation is done |

---

### 2.6 key_rotation.rs (964 lines) — Automatic Key Rotation

Provides perfect forward secrecy via periodic key rotation with chain derivation.

**Architecture:**
- `VersionedKey`: Key material + version + creation timestamp. Implements `Zeroize + Drop` for secure deletion.
- `KeyRotationManager`: State machine (Idle → Initiated → Committing → Complete/Failed)
- `PFSManager`: Ephemeral secret lifecycle with `ZeroizeOnDrop`

**Key derivation chain:** `new_key = SHA-256(chain_key || fresh_randomness || purpose)`. Compromise of current key cannot reveal previous keys (forward secrecy).

**Strengths:**
- Chain derivation prevents past compromise
- Configurable intervals with high-security/performance presets
- Old key retention (configurable `key_history_depth`)
- `Zeroize` for secure memory cleanup
- Per-party pairwise key tracking

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 16 | High | key_rotation.rs (lines ~475-500) | `finalize_rotation()` can succeed with **partial acknowledgments** — consensus not enforced. One party rotates while others keep old key → messages fail. | Require `acks.count() >= threshold` (e.g., n-1 or 2/3 of parties) before finalizing |
| 17 | High | key_rotation.rs (lines ~430-450) | Rotation randomness collection has **no commitment phase** — party can observe others' randomness contributions then choose its own to bias the derived key. Unlike establishment.rs which uses commit-reveal. | Add commit-reveal for rotation randomness (reuse pattern from establishment.rs) |
| 18 | Medium | key_rotation.rs (rotation messages) | `RotationMessage` enum lacks signatures — a MITM can inject fake rotation messages to force key rotation or denial-of-service | Add Ed25519 signature to each `RotationMessage` |
| 19 | Medium | key_rotation.rs (lines ~196-236) | `RotationAcknowledgments` has 30-second timeout with no Byzantine tolerance — one slow/malicious party blocks all rotations | Add fallback: if timeout expires with >= n-1 acks, proceed; blacklist timed-out party |
| 20 | Low | key_rotation.rs (cleanup) | `retain()` drops old keys but doesn't explicitly zeroize — relies on `Drop` impl which may be optimized away by compiler | Use `zeroize::Zeroize` explicitly before removing from collection |

---

### 2.7 multiplexer.rs (802 lines) — Message Multiplexing

Stream-based multiplexing with 6 standard streams (CONTROL, BEAVER, SHARES, COMMITMENTS, VERIFICATION, GRADIENTS), per-stream batching, and flow control.

**Strengths:**
- Per-stream priority levels (CONTROL=10 highest)
- Configurable batch size and delay
- Config presets: `low_latency()` (10 msg/1ms), `default()` (100 msg/10ms), `high_throughput()` (500 msg/50ms)
- Per-stream stats (messages sent/received, batches sent)

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 21 | Medium | multiplexer.rs:394 | Compression stubbed: `compressed: None, // TODO` — `high_throughput()` config enables compression but it does nothing | Implement zstd/lz4 compression or remove the flag |
| 22 | Medium | multiplexer.rs (flow control) | `flow_ack` sent but never used to update `outstanding` counter — window-based flow control is incomplete. Sender never blocks when window is full. | Process incoming acks to decrement `outstanding`; block `send_on_stream()` when `outstanding >= window_size` |
| 23 | Medium | multiplexer.rs:443 | Failed batch deserialization silently ignored — corrupt batches are dropped with no error logging | Log error and increment `deserialization_errors` counter |
| 24 | Low | multiplexer.rs | No ordering guarantee across streams — messages from same sender on different streams can arrive out of order | Document limitation; add cross-stream sequence if needed |

---

### 2.8 party_selection.rs (805 lines) — Adaptive Party Selection

Selects parties based on latency (EMA), reliability (success/failure ratio), and availability (last-seen timestamp).

**Scoring formula:**
```
score = latency_weight * (1 - latency_ms/max_latency)
      + reliability_weight * (successes / (successes + failures))
      + availability_weight * (seen_recently ? 1.0 : 0.5)
```

**Strengths:**
- EMA smoothing for latency
- Configurable weights with presets (strict, performance)
- Exclusion with cooldown recovery
- Heartbeat-based latency measurement
- Jitter option for load balancing

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 25 | Medium | party_selection.rs (scoring) | Latency scoring is cliff-edge: `1 - latency/max` drops linearly, hits 0 at max_latency. A single outlier spike tanks the score. | Use sigmoid curve or EMA the score itself; use median not mean for latency |
| 26 | Medium | party_selection.rs (availability) | Binary 1.0/0.5 jump at 60-second mark — no gradual decay | Use exponential decay: `exp(-(now - last_seen) / half_life)` |
| 27 | Low | party_selection.rs (heartbeat) | Heartbeat responses not cryptographically verified — a malicious party can fake low latency | Add HMAC or signature to heartbeat responses |
| 28 | Low | party_selection.rs (cleanup) | `cleanup_pending_heartbeats()` only called during specific events — stale entries accumulate | Call in `update_scores()` or use TTL-based eviction |

---

### 2.9 manager.rs (500 lines) — Session Lifecycle

High-level orchestration: registration, preprocessing (Beaver triple generation), model distribution, step advancement, resharing, completion.

**Phase transitions:**
```
Preprocessing → InputSharing → Computation → OutputReconstruction → Complete
```

**Issues:**

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| 29 | High | manager.rs:81-146 | `connect()` uses deterministic seed derivation for peers (not real handshake) — scaffolding that doesn't actually authenticate peers | Wire in `establishment.rs` session establishment |
| 30 | Medium | manager.rs (TrustedDealer) | Dealer initialized with fixed seed `0xBE11C` (incremented) — deterministic for all sessions. Any party can predict all triples. | Use `OsRng` or session-derived seed |
| 31 | Medium | manager.rs (channel) | `LocalChannel` hardcoded — no way to switch to `NetworkChannel` or `MultiplexedChannel` | Accept `Box<dyn MPCChannel>` in constructor |
| 32 | Low | manager.rs (registration) | No concurrent access guards on participant `HashMap` — safe for single-threaded but unsafe if called from multiple tokio tasks | Use `DashMap` or `RwLock<HashMap>` |

---

### 2.10 integration_tests.rs (751 lines) — Test Suite

Comprehensive integration tests covering session establishment, key derivation, session expiry, state machine transitions, authentication failure, local/network channels, and performance.

**Coverage:**
- Session establishment: 2-party, 3-party, 5-party
- Key derivation consistency across parties
- Session expiry and age tracking
- Authentication: invalid signature rejection, commitment mismatch detection
- Local channel: message ordering, broadcast exclusion
- Network channel: start/shutdown, TLS support
- Performance: establishment < 100ms/party, key derivation < 100μs

**Missing tests:**
- Real TCP communication (only `start()` and TLS config tested)
- Replay attack detection
- Concurrent sender contention (HMAC ordering)
- Byzantine scenarios (bad commitments, wrong phase transitions)
- Key rotation under concurrent message sending
- Multiplexer batch deserialization failure
- Network channel OOM attack (oversized length prefix)

---

## 3. Cross-Module Analysis

### 3.1 Security Layer Gaps

The module has strong cryptographic primitives but weak protocol-level guarantees:

| Layer | Crypto | Protocol | Gap |
|-------|--------|----------|-----|
| Transport | AES-256-GCM, HMAC-SHA256 | Sequence numbers | OK |
| TLS | rustls TLS 1.3 | Cert fingerprint pinning | No revocation |
| Key Exchange | X25519, Ed25519 | Commit-reveal | Unilateral finalize |
| Key Rotation | SHA-256 chain | State machine | No consensus |
| Multiplexer | N/A | Flow control | Incomplete |
| Network Channel | HMAC | Length-prefix framing | ~~OOM, no replay protection~~ ✅ Fixed |

### 3.2 Duplicate Security Mechanisms

`NetworkChannel` and `AuthenticatedTransport` both implement HMAC authentication but **differently**:

| Feature | NetworkChannel | AuthenticatedTransport |
|---------|---------------|----------------------|
| HMAC | SHA-256(key \|\| seq \|\| msg) ✅ Fixed | SHA-256(key \|\| seq \|\| msg) |
| Replay protection | Per-peer sequence tracking ✅ Fixed | Sequence tracking |
| Constant-time comparison | Not visible | Yes (XOR accumulation) |

This duplication is dangerous — a developer might use `NetworkChannel` directly (without `AuthenticatedTransport`) and get weaker security without realizing it.

**Fix**: Remove HMAC from `NetworkChannel` and require `AuthenticatedTransport` wrapping. Or unify into a single authenticated channel.

### 3.3 Key Derivation Inconsistency

Three different KDF approaches are used:

| Component | KDF | Standard |
|-----------|-----|----------|
| secure_channel.rs | `SHA-256(shared_secret \|\| ids)` | Non-standard |
| establishment.rs | `SHA-256(session_id \|\| all_contributions)` | Non-standard |
| key_rotation.rs | `SHA-256(chain_key \|\| randomness \|\| purpose)` | Non-standard |

All three should use HKDF-SHA256 (RFC 5869) with proper salt, IKM, and info parameters.

### 3.4 Trust Model

| Component | Trust Assumption | Acceptable? |
|-----------|-----------------|-------------|
| LocalTransport | All parties on same machine | Demo only |
| TcpTransport | Network not adversarial (no MITM) | **No** — must use with TLS |
| TlsTransport | CA not compromised; fingerprint list accurate | Yes |
| AuthenticatedTransport | Session key shared securely | Yes (if establishment.rs used) |
| manager.rs | TrustedDealer honest; LocalChannel only | Demo only |

---

## 4. Security Assessment Summary

### What Works Well
- **AuthenticatedTransport**: Proper HMAC with sequence numbers and constant-time comparison
- **TlsTransport**: Certificate fingerprint pinning is a strong MITM defense
- **establishment.rs**: Commit-reveal DH is the correct approach for MPC session setup
- **key_rotation.rs**: Chain derivation provides genuine forward secrecy
- **PFSManager**: Proper zeroization of ephemeral secrets

### What's Been Fixed (2026-02-11)
- ✅ ~~**network.rs OOM** (C1)~~ — 64MB max message size check added
- ✅ ~~**network.rs replay** (C2)~~ — HMAC now includes sequence number; per-peer tracking rejects replays

### Remaining Issues
- **Unilateral session finalize**: One party can finalize before others are ready
- **Key rotation without consensus**: Parties can end up with different keys

### What Needs Hardening
- KDF standardization (HKDF everywhere)
- Rotation message authentication (signatures)
- Multiplexer flow control completion
- Clock skew tolerance in establishment
- Byzantine fault handling in acknowledgments

---

## 5. Test Coverage

| File | Tests | In integration_tests.rs | Total | Key Gaps |
|------|-------|------------------------|-------|----------|
| transport.rs | 35+ | 4 | ~39 | No TLS tests, no TCP handshake verification |
| channel.rs | ~5 | 4 | ~9 | Adequate |
| network.rs | 5 | 0 | 5 | No OOM test, no replay test, no HMAC verification test |
| secure_channel.rs | 4 | 0 | 4 | No concurrent send test, no nonce reuse test |
| establishment.rs | 0 (in file) | 8+ | 8+ | No Byzantine scenario tests |
| key_rotation.rs | 13 | 0 | 13 | No concurrent rotation test, no Byzantine test |
| multiplexer.rs | 6 | 0 | 6 | No flow control exhaustion test |
| party_selection.rs | 13 | 0 | 13 | No Byzantine latency spoofing test |
| manager.rs | 3 | 0 | 3 | No concurrent registration test |

**Total**: ~100 tests. Good coverage for happy-path scenarios; weak coverage for adversarial/edge cases.

---

## 6. Demo Readiness (ETHDenver)

**Ready:**
- `LocalTransport` + `LocalChannel`: Perfect for single-machine demos
- `establishment.rs`: Session setup works for cooperative parties
- `manager.rs`: Session lifecycle orchestration works
- `party_selection.rs`: Useful for multi-worker demos

**Partially ready:**
- `secure_channel.rs`: Works but KDF should be strengthened
- `key_rotation.rs`: Works for cooperative parties; skip for demo if rotation not needed

**Not ready for multi-machine deployment:**
- ~~`network.rs`: OOM vulnerability~~ — ✅ Fixed; 64MB max message size check in place. HMAC replay protection added.
- `multiplexer.rs`: Flow control incomplete; fine for demos with small message counts
- `TcpTransport` without TLS: Vulnerable to MITM

**Recommendation**: For ETHDenver, use `LocalTransport` (single machine, multiple threads). If multi-machine demo is needed, use `TlsTransport` + `AuthenticatedTransport` (skip `NetworkChannel`).

---

## 7. Prioritized Recommendations

### Critical (fix before any network deployment)
1. ✅ ~~**Fix network.rs OOM**~~ — **RESOLVED** (2026-02-11): Added 64MB max message size check before allocation.
2. ✅ ~~**Fix network.rs replay**~~ — **RESOLVED** (2026-02-11): HMAC now includes sequence number. Per-peer sequence tracking rejects replayed messages.

### High (fix before multi-party deployment)
3. **Standardize KDF** — Replace all `SHA-256(concat)` with HKDF-SHA256 across secure_channel.rs, establishment.rs, key_rotation.rs
4. **Add consensus to session finalize** — Require all parties to signal "ready" before any party can call finalize
5. **Add consensus to key rotation** — Require threshold acknowledgments before committing new key
6. **Sign rotation messages** — Add Ed25519 signature to prevent fake rotation injection

### Medium (improves production quality)
7. Complete multiplexer flow control (process acks, implement backpressure)
8. Add clock skew tolerance to establishment timestamp checks
9. Implement compression in multiplexer (zstd or lz4)
10. Remove duplicate HMAC between NetworkChannel and AuthenticatedTransport
11. Add certificate revocation support to TlsTransport

### Nice-to-have
12. Exponential decay for party availability scoring
13. Byzantine fault tolerance in key rotation acknowledgments
14. Cross-stream ordering guarantees in multiplexer
15. Connection pooling in NetworkChannel (currently one spawn per peer)
