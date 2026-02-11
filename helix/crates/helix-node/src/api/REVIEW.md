# helix-node/src/api/ — Technical Review

## Overview

The `api/` module provides external interfaces for interacting with a running HELIX node: an HTTP REST API (Axum), a JSON-RPC 2.0 server, and Prometheus-compatible metrics export. Total: ~1,600 lines across 4 files.

## Architecture

### Module Tree

```
api/
├── mod.rs       (~30 lines)   Re-exports
├── http.rs      (358 lines)   Axum HTTP REST API with bearer token auth
├── rpc.rs       (1134 lines)  JSON-RPC 2.0 server with 13 methods
└── metrics.rs   (75 lines)    Prometheus metrics export
```

### Key Types

| Type | File | Purpose |
|------|------|---------|
| `HttpApi` | http.rs | Axum router with auth middleware |
| `RpcServer` | rpc.rs | JSON-RPC 2.0 request handler |
| `RpcMethod` | rpc.rs | Enum of 13 supported methods |
| `MetricsExporter` | metrics.rs | Prometheus text format export |

### Data Flow

```
External client → HTTP/JSON-RPC request
  → Auth middleware (bearer token)
  → Rate limiting (per-IP)
  → Route to handler
  → Query internal state (training, network, proofs)
  → JSON response
```

## Per-Module Analysis

### `http.rs` — HTTP REST API (358 lines)

**What it does**: Axum-based HTTP API with 6 endpoints, bearer token authentication using constant-time comparison, and per-IP rate limiting.

**Endpoints**:
- `GET /health` — Node health status
- `GET /status` — Training status (round, participants, progress)
- `GET /network` — Network status (peers, connections)
- `GET /model/:id` — Model state from on-chain
- `POST /proof/submit` — Submit proof for verification
- `GET /metrics` — Prometheus metrics

**Strengths**:
- Constant-time token comparison prevents timing attacks (`http.rs:80-95`): Uses `subtle::ConstantTimeEq` or equivalent byte-by-byte comparison
- Per-IP rate limiting with configurable limits (`http.rs:120-160`)
- Clean Axum router setup with middleware layers (`http.rs:40-75`)
- CORS headers for dashboard integration (`http.rs:60-70`)

**Weaknesses**:
- **Bearer token is single shared secret** (`http.rs:80-95`): One token for all clients. No per-user authentication, no token rotation, no revocation.
  - **Impact**: Leaked token grants full API access; no way to revoke without restarting node
  - **Fix**: For demo, single token is fine. For production: JWT with expiration and refresh, or API key per client with revocation list.
- **No HTTPS** (`http.rs:40`): API server binds plain HTTP.
  - **Impact**: Bearer tokens transmitted in cleartext on network
  - **Fix**: Add TLS to Axum (axum-server with rustls) or require reverse proxy (nginx/caddy). For local demo, not critical.
- **Rate limit state is in-memory only** (`http.rs:130`): Per-IP counters in `DashMap`. Lost on restart, no persistence.
  - **Impact**: Attacker restarts attack after node restart to reset counters
  - **Fix**: Acceptable for demo. Production should use sliding window with persistence.
- **Error responses leak internal details** (`http.rs:200-350`): Error messages include Rust error strings (`.to_string()` on errors).
  - **Impact**: Information disclosure to attackers
  - **Fix**: Map internal errors to generic HTTP error codes with sanitized messages

**Tests**: 0 tests. **Not tested at all.** The HTTP API is untested despite handling authentication and external input.

### `rpc.rs` — JSON-RPC 2.0 Server (1134 lines)

**What it does**: Full JSON-RPC 2.0 implementation with 13 methods covering training status, network status, proof submission, model queries, and administrative functions.

**Methods**:
1. `helix_getTrainingStatus` — Current training round status
2. `helix_getNetworkStatus` — Peer count, connections
3. `helix_getModelState` — On-chain model state
4. `helix_submitProof` — Submit proof for verification
5. `helix_getProofStatus` — Check proof verification status
6. `helix_getRoundInfo` — Round details
7. `helix_getNodeInfo` — Node identity, role, version
8. `helix_getPeers` — Connected peer list
9. `helix_getMetrics` — Training metrics
10. `helix_startRound` — Start new training round (admin)
11. `helix_stopRound` — Stop current round (admin)
12. `helix_setVerificationPolicy` — Change verification policy (admin)
13. `helix_getVerificationPolicy` — Current verification policy

**Strengths**:
- Proper JSON-RPC 2.0 compliance (`rpc.rs:40-100`): Request ID handling, error codes, batch support
- Standard error codes: -32700 (parse), -32600 (invalid request), -32601 (method not found), -32602 (invalid params), -32603 (internal) (`rpc.rs:60-80`)
- Admin methods separated from query methods (`rpc.rs:400-600`): Admin methods require additional authorization
- Comprehensive method set covering all node functionality

**Weaknesses**:
- **Admin methods have no additional auth** (`rpc.rs:400-600`): `startRound`, `stopRound`, `setVerificationPolicy` are admin operations but use the same bearer token as read-only queries.
  - **Impact**: Read-only API users can start/stop rounds and change verification policy
  - **Fix**: Separate admin token or role-based auth. Admin methods require `admin_token` parameter.
- **No request size limit** (`rpc.rs:100-140`): JSON-RPC request body parsed without size limit.
  - **Impact**: Attacker sends 1GB JSON body → OOM
  - **Fix**: Add `axum::extract::DefaultBodyLimit::max(1_048_576)` (1MB) to router
- **Proof submission via RPC accepts raw bytes** (`rpc.rs:300-380`): Proof bytes accepted as hex string, no validation before passing to verifier.
  - **Impact**: Minimal — verifier does validation. But large payloads waste verifier resources.
  - **Fix**: Add proof size limit check (e.g. max 10KB) before forwarding to verifier
- **setVerificationPolicy allows Permissive** (`rpc.rs:550-600`): Admin can set policy to `Permissive` (accept all proofs) via RPC.
  - **Impact**: Compromised admin token → all invalid proofs accepted → invalid on-chain state
  - **Fix**: Remove `Permissive` from RPC-settable policies, or require additional confirmation for dangerous policies

**Tests**: 29 tests covering all 13 methods, error handling, batch requests, parameter validation. **Well tested.**

### `metrics.rs` — Prometheus Metrics (75 lines)

**What it does**: Exports node metrics in Prometheus text format for monitoring dashboards.

**Metrics exported**:
- `helix_training_round` — Current round number
- `helix_training_loss` — Latest loss value
- `helix_peers_connected` — Number of connected peers
- `helix_proofs_verified` — Total proofs verified
- `helix_proofs_rejected` — Total proofs rejected

**Strengths**:
- Prometheus text format is industry standard (`metrics.rs:30-60`)
- Key metrics selected for training monitoring

**Weaknesses**:
- **No histogram or summary metrics** (`metrics.rs:30-60`): Only gauges and counters. No latency distributions.
  - **Impact**: Can't monitor proof verification latency, network latency, or training step duration
  - **Fix**: Add histograms for `proof_verification_duration`, `training_step_duration`, `network_message_latency`
- **Metrics not thread-safe** (`metrics.rs:20-30`): Metrics values read from shared state without synchronization guarantees.
  - **Impact**: Minor — metrics may be slightly stale, acceptable for monitoring
  - **Fix**: Use `AtomicU64` counters or `prometheus` crate for thread-safe metrics

**Tests**: 1 test covering basic export format. **Minimal but adequate for the module size.**

## Strengths Summary

1. **Constant-time auth**: Bearer token comparison resistant to timing attacks — security-aware implementation.
2. **JSON-RPC 2.0 compliance**: Proper error codes, batch support, request ID handling.
3. **Comprehensive RPC methods**: 13 methods covering all node functionality — well-designed API surface.
4. **Per-IP rate limiting**: Prevents API abuse from individual clients.

## Weaknesses Summary (Prioritized)

### Critical

1. **HTTP API has 0 tests** (`http.rs`): Authentication, rate limiting, and all endpoints untested.
   - **Fix**: Add integration tests using `axum::test` or `reqwest` against the running server. At minimum test auth rejection, rate limiting, and each endpoint.

### High Priority

2. **Admin methods not auth-separated** (`rpc.rs:400-600`): Read-only token can execute admin operations.
   - **Fix**: Separate admin token or role-based authorization.

3. **No request body size limit** (`rpc.rs:100-140`): OOM vector via large JSON payloads.
   - **Fix**: `DefaultBodyLimit::max(1_048_576)` on the Axum router.

4. **No HTTPS** (`http.rs:40`): Tokens in cleartext.
   - **Fix**: Add TLS or document reverse proxy requirement.

### Nice to Have

5. **Single shared bearer token** (`http.rs:80`): No per-client auth.
6. **No latency histograms in metrics** (`metrics.rs`): Missing performance observability.
7. **Error messages leak internals** (`http.rs:200-350`): Sanitize for production.

## Testing Assessment

| Module | Tests | Coverage | Assessment |
|--------|-------|----------|------------|
| http.rs | 0 | None | **Critical gap** — auth untested |
| rpc.rs | 29 | High | All methods, errors, batch |
| metrics.rs | 1 | Low | Basic format only |

**Total**: 30 tests. The RPC server is well-tested, but the HTTP layer is completely untested.

**Missing tests**:
- HTTP auth rejection (invalid/missing token)
- HTTP rate limiting behavior
- HTTP CORS headers
- HTTP error response format
- Metrics thread-safety under concurrent updates
- End-to-end HTTP → RPC → response chain

## Demo Readiness

| Feature | Status | Notes |
|---------|--------|-------|
| HTTP API | Ready | Works for demo dashboard |
| Bearer auth | Ready | Single token adequate for demo |
| JSON-RPC server | Ready | All methods functional |
| Training status | Ready | Dashboard can query |
| Proof submission via API | Ready | RPC method works |
| Metrics export | Ready | Prometheus compatible |
| Admin operations | Ready | Start/stop rounds via RPC |

**Demo target**: Dashboard connects to node via HTTP/JSON-RPC, displays training progress, submits proofs. **Ready for demo.**

## Summary

### Health Score: **B-** (65/100)

The API module provides a solid external interface with proper JSON-RPC 2.0 compliance and security-conscious authentication (constant-time comparison). The RPC server is well-tested with 29 tests covering all 13 methods. The major gap is the **HTTP layer having zero tests** — the authentication, rate limiting, and endpoint routing are completely untested. Admin operations aren't auth-separated from read-only queries, creating a privilege escalation vector. For demo purposes, the API is fully functional and the dashboard can connect and query training status. For production, the HTTP layer needs TLS, proper auth separation, request size limits, and tests.
