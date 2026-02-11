# helix-node/src/storage/ — Technical Review

## Overview

The `storage/` module provides persistence backends for HELIX node state: local file storage with atomic writes, IPFS content-addressed storage, and an Ethereum storage stub. Total: ~780 lines across 4 files.

## Architecture

### Module Tree

```
storage/
├── mod.rs        (~30 lines)   StorageBackend trait + re-exports
├── local.rs      (475 lines)   File-based storage with atomic writes
├── ipfs.rs       (262 lines)   IPFS HTTP API client
└── ethereum.rs   (15 lines)    Documentation stub
```

### Key Types

| Type | File | Purpose |
|------|------|---------|
| `StorageBackend` | mod.rs | Trait: store/fetch/delete/list with async |
| `LocalStorage` | local.rs | File-based persistence with integrity sidecars |
| `IpfsStorage` | ipfs.rs | IPFS HTTP API client (add/cat/pin/unpin) |

### Data Flow

```
Store: data → serialize → write to temp file → SHA-256 sidecar → atomic rename
Fetch: path → read file → verify SHA-256 sidecar → deserialize → data
```

## Per-Module Analysis

### `mod.rs` — StorageBackend Trait

**What it does**: Defines the `StorageBackend` async trait with methods: `store(key, data) → Result<()>`, `fetch(key) → Result<Vec<u8>>`, `delete(key) → Result<()>`, `list(prefix) → Result<Vec<String>>`.

**Strengths**:
- Clean trait abstraction allows swapping backends (`mod.rs:10-25`)
- Async methods for non-blocking I/O

**Weaknesses**:
- **No `exists()` method** (`mod.rs:10-25`): Must `fetch()` to check if key exists, which loads full data.
  - **Fix**: Add `async fn exists(&self, key: &str) -> Result<bool>`

### `local.rs` — Local File Storage (475 lines)

**What it does**: File-based storage with atomic writes (write to `.tmp` then rename), SHA-256 integrity sidecars (`.sha256` files alongside data files), checkpoint garbage collection, and peer list persistence.

**Strengths**:
- **Atomic writes** (`local.rs:80-130`): Write to `{path}.tmp` then `std::fs::rename()` — crash-safe, no partial writes visible
- **SHA-256 integrity sidecars** (`local.rs:140-200`): Every stored file has a `.sha256` companion file. Fetch verifies hash before returning data.
  - This catches disk corruption, partial writes from other processes, and tampering
- **Checkpoint GC** (`local.rs:250-300`): Configurable retention count, deletes old checkpoints to prevent disk exhaustion
- **Peer list persistence** (`local.rs:320-400`): Stores/loads known peers for faster restart bootstrap
- **Directory creation on demand** (`local.rs:60-75`): `create_dir_all` ensures parent directories exist

**Weaknesses**:
- **No file locking** (`local.rs:80-130`): Atomic rename prevents partial reads, but two processes can write the same key simultaneously. Last write wins silently.
  - **Impact**: Concurrent nodes sharing a data directory can corrupt each other's state
  - **Fix**: Add `flock()` / `fs2::FileExt::lock_exclusive()` on the temp file before rename. Or use a lock file per data directory.
- **SHA-256 sidecar can be desynchronized** (`local.rs:140-200`): The data file and sidecar are written in two separate operations. Crash between them leaves inconsistent state.
  - **Impact**: After crash, data exists but sidecar doesn't → fetch fails with integrity error even though data is valid
  - **Fix**: Write sidecar first (or atomically with data in a single temp file), or handle missing sidecar gracefully (recompute and re-store)
- **Checkpoint GC sorts by name** (`local.rs:260-280`): Checkpoints sorted alphabetically to determine oldest. This only works if checkpoint names sort chronologically (e.g. timestamps).
  - **Impact**: Non-timestamp-based checkpoint names → wrong checkpoints deleted
  - **Fix**: Sort by file modification time (`metadata().modified()`), or enforce timestamp-based naming
- **No disk space checking** (`local.rs:80-130`): Writes proceed without checking available space.
  - **Impact**: Node crashes when disk full, potentially mid-write (though atomic rename mitigates partial corruption)
  - **Fix**: Check `fs2::available_space()` before large writes, warn when below threshold

**Tests**: 15 tests covering store/fetch/delete, integrity verification, checkpoint GC, peer persistence, atomic writes. **Well tested.**

### `ipfs.rs` — IPFS Storage (262 lines)

**What it does**: HTTP API client for IPFS. Supports `add` (store), `cat` (fetch), `pin add` (pin), `pin rm` (unpin). Uses `reqwest` for HTTP calls to a local IPFS daemon.

**Strengths**:
- Proper multipart form upload for `add` (`ipfs.rs:60-100`): Correctly uses `multipart/form-data` as required by IPFS API
- Pin management for persistence guarantees (`ipfs.rs:130-180`)
- Content-addressed storage naturally deduplicates (`ipfs.rs:40-55`)

**Weaknesses**:
- **No connection pooling** (`ipfs.rs:40-55`): Creates new `reqwest::Client` per operation (or uses default which creates new connections).
  - **Impact**: Connection setup overhead on every IPFS operation
  - **Fix**: Store `reqwest::Client` in `IpfsStorage` struct (reqwest clients pool internally)
- **No timeout configuration** (`ipfs.rs:60-100`): HTTP calls to IPFS have no explicit timeout.
  - **Impact**: Hanging IPFS daemon blocks node indefinitely
  - **Fix**: Add `timeout(Duration::from_secs(30))` to reqwest calls
- **No retry on transient IPFS failures** (`ipfs.rs:60-180`): Network errors propagated immediately.
  - **Impact**: Flaky IPFS connection causes data fetch failures
  - **Fix**: Add retry with backoff for 5xx and network errors
- **IPFS daemon URL hardcoded or env-only** (`ipfs.rs:45-55`): No integration with `NodeConfig`.
  - **Impact**: Configuration scattered
  - **Fix**: Wire IPFS URL to `NodeConfig.ipfs_url` field

**Tests**: 0 tests. **Not tested.** Requires running IPFS daemon, so integration tests behind a feature flag would be appropriate.

### `ethereum.rs` — Ethereum Storage (15 lines)

**What it does**: Documentation stub explaining that on-chain storage is handled by `SCClient` in `sc_client.rs`, not this module.

**Assessment**: This is correctly a documentation file, not dead code. The module exists to explain the architectural decision. Acceptable.

## Strengths Summary

1. **Atomic writes**: `LocalStorage` uses temp-file-then-rename, the gold standard for crash-safe file operations.
2. **Integrity verification**: SHA-256 sidecars catch corruption and tampering — unusual thoroughness for a prototype.
3. **Checkpoint GC**: Prevents disk exhaustion from accumulated checkpoints.
4. **Clean trait**: `StorageBackend` allows backend swapping without changing callers.

## Weaknesses Summary (Prioritized)

### High Priority

1. **No file locking in LocalStorage** (`local.rs:80-130`): Concurrent writers can corrupt state.
   - **Fix**: `fs2::FileExt::lock_exclusive()` on temp file.

2. **SHA-256 sidecar desync on crash** (`local.rs:140-200`): Data and sidecar written separately.
   - **Fix**: Write sidecar first, or handle missing sidecar gracefully.

3. **IPFS has no timeouts** (`ipfs.rs:60-100`): Hanging daemon blocks node.
   - **Fix**: Add 30-second timeout to reqwest calls.

### Nice to Have

4. **IPFS has 0 tests** (`ipfs.rs`): Add integration tests behind feature flag.
5. **No `exists()` on StorageBackend** (`mod.rs`): Requires full fetch to check existence.
6. **No disk space monitoring** (`local.rs`): Could warn before disk full.

## Testing Assessment

| Module | Tests | Coverage | Assessment |
|--------|-------|----------|------------|
| mod.rs | 0 | N/A | Trait definition only |
| local.rs | 15 | High | Store/fetch/delete, integrity, GC, peers |
| ipfs.rs | 0 | None | Needs IPFS daemon for integration tests |
| ethereum.rs | 0 | N/A | Documentation stub |

**Total**: 15 tests. LocalStorage is well-tested. IPFS is untested (acceptable given it requires external daemon).

**Missing tests**:
- Concurrent write behavior
- Crash recovery (sidecar desync)
- Large file handling
- Disk full handling
- IPFS integration tests (behind feature flag with mock or real daemon)

## Demo Readiness

| Feature | Status | Notes |
|---------|--------|-------|
| Local file storage | Ready | Atomic writes, integrity checks |
| Checkpoint persistence | Ready | Save/load with GC |
| Peer list persistence | Ready | Fast restart bootstrap |
| IPFS storage | Partial | Works if IPFS daemon running |
| On-chain storage | N/A | Handled by SCClient |

**Demo target**: Node persists state across restarts. **Ready with LocalStorage.**

## Summary

### Health Score: **B-** (68/100)

LocalStorage is a well-implemented file-based persistence layer with atomic writes, SHA-256 integrity verification, and checkpoint management — significantly better than typical prototype storage. The main concerns are the lack of file locking (concurrent access safety) and the SHA-256 sidecar desync window during crashes. IPFS storage is functional but lacks timeouts and tests. For demo purposes, LocalStorage handles all persistence needs reliably.
